import os
import time
import uuid

import multiprocessing as mp
import gc

import argparse
from datetime import datetime
import numpy as np

from addresses_config import MODEL_PROVIDER_ADDRESS, WRITER_ADDRESS, get_inference_service_address
from board import Board
from config import MAX_MOVES_PER_GAME, BLACK, WHITE, PROCESSES_PER_WORKER, symbols, EMPTY, MAX_KOMI, \
    ENABLE_KOMI_BALANCING, MAX_KOMI_BALANCING_DELTA, MIN_KOMI_BALANCING_DELTA
from game_data_sender import GameDataSender
from mcts_agent import MCTS_Agent
from model_manager import ModelManager
from random_network import PytorchAgentWrapper, ZMQNetworkClient
from rl_agent import RLAgent
from self_play import self_play, training_data_from_game



def run_worker(writer_address, process_id: int = 0):
    gc.enable()

    sender = GameDataSender(writer_address)

    # Будем сохранять скачанные модели во временную директорию ОС (в докере это /tmp)
    #temp_dir = tempfile.gettempdir()
    #model_client = ModelClient(save_dir=temp_dir)

    # ModelManager натравливаем на ту же директорию, куда скачиваются веса
    #model_manager = ModelManager(RLAgent, save_dir=temp_dir, device=device)

    #current_model_version = None
    wrapper = ZMQNetworkClient(host=get_inference_service_address(process_id))
    rl_agent = MCTS_Agent(temperature=1.0)
    rl_agent.set_network(wrapper)

    index = 0

    whites, blacks, draws = 0,0,0

    current_komi_offset = 0

    threshold = 0.6

    correction_period = 20

    print(f"Воркер запущен. Инференс через {get_inference_service_address(process_id)}")

    while True:
        # 3. Генерируем 1 игру (Self-Play)
        start_time = time.time()

        current_seed_base = np.random.randint(0, 1000000)
        rl_agent.set_random_seed(current_seed_base)

        history, game = self_play(agent=rl_agent, visualise=False, komi_offset=current_komi_offset)

        interrupted = game.interrupted()

        # 4. Формируем данные
        game_data = training_data_from_game(
            history=history,
            final_board_state=game.board.current_state,
            score=game.get_score(),
            agent_name=rl_agent.network.name,
            noise_seed=current_seed_base
        )



        if game_data['winner_meta'] == BLACK:
            blacks += 1
            res_str = symbols[BLACK]
        elif game_data['winner_meta'] == WHITE:
            whites += 1
            res_str = symbols[WHITE]
        else:
            draws += 1
            res_str = symbols[EMPTY]

        total = blacks + whites
        if ENABLE_KOMI_BALANCING and (total + draws) >= correction_period:
            if blacks / total > threshold and current_komi_offset < MAX_KOMI_BALANCING_DELTA:
                current_komi_offset += 1
            elif whites / total > threshold and current_komi_offset > MIN_KOMI_BALANCING_DELTA:
                current_komi_offset -= 1

            blacks = 0
            whites = 0
            draws = 0


        # 5. Отправляем писателю
        game_id = f"game_{index:05d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        sender.send_game(game_id, game_data)

        # 6. Очистка памяти агента (zero-allocation возврат нод в пул)
        rl_agent.flush()

        duration = time.time() - start_time
        print(f"Игра завершена за {duration:.1f} сек. Ходов: {len(history)} Поб {res_str}. Сдвиг {current_komi_offset}. Раннее завершение {interrupted}")

        index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer", type=str, default=WRITER_ADDRESS)
    parser.add_argument("--provider", type=str, default=MODEL_PROVIDER_ADDRESS)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    mp.set_start_method('fork')
    gc.freeze()

    compile_board = Board()
    compile_board.compile()

    processes = []
    for i in range(PROCESSES_PER_WORKER):
        p = mp.Process(target=run_worker, args=(args.writer, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
