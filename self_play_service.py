import os
import time
import uuid

import argparse
from datetime import datetime

from addresses_config import MODEL_PROVIDER_ADDRESS, WRITER_ADDRESS
from config import MAX_MOVES_PER_GAME
from game_data_sender import GameDataSender
from mcts_agent import MCTS_Agent
from model_manager import ModelManager
from random_network import PytorchAgentWrapper, ZMQNetworkClient
from rl_agent import RLAgent
from self_play import self_play, process_game_history



def run_worker(writer_address, provider_address, device="cuda"):
    sender = GameDataSender(writer_address)

    # Будем сохранять скачанные модели во временную директорию ОС (в докере это /tmp)
    #temp_dir = tempfile.gettempdir()
    #model_client = ModelClient(save_dir=temp_dir)

    # ModelManager натравливаем на ту же директорию, куда скачиваются веса
    #model_manager = ModelManager(RLAgent, save_dir=temp_dir, device=device)

    #current_model_version = None
    wrapper = ZMQNetworkClient()
    rl_agent = MCTS_Agent(temperature=1.0)
    rl_agent.set_network(wrapper)

    index = 0

    print("Воркер запущен. Ожидание первой модели...")

    while True:

        # 1. Проверяем, есть ли новая модель (или ждем, пока Тренер создаст первую)
        #new_version, is_updated = model_client.get_latest_model(current_model_version)

        #if new_version is None:
        #    print("Модель еще не готова, ждем 5 секунд...")
        #    time.sleep(5)
        #    continue

        # 2. Если модель обновилась (или это первый запуск), обновляем агента
        #if is_updated or not rl_agent.has_network():
        #    print(f"=== Инициализация агента с весами {new_version} ===")
        #    best_model = model_manager.load_model_weights(new_version)
        #    network = PytorchAgentWrapper(best_model, device=device)

        #    # Создаем агента заново, чтобы очистить все старые MCTS-деревья
        #    rl_agent.set_network(network)
        #    current_model_version = new_version

        # 3. Генерируем 1 игру (Self-Play)
        print(f"Игра {index:05d} стартовала {time.time()}")
        start_time = time.time()

        history, game = self_play(agent=rl_agent, max_moves=MAX_MOVES_PER_GAME, visualise=False)

        # 4. Формируем данные
        game_data = process_game_history(
            history=history,
            final_board_state=game.board.current_state,
            score=game.get_score()
        )

        # 5. Отправляем писателю
        game_id = f"game_{index:05d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        sender.send_game(game_id, game_data)

        # 6. Очистка памяти агента (zero-allocation возврат нод в пул)
        rl_agent.flush()

        duration = time.time() - start_time
        print(f"Игра завершена за {duration:.1f} сек. Ходов: {len(history)}\n")

        index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer", type=str, default=WRITER_ADDRESS)
    parser.add_argument("--provider", type=str, default=MODEL_PROVIDER_ADDRESS)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_worker(args.writer, args.provider, args.device)
