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
    ENABLE_KOMI_BALANCING, MAX_KOMI_BALANCING_DELTA, MIN_KOMI_BALANCING_DELTA, DEEP_DEPTH, SHALLOW_DEPTH, \
    DEEP_SEARCH_CHANCE, UNTIL_THE_END_CHANCE, MIN_TURNS, CONCEDE_AT
from game_data_sender import GameDataSender
from mcts_agent import MCTS_Agent
from model_manager import ModelManager
from random_komi import generate_komi
from random_network import PytorchAgentWrapper, ZMQNetworkClient
from self_play import self_play, training_data_from_game, game_log_from_game, BAD_VALUES_COUNT, all_less_than_threshold
import collections

GAMES_PER_WORKER = 32

class AgentContext:
    """Хранит состояние одной сессии self-play."""

    def __init__(self, process_id: int, agent_index: int):
        self.index = agent_index

        # Индивидуальный клиент (сокет) на каждого агента!
        host = get_inference_service_address(process_id)
        self.client = ZMQNetworkClient(host=host, name=f"ZMQClient_{process_id}_{agent_index}")

        self.agent = MCTS_Agent(temperature=1.0)
        self.agent.set_network(self.client)

        self.history = []
        self.komi = 0
        self.flat_komi = False
        self.till_the_end = False
        self.is_deep = False
        self.depth_target = 0
        self.values_cache = [0] * (BAD_VALUES_COUNT * 2)
        self.values_cache_ptr = 0
        self.current_seed_base = 0

        # Состояние MCTS
        self.search_path = []
        self.node_to_expand = None
        self.simulations_done = 0
        self.best_node = None
        self.waiting_for_network = False

        self.game_started_at = 0

    def init_new_game(self, komi_offset=0):
        self.game_started_at = time.time()

        self.agent.flush()

        rng = np.random.default_rng()
        self.current_seed_base = int(rng.integers(0, 2 ** 30))
        self.agent.set_random_seed(self.current_seed_base)

        game = self.agent.game_state
        game_rng = np.random.default_rng(self.agent.get_random_seed())
        self.komi, self.flat_komi = generate_komi(komi_offset, game_rng)
        game.set_komi(self.komi)

        game_rng2 = np.random.default_rng(self.agent.get_random_seed())
        self.till_the_end = self.flat_komi or (game_rng2.uniform(0, 1) <= UNTIL_THE_END_CHANCE)

        self.history = []
        self.values_cache = [0] * (BAD_VALUES_COUNT * 2)
        self.values_cache_ptr = 0
        self.best_node = None

        self.start_new_turn()



    def start_new_turn(self):
        self.is_deep = (np.random.uniform(0, 1) <= DEEP_SEARCH_CHANCE)
        self.depth_target = DEEP_DEPTH if self.is_deep else SHALLOW_DEPTH
        self.agent.begin_search(root=self.best_node)
        self.simulations_done = self.agent.root.visit_count
        self.waiting_for_network = False


def run_game_pool_worker(writer_address, process_id: int = 0):
    gc.enable()
    sender = GameDataSender(writer_address)

    # Инициализируем пул агентов, каждый со своим сокетом
    pool = [AgentContext(process_id, i) for i in range(GAMES_PER_WORKER)]

    # Статистика и балансировка (можно использовать вашу логику из оригинального run_worker)
    games_completed = 0
    current_komi_offset = 0

    print(f"Пул-воркер {process_id} запущен. Агентов в пуле: {GAMES_PER_WORKER}")

    # Запускаем все партии
    for ctx in pool:
        ctx.init_new_game(current_komi_offset)

    results = collections.deque(maxlen=1000)

    statistics_per = 1000
    current = 0

    while True:
        progress_made = False  # Флаг, чтобы не гонять цикл вхолостую

        for ctx in pool:
            # 1. Если агент ждет ответа от сети — проверяем готовность
            if ctx.waiting_for_network:
                if ctx.client.result_ready():
                    policy, value, score = ctx.client.get_result()

                    utility = ctx.agent._expand_node_from_network_result(ctx.node_to_expand, policy, value, score)
                    ctx.agent._backpropagate(ctx.search_path, utility)

                    ctx.simulations_done += 1
                    ctx.waiting_for_network = False
                    progress_made = True
                continue  # Переходим к следующему агенту

            # 2. Если агент не ждет сеть, проверяем, закончил ли он симуляции для хода
            if ctx.simulations_done < ctx.depth_target:
                node, search_path = ctx.agent.select_leaf()
                ctx.node_to_expand = node
                ctx.search_path = search_path

                if ctx.agent.game_state.game_over:
                    # Терминальный узел — справляемся без сети
                    value = ctx.agent._get_terminal_value(node)
                    node.is_terminal = True
                    ctx.agent._backpropagate(search_path, value)
                    ctx.simulations_done += 1
                else:
                    # Нетерминальный узел — запрашиваем сеть
                    state_tensor = ctx.agent.game_state.get_network_input_pytorch()
                    ctx.client.send_request(state_tensor)
                    ctx.waiting_for_network = True

                progress_made = True
                continue

            # 3. MCTS для хода завершен — делаем физический ход
            game = ctx.agent.game_state
            encoded_move, final_policy, best_node = ctx.agent._select_action()
            ctx.best_node = best_node

            ctx.history.append({
                'state_tensor': game.get_network_input_pytorch().copy(),
                'mcts_policy': final_policy,
                'player': game.current_player,
                'move': encoded_move,
                'deep_search': ctx.is_deep
            })

            ctx.values_cache[ctx.values_cache_ptr] = ctx.agent.root.raw_value

            if (not ctx.till_the_end) and (game.current_move > MIN_TURNS):
                if all_less_than_threshold(CONCEDE_AT, ctx.values_cache,
                                           ctx.values_cache_ptr) and game.current_player != game.get_winner():
                    game.game_over = True

            ctx.values_cache_ptr = (ctx.values_cache_ptr + 1) % (BAD_VALUES_COUNT * 2)
            game.apply_delta(best_node.delta_game)
            progress_made = True

            # 4. Проверяем, не закончилась ли игра
            if game.game_over:
                game_data = training_data_from_game(
                    history=ctx.history,
                    final_board_state=game.board.current_state,
                    score=game.get_score(),
                    agent_name=ctx.client.name,
                    noise_seed=ctx.current_seed_base
                )

                game_id = f"game_{games_completed:05d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                sender.send_game(game_id, game_data)

                game_log = game_log_from_game(
                    history=ctx.history,
                    result=game.get_score(),
                    agent_name=ctx.client.name
                )
                sender.send_game_log(game_id, game_log)

                res_str = symbols.get(game_data['winner_meta'], 'D')
                print(
                    f"[Воркер {process_id}] Игра завершена ({res_str}). За: {len(ctx.history)} ходов {(time.time() - ctx.game_started_at):.1f} сек. Агент: {ctx.index}")

                games_completed += 1

                # Запускаем новую партию на этом же агенте
                ctx.init_new_game(current_komi_offset)
            else:
                # Начинаем поиск следующего хода
                ctx.start_new_turn()

        # Если за весь проход ни один агент не продвинулся (все ждут сеть),
        # коротко спим, чтобы ядро не молотило 100% времени вхолостую

        results.append(1 if progress_made else 0)
        current += 1
        if current >= statistics_per and process_id == 0:
            print(f'{(sum(results) / len(results)):.3} доля полезных обходов')
            current = 0

        if not progress_made:
            #print("No progress")
            pass
            #time.sleep(0.001)


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

    rng = np.random.default_rng()

    index = 0

    whites, blacks, draws = 0,0,0

    current_komi_offset = 0

    threshold = 0.6

    correction_period = 20

    print(f"Воркер запущен. Инференс через {get_inference_service_address(process_id)}")

    while True:
        # 3. Генерируем 1 игру (Self-Play)
        start_time = time.time()

        current_seed_base = int(rng.integers(0, 2**30)) #Чтобы вместить модификацию сида текущим ходом

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

        game_log = game_log_from_game(
            history=history,
            result=game.get_score(),
            agent_name=rl_agent.network.name)
        sender.send_game_log(game_id, game_log)

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
        p = mp.Process(target=run_game_pool_worker, args=(args.writer, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
