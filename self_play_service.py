import time
import uuid

import multiprocessing as mp
import gc

import argparse
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from addresses_config import MODEL_PROVIDER_ADDRESS, WRITER_ADDRESS, get_inference_service_address, \
    TOURNAMENT_ACK_ADDRESS
from config import PROCESSES_PER_WORKER, symbols, GAMES_PER_WORKER_PROCESS, BOARD_SIZE
from game import get_score_from_game, \
    get_territories_from_game
from game_data_sender import GameDataSender
from game_rules import GameRules
from mcts_agent_context import AgentContext, update_ctx
from random_network import ZMQNetworkClient, RandomNetworkClient
from self_play import training_data_from_game, game_log_from_game
import collections

from tournament_service import run_tournament_worker


def try_prepare_and_send(ctx: AgentContext, sender: GameDataSender, games_completed: int, current_komi_offset: int):
    # 4. Проверяем, не закончилась ли игра

    terr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

    get_territories_from_game(ctx.tree.data.game_state, terr)

    score = get_score_from_game(ctx.tree.data.game_state)

    if ctx.tree.data.game_state.game_over[0]:
        game_data = training_data_from_game(
            history=ctx.history,
            final_board_state=terr,
            score=score,
            agent_name=ctx.client.name,
            noise_seed=ctx.current_seed_base
        )

        game_id = f"game_{games_completed:05d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        sender.send_game(game_id, game_data)

        game_log = game_log_from_game(
            history=ctx.history,
            result=score,
            agent_name=ctx.client.name
        )
        sender.send_game_log(game_id, game_log)

        res_str = symbols.get(game_data['winner_meta'], 'D')
        print(
            f"Игра завершена {res_str}. За: {len(ctx.history)} ходов {(time.time() - ctx.game_started_at):.1f} сек. Коми: {ctx.komi}")

        games_completed += 1

        # Запускаем новую партию на этом же агенте
        ctx.init_new_game(current_komi_offset)


@dataclass
class WorkerConfig:
    process_id: int
    games_per_process: int
    sender_cls: type
    sender_args: tuple
    sender_kwargs: dict
    client_cls: type
    client_args: tuple
    client_kwargs: dict
    rules: GameRules = None

    def __post_init__(self):
        if self.rules is None:
            self.rules = GameRules.for_self_play()

    def build_sender(self):
        return self.sender_cls(*self.sender_args, **self.sender_kwargs)

    def build_client(self):
        return self.client_cls(*self.client_args, **self.client_kwargs)


def run_game_pool_worker(cfg: WorkerConfig):
    gc.enable()

    process_id = cfg.process_id



    # Инициализируем пул агентов, каждый со своим сокетом
    pool = [AgentContext(process_id, i, cfg.build_client(), cfg.rules) for i in range(cfg.games_per_process)]

    sender = cfg.build_sender()

    # Статистика и балансировка (можно использовать вашу логику из оригинального run_worker)
    games_completed = 0
    current_komi_offset = 0

    print(f"Пул-воркер {process_id} запущен. Агентов в пуле: {cfg.games_per_process}")

    # Запускаем все партии
    for ctx in pool:
        ctx.init_new_game(current_komi_offset)

    results = collections.deque(maxlen=1000)

    statistics_per = 1000
    current = 0

    while True:
        progress_made = False  # Флаг, чтобы не гонять цикл вхолостую

        for ctx in pool:
            progress_made = update_ctx(ctx) > 0 or progress_made

            try_prepare_and_send(ctx, sender, games_completed, current_komi_offset)

        # Если за весь проход ни один агент не продвинулся (все ждут сеть),
        # коротко спим, чтобы ядро не молотило 100% времени вхолостую

        results.append(1 if progress_made else 0)
        current += 1
        if current >= statistics_per and process_id == 0:
            #print(f'{(sum(results) / len(results)):.3} доля полезных обходов')
            current = 0

        if not progress_made:
            pass


def debug_worker(count: int = 1, total_iter: int = 4000):

    pool = [AgentContext(0, i, RandomNetworkClient(name="rng_net"), GameRules.for_self_play()) for i in range(count)]

    for ctx in pool:
        ctx.init_new_game(0)

    for i in range(total_iter):
        for ctx in pool:
            update_ctx(ctx)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",   type=str, default="cuda")
    parser.add_argument("--mode",     type=str, default="selfplay",
                        choices=["selfplay", "tournament"])
    args = parser.parse_args()

    # Всегда прогоняем debug перед стартом — проверяем что всё компилируется
    debug_worker(1, 400)

    mp.set_start_method("fork")
    gc.freeze()

    processes = []

    if args.mode == "selfplay":
        configs = [
            WorkerConfig(
                process_id=i,
                games_per_process=GAMES_PER_WORKER_PROCESS,
                sender_cls=GameDataSender,
                sender_args=(WRITER_ADDRESS,),
                sender_kwargs={},
                client_cls=ZMQNetworkClient,
                client_args=(),
                client_kwargs={
                    "host": get_inference_service_address(i),
                    "name": f"ZMQClient_{i}",
                },
                rules=GameRules.for_self_play(),
            )
            for i in range(PROCESSES_PER_WORKER)
        ]
        for cfg in configs:
            p = mp.Process(target=run_game_pool_worker, args=(cfg,))
            p.start()
            processes.append(p)

    else:  # tournament
        for i in range(PROCESSES_PER_WORKER):
            p = mp.Process(
                target=run_tournament_worker,
                args=(i, WRITER_ADDRESS),
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()