from dataclasses import dataclass
from datetime import datetime
import time
import msgpack
import zmq
import json
from addresses_config import WRITER_ADDRESS, get_worker_control_address, get_inference_service_address, \
    TOURNAMENT_ACK_ADDRESS
from config import PROCESSES_PER_WORKER, BOARD_SIZE, EMPTY, BLACK, GAMES_PER_TOURNAMENT_PROCESS, WHITE
from game import get_territories_from_game, get_score_from_game, get_winner_and_margin_numba, get_current_player_numba
from game_rules import GameRules
from mcts_tree import count_terminal_numba
from random_network import ZMQNetworkClient
from self_play import training_data_from_game, game_log_from_game
from mcts_agent_context import AgentContext, MOVE_MADE, update_ctx
import numpy as np
import uuid
import gc

@dataclass
class MatchAssignment:
    inference_a_index:  int
    inference_b_index:  int
    agent_a_name:       str
    agent_b_name:       str
    games_to_generate:  int
    rules_a:            GameRules
    rules_b:            GameRules

    @classmethod
    def from_dict(cls, d: dict) -> "MatchAssignment":
        return cls(
            inference_a_index=d["inference_a_index"],
            inference_b_index=d["inference_b_index"],
            agent_a_name=d["agent_a_name"],
            agent_b_name=d["agent_b_name"],
            games_to_generate=d["games_to_generate"],
            rules_a=GameRules.from_dict(d.get("rules_a", {})),
            rules_b=GameRules.from_dict(d.get("rules_b", {})),
        )

    def to_dict(self) -> dict:
        return {
            "inference_a_index":  self.inference_a_index,
            "inference_b_index":  self.inference_b_index,
            "agent_a_name":       self.agent_a_name,
            "agent_b_name":       self.agent_b_name,
            "games_to_generate":  self.games_to_generate,
            "rules_a":            self.rules_a.to_dict(),
            "rules_b":            self.rules_b.to_dict()
        }


class TournamentResultSender:
    """
    Отправляет три типа сообщений в writer_address:
        b"game_data"         — pickle, как у GameDataSender
        b"game_log"          — pickle
        b"tournament_result" — JSON
    """
    def __init__(self, writer_address: str = WRITER_ADDRESS, ack_address: str = TOURNAMENT_ACK_ADDRESS):
        import pickle as _pickle
        self._pickle = _pickle
        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.PUSH)
        self.socket.connect(writer_address)

        self.ack = ctx.socket(zmq.PUSH)   # PUSH → ТМ PULL
        self.ack.connect(ack_address)

    def send(self, game_id: str, game_data: dict,
             game_log: dict, tournament_result: dict):

        self.socket.send_multipart([
            "train_game".encode('utf-8'),
            game_id.encode('utf-8'),
            msgpack.packb(game_data, use_bin_type=True),  # ← msgpack
        ])
        self.socket.send_multipart([
            "game_log".encode('utf-8'),
            game_id.encode('utf-8'),
            msgpack.packb(game_log, use_bin_type=True),  # ← msgpack
        ])
        self.socket.send_multipart([
            "tournament_result".encode('utf-8'),
            game_id.encode('utf-8'),
            json.dumps(tournament_result).encode(),  # ← json, как и было
        ])

        self.ack.send(b"")

CMD_START = "START"
CMD_STOP  = "STOP"


def _build_tournament_result(
    ctx_black: AgentContext,
    ctx_white: AgentContext
) -> dict:
    """Определяет победителя и формирует tournament_result."""
    winner_raw, margin = get_winner_and_margin_numba(ctx_black.tree.data.game_state)


    # Чётный game_index: A=BLACK, B=WHITE
    if winner_raw == EMPTY:
        result_for_black = 0.0
    elif winner_raw == BLACK:
        result_for_black = 1.0
    else:
        result_for_black = -1.0

    return {
        "agent_black":       f"{ctx_black.client.name}",
        "rules_black":       ctx_black.rules.to_dict(),
        "agent_white":       f"{ctx_white.client.name}",
        "rules_white":       ctx_white.rules.to_dict(),
        "result_for_black":  result_for_black,
    }


def merge_histories(history_black: list, history_white: list) -> list:
    """
    Восстанавливает полную историю партии из двух половинок.
    Чёрный ходит первым (индекс 0, 2, 4...), белый — (1, 3, 5...).
    """
    total = len(history_black)
    merged = [None] * total

    for i, entry in enumerate(history_black):
        if i % 2 == 0:
            merged[i] = entry        # ходы 0, 2, 4...

    for i, entry in enumerate(history_white):
        if i % 2 != 0:
            merged[i] = entry   # ходы 1, 3, 5...

    # Убираем None на случай нечётного числа ходов (последний ход чёрного)
    return [e for e in merged if e is not None]


class GamePair:
    """
    Одна параллельная игра: ctx_black + ctx_white + метаданные.
    Заменяет одиночный AgentContext в пуле run_game_pool_worker.
    """
    __slots__ = (
        "ctx_black", "ctx_white",
        "game_index_in_match",   # глобальный индекс игры в матче (определяет цвета A/B)
        "game_start",
        "games_completed",       # сколько игр эта пара уже завершила
        "total_games",           # сколько ей назначено всего
    )

    def __init__(self, ctx_black: AgentContext, ctx_white: AgentContext,
                 game_index: int, total_games: int):
        self.ctx_black           = ctx_black
        self.ctx_white           = ctx_white
        self.game_index_in_match = game_index
        self.game_start          = time.time()
        self.games_completed     = 0
        self.total_games         = total_games


    @property
    def done(self) -> bool:
        return self.games_completed >= self.total_games

    def start_next_game(self):
        """Переключает пару на следующую игру (чередует цвета)."""
        self._init_game()

    def _init_game(self):
        self.ctx_black.init_new_game()
        self.ctx_white.init_new_game()

        if self.game_index_in_match % 2 == 0:
            pass

        self.ctx_white.tree.data.game_state.komi_norm[0] = \
            self.ctx_black.tree.data.game_state.komi_norm[0]
        self.game_start = time.time()

    def init_first_game(self):
        self._init_game()

    @property
    def active(self) -> AgentContext:
        player = self.ctx_black.root_player
        return self.ctx_black if player == BLACK else self.ctx_white

    @property
    def passive(self) -> AgentContext:
        player = self.ctx_black.root_player
        return self.ctx_white if player == BLACK else self.ctx_black

    @property
    def game_over(self) -> bool:
        return bool(self.ctx_black.tree.data.game_state.game_over[0])


def update_pair(pair: GamePair) -> int:
    """
    Шагает активного агента в паре.
    Если активный сделал ход — синхронизирует пассивного через apply_a_move.
    Возвращает True если был прогресс.
    """
    if pair.game_over:
        return False

    active  = pair.active
    passive = pair.passive

    #ime.sleep(0.1)
    #print(f"update {pair.active.client.name}")

    result = update_ctx(active)



    if result == MOVE_MADE:
        passive.apply_a_move(active.history[-1]['move'])

    return result


def try_finalize_pair(
    pair: GamePair,
    sender: "TournamentResultSender",
    process_id: int,
    agent_a_name: str,
    agent_b_name: str,
) -> bool:
    """
    Если игра в паре закончена — собирает результат, отправляет, запускает следующую.
    Возвращает True если игра была завершена.
    """
    if not pair.game_over:
        return False

    terr = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    get_territories_from_game(pair.ctx_black.tree.data.game_state, terr)
    score = get_score_from_game(pair.ctx_black.tree.data.game_state)

    game_id = (
        f"t_g{pair.game_index_in_match:05d}_p{process_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    full_history = merge_histories(pair.ctx_black.history, pair.ctx_white.history)

    game_data = training_data_from_game(
        history=full_history,
        final_board_state=terr,
        score=score,
        agent_name=f"{agent_a_name} vs {agent_b_name}",
        noise_seed=0,
    )
    game_log = game_log_from_game(
        history=full_history,
        result=score,
        agent_name=f"{agent_a_name} vs {agent_b_name}",
    )
    t_result = _build_tournament_result(
        pair.ctx_black, pair.ctx_white,
    )
    t_result["game_id"]      = game_id

    sender.send(game_id, game_data, game_log, t_result)

    pair.games_completed += 1

    print(
        f"[W{process_id}] game {pair.game_index_in_match}"
    )

    if not pair.done:
        pair.start_next_game()

    return True




def run_tournament_worker(process_id: int, writer_address: str = WRITER_ADDRESS):
    """
    Жизненный цикл:
      1. Биндимся на управляющий сокет, ждём START
      2. Получаем MatchAssignment → создаём/обновляем пул (ctx_a, ctx_b)
      3. Играем games_to_generate игр по одной (не пул, а пара контекстов)
      4. Каждую игру отправляем через TournamentResultSender
      5. Ждём следующего START (или STOP)

    Один «слот» пула здесь = одна пара (ctx_a, ctx_b).
    Параллельность внутри процесса не нужна: пары достаточно,
    т.к. каждый из N процессов работает независимо.
    """
    gc.enable()

    ctrl_address = get_worker_control_address(process_id)
    zmq_ctx = zmq.Context()
    ctrl_socket = zmq_ctx.socket(zmq.PULL)
    ctrl_socket.bind(ctrl_address)

    sender = TournamentResultSender(writer_address)

    # Отложенная инициализация контекстов — создаём при первом START

    print(f"[TournamentWorker {process_id}] ready, ctrl={ctrl_address}")

    while True:
        raw = ctrl_socket.recv_json()
        cmd = raw.get("command")

        if cmd == CMD_STOP:
            print(f"[TournamentWorker {process_id}] STOP, exiting.")
            break

        if cmd != CMD_START:
            print(f"[TournamentWorker {process_id}] unknown command: {cmd}")
            continue


        assignment = MatchAssignment.from_dict(raw["assignment"])
        rules_a    = assignment.rules_a
        rules_b    = assignment.rules_b

        print(
            f"[TournamentWorker {process_id}] START: "
            f"{assignment.agent_a_name} vs {assignment.agent_b_name}, "
            f"{assignment.games_to_generate} games, "
            f"pool={GAMES_PER_TOURNAMENT_PROCESS} pairs"
        )


        # Создаём или переиспользуем клиентов / контексты
        # Клиенты пересоздаём только если поменялись индексы инференсов
        n_pairs    = GAMES_PER_TOURNAMENT_PROCESS
        base_games = assignment.games_to_generate // n_pairs
        remainder  = assignment.games_to_generate % n_pairs

        pairs = []
        game_cursor = 0
        for slot in range(n_pairs):
            games_for_slot = base_games + (1 if slot < remainder else 0)
            if games_for_slot == 0:
                continue

            # Каждая пара — свои два клиента, свои два дерева
            ca = ZMQNetworkClient(
                host=get_inference_service_address(assignment.inference_a_index),
                name=assignment.agent_a_name,
            )
            cb = ZMQNetworkClient(
                host=get_inference_service_address(assignment.inference_b_index),
                name=assignment.agent_b_name,
            )

            ctx_a = AgentContext(process_id, slot * 2, ca, rules_a)
            ctx_b = AgentContext(process_id, slot * 2 + 1, cb, rules_b)

            if game_cursor % 2 == 0:
                ctx_black, ctx_white = ctx_a, ctx_b
            else:
                ctx_black, ctx_white = ctx_b, ctx_a

            pair = GamePair(ctx_black, ctx_white,
                            game_index=game_cursor,
                            total_games=games_for_slot)
            pair.init_first_game()
            pairs.append(pair)
            game_cursor += 1

        # ── Главный цикл матча ────────────────────────────────────────
        while any(not p.done for p in pairs):
            progress = False

            for pair in pairs:
                if pair.done:
                    continue

                progress = update_pair(pair) or progress
                try_finalize_pair(
                    pair, sender, process_id,
                    assignment.agent_a_name,
                    assignment.agent_b_name,
                )

            if not progress:
                pass  # все пары ждут сеть — yield CPU

        print(
            f"[TournamentWorker {process_id}] match done "
            f"({assignment.games_to_generate} games)."
        )
