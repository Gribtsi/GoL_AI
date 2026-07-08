
import time
from datetime import datetime
from typing import Tuple

from IPython.display import clear_output
from numpy.random import uniform
from triton.language import dtype

from config import EMPTY, DEEP_DEPTH, SHALLOW_DEPTH, UNTIL_THE_END_CHANCE, CONCEDE_AT, \
    DEEP_SEARCH_CHANCE, BOARD_SIZE, possible_moves_total, IN_CHANNELS, MIN_TURNS_BEFORE_RESIGN, MAX_MOVES_PER_GAME, \
    EXPECTED_MAX_LENGTH, GAME_LENGTH_PENALTY_WEIGHT, PUNISH_LONG_GAME, BAD_VALUES_COUNT, MAX_KOMI, NOISE_FRAC, C_PUCT, \
    Q_INIT_LOSS, symbols
import numpy as np
import h5py
import uuid

from game import get_current_player_numba, get_winner_and_margin_numba, get_network_input_from_game, get_board_text, \
    apply_delta_game_numba, get_score_from_game
from mcts_tree import reset_tree_numba, begin_search_numba, add_dirichlet_noise_numba, select_leaf_numba, \
    backpropagate_numba, expand_node_from_network_result_numba, select_action_numba, MCTS_Tree
from random_network import NetworkBase


def generate_name():
    big_number = 4000000000

    #Чтобы в алфавитной сортировке новее было выше
    reversed_time = int(big_number - time.time())

    game_id = f"game_{reversed_time}_{datetime.now()}_{uuid.uuid4().hex[:8]}"

    return game_id


def training_data_from_game(history: list, final_board_state: np.ndarray, score: dict, agent_name: str, noise_seed: int) -> dict:
    """
    Подготавливает историю игры в виде словаря NumPy-массивов
    для прямой и быстрой записи в HDF5-датасет.
    """
    game_length = len(history)

    filtered_items = [
        (i, turn) for i, turn in enumerate(history)
        if turn.get('deep_search')
    ]

    num_turns = len(filtered_items)
    all_turns = len(history)

    # Заранее выделяем память под скалярные массивы
    values = np.zeros(num_turns, dtype=np.int32)
    scores = np.zeros(num_turns, dtype=np.float32)
    turns = np.array([i for i, _ in filtered_items], dtype=np.int32)
    filtered_history = [turn for _, turn in filtered_items]

    # Списки для тензоров (размеры могут зависеть от конфигурации,
    # np.array() в конце соберет их в (N, C, H, W) или подобный формат)
    state_tensors = []
    mcts_policies = []
    mcts_opp_policies = []
    territories_list = []

    # Метаданные (например, ходы). HDF5 не любит вложенные словари Python.
    # Поэтому мы сериализуем словари в JSON-строки или сохраняем как байты.
    moves_meta = []

    winner = score.get('winner')

    margin = int(score.get('margin_no_komi', 0))

    uniform_policy = np.full(possible_moves_total, 1.0 / possible_moves_total, dtype=np.float32)

    for i, turn_data in enumerate(filtered_history):

        current_player = turn_data['player']

        # 1. Расчет Value и Score
        if current_player == winner:
            values[i] = 0
            scores[i] = margin
        elif winner == EMPTY:
            values[i] = 1
            scores[i] = 0
        else:
            values[i] = 2
            scores[i] = -margin

        # 2. Быстрый расчет территорий (векторизация NumPy вместо двойного цикла)
        territories = np.zeros_like(final_board_state, dtype=np.float32)
        territories[final_board_state == current_player] = 1.0
        territories[(final_board_state == EMPTY)] = 0.5

        # 3. Сбор многомерных тензоров
        state_tensors.append(turn_data['state_tensor'])
        mcts_policies.append(turn_data['mcts_policy'])

        if turns[i] >= all_turns - 1:
            mcts_opp_policies.append(uniform_policy)
        else:
            mcts_opp_policies.append(history[turns[i] + 1]['mcts_policy'])

        territories_list.append(territories)

        # 4. Обработка метаданных (конвертация словаря в строку для HDF5)
        moves_meta.append(turn_data['move'])

    # Возвращаем "колонки" данных
    return {
        'state_tensor': np.array(state_tensors, dtype=np.float32),
        'mcts_policy': np.array(mcts_policies, dtype=np.float32),
        'mcts_opp_policy':np.array(mcts_opp_policies, dtype=np.float32),
        'value': values,
        'score': scores,
        'territories': np.array(territories_list, dtype=np.float32),
        'turn': turns,
        # Сохраняем как массив байтовых строк (строковый тип, понятный HDF5)
        'move_meta': np.array(moves_meta, dtype=np.int32),
        'winner_meta': winner,
        'agent_meta': agent_name,
        'noise_seed_meta': noise_seed
    }

def game_log_from_game(history: list, result: dict, agent_name: str) -> dict:
    game_length = len(history)

    turns = np.arange(game_length, dtype=np.int32)
    deep_search = np.array([bool(turn.get('deep_search', False)) for turn in history], dtype=np.bool_)
    moves = np.array([i['move'] for i in history], dtype=np.int32)

    return {
        'agent_meta': agent_name.encode('utf-8'),
        'game_length': np.int32(game_length),

        'turn': turns,
        'deep_search': deep_search,
        'move': moves,

        'winner': np.int8(result.get('winner', 0)),
        'margin': np.float32(result.get('margin', 0)),
        'margin_no_komi': np.float32(result.get('margin_no_komi', 0)),

        'black_stones': np.int32(result.get('black_stones', 0)),
        'white_stones': np.int32(result.get('white_stones', 0)),
        'black_territory': np.int32(result.get('black_territory', 0)),
        'white_territory': np.int32(result.get('white_territory', 0)),

        'black_komi': np.float32(result.get('black_komi', 0)),
        'white_komi': np.float32(result.get('white_komi', 0)),
    }


def self_play(
    network: NetworkBase,
    visualise: bool = True,
    delay_between_turns: float = 0,
    temperature: float = 1.0,
    mcts_iterations: int = 400,
        komi: float = 7.5
) -> dict:
    """
    Проводит одну игру self-play с заданными параметрами.

    Args:
        network:           Экземпляр NetworkBase (sync) или AsyncNetworkBase.
        visualise:         Выводить доску после каждого хода.
        delay_between_turns: Пауза (секунды) между ходами для визуализации.
        temperature:       Температура при выборе хода (постоянная на всю игру).
        mcts_iterations:   Количество симуляций MCTS на каждый ход.

    Returns:
        (states, policies, result_meta) — тренировочные данные партии.
    """
    rng = np.random.default_rng()

    # ── инициализация дерева ──────────────────────────────────────────────
    tree = MCTS_Tree(max_nodes=mcts_iterations + 10, komi_norm=komi / MAX_KOMI)
    reset_tree_numba(tree.data)
    begin_search_numba(tree.data, tree.delta_games, root_idx=-1)

    history: list[dict] = []
    
    # ── главный цикл игры ─────────────────────────────────────────────────
    while not tree.data.game_state.game_over[0]:
        root_player = get_current_player_numba(tree.data.game_state)
        sims_done = tree.data.visit_count[tree.data.root_index[0]]

        # ── цикл симуляций MCTS ───────────────────────────────────────────
        while sims_done < mcts_iterations:
            root_idx = tree.data.root_index[0]

            if tree.data.is_expanded[root_idx]:
                add_dirichlet_noise_numba(
                    tree.data, tree.temp, tree.delta_games,
                    root_idx, NOISE_FRAC, rng
                )

            leaf_idx = select_leaf_numba(
                tree.data, tree.temp,
                tree.delta_games, tree.delta_boards,
                C_PUCT, Q_INIT_LOSS
            )

            if tree.data.game_state.game_over[0]:
                # терминальный узел — оцениваем напрямую
                winner, _ = get_winner_and_margin_numba(tree.data.game_state)
                value = 1.0 if winner == root_player else (
                    0.0 if winner == EMPTY else -1.0
                )
                tree.data.is_terminal[leaf_idx] = True
                backpropagate_numba(tree.data, tree.temp, tree.delta_games, value)
                sims_done += 1
                continue

            # ── запрос к нейросети ──────────────────────── ────────────────
            state_tensor = get_network_input_from_game(tree.data.game_state)


            policy, value, score = network.predict(state_numpy=state_tensor, game=tree, node_idx=leaf_idx)

            utility = expand_node_from_network_result_numba(
                tree.data, tree.temp, tree.delta_games,
                leaf_idx, root_player, 1.0,
                policy, value, score
            )
            backpropagate_numba(tree.data, tree.temp, tree.delta_games, utility)
            sims_done += 1

        # ── выбор хода по результатам поиска ─────────────────────────────
        encoded_move, best_node_idx = select_action_numba(
            tree.data, tree.temp, temperature, rng
        )

        # сохраняем тренировочную запись
        state_for_training = get_network_input_from_game(
            tree.data.game_state
        ).copy()
        mcts_policy = tree.temp.actions_value_2.copy()
        history.append({
            "state_tensor":      state_for_training,
            "mcts_policy": mcts_policy,
            "player":     root_player,
            "move":       encoded_move,
            'deep_search': np.random.random() > 0.5
        })


        # ── визуализация ──────────────────────────────────────────────────
        if visualise:

            turn = tree.data.game_state.current_move[0]
            player_sym = symbols.get(root_player, "?")
            clear_output()
            print(f"\n── Ход {turn + 1}, игрок: {player_sym} ──")
            print(get_board_text(tree.data.game_state))
            print(f"   Q(root) = {tree.data.mean_value[tree.data.root_index[0]]:.3f}")
            if delay_between_turns > 0:
                time.sleep(delay_between_turns)


        # ── применяем ход, переходим к следующему узлу ───────────────────
        apply_delta_game_numba(
            tree.data.game_state,
            tree.delta_boards,
            tree.delta_games,
            best_node_idx
        )
        begin_search_numba(tree.data, tree.delta_games, root_idx=best_node_idx)

    # ── итог партии ───────────────────────────────────────────────────────
    score_meta = get_score_from_game(tree.data.game_state)
    winner = score_meta["winner"]

    if visualise:
        w_sym = symbols.get(winner, "Ничья") if winner != EMPTY else "Ничья"
        print(f"\n══ Партия завершена. Победитель: {w_sym} "
              f"| Счёт: B={score_meta['black']:.1f} "
              f"W={score_meta['white']:.1f} ══")

    return training_data_from_game(history, tree.data.game_state.board.current_state, get_score_from_game(tree.data.game_state), "test", 42)