
import time
from datetime import datetime
from typing import Tuple, List

from IPython.display import clear_output
from numpy.random import uniform

from game import Game
from mcts_agent import MCTS_Agent
from move import move_to_dict
from config import EMPTY, DEEP_DEPTH, SHALLOW_DEPTH, UNTIL_THE_END_CHANCE, CONCEDE_AT, \
    DEEP_SEARCH_CHANCE, BOARD_SIZE, possible_moves_total, IN_CHANNELS, MIN_TURNS, MAX_MOVES_PER_GAME, \
    EXPECTED_MAX_LENGTH, LONG_GAME_VALUE, GAME_LENGTH_PENALTY_WEIGHT, symbols, PUNISH_LONG_GAME
import numpy as np
import json
import h5py
import uuid

from random_komi import generate_komi

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

    # Заранее выделяем память под скалярные массивы
    values = np.zeros(num_turns, dtype=np.float32)
    scores = np.zeros(num_turns, dtype=np.float32)
    turns = np.array([i for i, _ in filtered_items], dtype=np.int32)
    filtered_history = [turn for _, turn in filtered_items]

    # Списки для тензоров (размеры могут зависеть от конфигурации,
    # np.array() в конце соберет их в (N, C, H, W) или подобный формат)
    state_tensors = []
    mcts_policies = []
    territories_list = []

    # Метаданные (например, ходы). HDF5 не любит вложенные словари Python.
    # Поэтому мы сериализуем словари в JSON-строки или сохраняем как байты.
    moves_meta = []

    winner = score.get('winner')

    margin = int(score.get('margin_no_komi', 0))

    pure_value = 1
    penalty = max(0.0, (game_length - EXPECTED_MAX_LENGTH) / (MAX_MOVES_PER_GAME - EXPECTED_MAX_LENGTH)) * GAME_LENGTH_PENALTY_WEIGHT

    adjusted_value = pure_value - penalty if PUNISH_LONG_GAME else pure_value

    for i, turn_data in enumerate(filtered_history):

        current_player = turn_data['player']

        # 1. Расчет Value и Score
        if current_player == winner:
            values[i] = adjusted_value
            scores[i] = margin
        elif winner == EMPTY:
            values[i] = LONG_GAME_VALUE
            scores[i] = 0
        else:
            values[i] = -pure_value
            scores[i] = -margin

        # 2. Быстрый расчет территорий (векторизация NumPy вместо двойного цикла)
        territories = np.zeros_like(final_board_state, dtype=np.float32)
        territories[final_board_state == current_player] = 1.0
        territories[(final_board_state != current_player) & (final_board_state != EMPTY)] = -1.0

        # 3. Сбор многомерных тензоров
        state_tensors.append(turn_data['state_tensor'])
        mcts_policies.append(turn_data['mcts_policy'])
        territories_list.append(territories)

        # 4. Обработка метаданных (конвертация словаря в строку для HDF5)
        moves_meta.append(turn_data['move'])

    # Возвращаем "колонки" данных
    return {
        'state_tensor': np.array(state_tensors, dtype=np.float32),
        'mcts_policy': np.array(mcts_policies, dtype=np.float32),
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
    players = np.array([turn['player'] for turn in history], dtype=np.int8)
    deep_search = np.array([bool(turn.get('deep_search', False)) for turn in history], dtype=np.bool_)
    moves = np.array([i['move'] for i in history], dtype=np.int32)

    return {
        'agent_meta': agent_name.encode('utf-8'),
        'game_length': np.int32(game_length),

        'turn': turns,
        'player': players,
        'deep_search': deep_search,
        'move': moves,

        'winner': np.int8(result.get('winner', 0)),
        'margin': np.float32(result.get('margin', 0)),
        'margin_no_komi': np.float32(result.get('margin_no_komi', 0)),

        'black_total': np.float32(result.get('black', 0)),
        'white_total': np.float32(result.get('white', 0)),
        'neutral': np.int32(result.get('neutral', 0)),

        'black_stones': np.int32(result.get('black_stones', 0)),
        'white_stones': np.int32(result.get('white_stones', 0)),
        'black_territory': np.int32(result.get('black_territory', 0)),
        'white_territory': np.int32(result.get('white_territory', 0)),

        'black_base': np.float32(result.get('black_base', 0)),
        'white_base': np.float32(result.get('white_base', 0)),
        'black_komi': np.float32(result.get('black_komi', 0)),
        'white_komi': np.float32(result.get('white_komi', 0)),
        'max_komi': np.float32(result.get('max_komi', 0)),
    }


BAD_VALUES_COUNT = 3

def all_less_than_threshold(threshold : float, values: List[float], current_ptr : int):
    values_sum = 0
    index = current_ptr
    for i in range(BAD_VALUES_COUNT):
        values_sum += values[index]
        index = (index + 2) % (BAD_VALUES_COUNT * 2)
    values_sum /= BAD_VALUES_COUNT

    return values_sum < threshold

def self_play(agent: MCTS_Agent, visualise : bool = True, extra_text : str = None, delay: float = 0, komi_offset: float = 0, deep_search_chance = DEEP_SEARCH_CHANCE) -> Tuple[list, Game]:

    game : Game = agent.game_state

    komi, flat_komi = generate_komi(komi_offset)

    game.set_komi(komi)

    till_the_end = flat_komi or (uniform(0,1) <= UNTIL_THE_END_CHANCE)

    best_node = None
    history = []

    values_cache = [0] * BAD_VALUES_COUNT * 2
    values_cache_ptr = 0

    if visualise:
        print("=" * 60)
        print("НАЧАЛО ИГРЫ")
        print("=" * 60)

    if till_the_end:
        till_the_end_str = ' | до конца'
    else:
        till_the_end_str = ''

    legal_mask = np.zeros(possible_moves_total, dtype=np.float32)

    while (game is not None) and (not game.game_over):

        started = time.time()

        if visualise:
            clear_output(wait=True)
            print("=" * 60)
            if extra_text is not None:
                print(extra_text)
            print(f"ХОД {game.current_move + 1}{till_the_end_str} {[f'{values_cache[i]:.2f}' for i in range(values_cache_ptr-1, values_cache_ptr - 1 - BAD_VALUES_COUNT * 2, -1)]}")
            print(game.get_header_text())
            print("=" * 60)
            game.get_legal_moves_mask_preciese(legal_mask)
            print(game.board.board_with_permissions_as_text(legal_mask))


        is_deep = (uniform(0, 1) <= deep_search_chance)

        depth = DEEP_DEPTH if is_deep else SHALLOW_DEPTH

        if best_node is None:
            encoded_move, policy, best_node = agent.search(num_simulations=depth)
        else:
            encoded_move, policy, best_node = agent.search(root=best_node, num_simulations=depth)

        current_state_tensor = game.get_network_input_pytorch()

        history.append({
            'state_tensor': current_state_tensor.copy(),
            'mcts_policy': policy,
            'player': game.current_player,
            'move': encoded_move,
            'deep_search': is_deep
        })

        values_cache[values_cache_ptr] = agent.root.mean_value

        if (not till_the_end) and (game.current_move > MIN_TURNS):
            if all_less_than_threshold(CONCEDE_AT, values_cache, values_cache_ptr) and game.current_player != game.get_winner():
                game.game_over = True

        values_cache_ptr = (values_cache_ptr + 1) % (BAD_VALUES_COUNT * 2)
        game.apply_delta(best_node.delta_game)

        elapsed = time.time() - started
        if elapsed < delay:
            time.sleep(delay - elapsed)

    if visualise:
        clear_output(wait=True)
        print("=" * 60)
        if extra_text is not None:
            print(extra_text)
        print(f"Игра окончена за {game.current_move} ходов!")
        print(game.get_winner_text())
        print("=" * 60)

        game.get_legal_moves_mask_preciese(legal_mask)
        print(game.board.board_with_permissions_as_text(legal_mask))
        print()

        # Подсчет очков
        game.print_score()

    return history, game

