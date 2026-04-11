
import time
from typing import Tuple

from IPython.display import clear_output
from numpy.random import uniform
from win32ctypes.pywin32.pywintypes import datetime

from game import Game
from mcts_agent import MCTS_Agent
from move import move_to_dict
from config import EMPTY, DEEP_DEPTH, SHALLOW_DEPTH, UNTIL_THE_END_CHANCE, CONCEDE_AT, \
    DEEP_SEARCH_CHANCE
import numpy as np
import json
import h5py
import uuid

from random_komi import generate_komi


def process_game_history(history: list, final_board_state: np.ndarray, score: dict) -> dict:
    """
    Подготавливает историю игры в виде словаря NumPy-массивов
    для прямой и быстрой записи в HDF5-датасет.
    """
    filtered_history = [turn for turn in history if turn.get('deep_search')]

    num_turns = len(filtered_history)

    # Заранее выделяем память под скалярные массивы
    values = np.zeros(num_turns, dtype=np.float32)
    scores = np.zeros(num_turns, dtype=np.float32)
    turns = np.arange(num_turns, dtype=np.int32)

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

    for i, turn_data in enumerate(filtered_history):

        if not turn_data['deep_search']:
            continue

        current_player = turn_data['player']

        # 1. Расчет Value и Score
        if current_player == winner:
            values[i] = 1.0
            scores[i] = margin
        elif winner is None:
            values[i] = 0.0
            scores[i] = 0
        else:
            values[i] = -1.0
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
        move_dict = move_to_dict(*turn_data['move'])
        moves_meta.append(json.dumps(move_dict).encode('utf-8'))

    # Возвращаем "колонки" данных
    return {
        'state_tensor': np.array(state_tensors, dtype=np.float32),
        'mcts_policy': np.array(mcts_policies, dtype=np.float32),
        'value': values,
        'score': scores,
        'territories': np.array(territories_list, dtype=np.float32),
        'turn': turns,
        # Сохраняем как массив байтовых строк (строковый тип, понятный HDF5)
        'move_meta': np.array(moves_meta, dtype='S')
    }


def save_game_to_hdf5(h5_file_path: str, game_data: dict, game_id: str = None):
    """
    Сохраняет обработанную партию в HDF5 базу данных.
    """
    if game_id is None:
        game_id = f"game_{uuid.uuid4().hex}"

    with h5py.File(h5_file_path, 'a') as f:
        # Создаем директорию (группу) для конкретной партии
        game_group = f.create_group(game_id)

        # Основные тензоры для обучения (используем компрессию для экономии места)
        game_group.create_dataset('state_tensor', data=game_data['state_tensor'], compression='lzf')
        game_group.create_dataset('mcts_policy', data=game_data['mcts_policy'], compression='lzf')
        game_group.create_dataset('territories', data=game_data['territories'], compression='lzf')

        # Скалярные значения и таргеты
        game_group.create_dataset('value', data=game_data['value'])
        game_group.create_dataset('score', data=game_data['score'])

        # Метаданные (переменная длина строк, поэтому указываем спец. формат h5py)
        game_group.create_dataset('turn', data=game_data['turn'])
        game_group.create_dataset('move_meta', data=game_data['move_meta'])

        # Если есть общая метадата для всей игры (не по ходам), сохраняем в атрибуты группы:
        #game_group.attrs['board_size'] = metadata['board_size']
        #game_group.attrs['komi'] = metadata['komi']
        #game_group.attrs['max_komi'] = metadata['max_komi']


def self_play(agent: MCTS_Agent, max_moves: int = 300, visualise : bool = True, extra_text : str = None) -> Tuple[list, Game]:

    game = agent.root.game_state

    komi, flat_komi = generate_komi()

    game.set_komi(komi)

    till_the_end = flat_komi or (uniform(0,1) <= UNTIL_THE_END_CHANCE)

    best_node = None
    history = []

    if visualise:
        print("=" * 60)
        print("НАЧАЛО ИГРЫ")
        print("=" * 60)

    if till_the_end:
        till_the_end_str = ' | до конца'
    else:
        till_the_end_str = ''

    while (game is None) or (not game.game_over) and (game.current_move < max_moves):

        if visualise:
            clear_output(wait=True)
            print("=" * 60)
            if extra_text is not None:
                print(extra_text)
            print(f"ХОД {game.current_move + 1}{till_the_end_str}")
            print(game.get_header_text())
            print("=" * 60)

            print(game.board.board_with_permissions_as_text(game.current_player, game.get_legal_moves_mask()))


        is_deep = (uniform(0, 1) <= DEEP_SEARCH_CHANCE)

        depth = DEEP_DEPTH if is_deep else SHALLOW_DEPTH

        if best_node is None:
            x, y, life_cycle, color, is_pass, policy, best_node = agent.search(num_simulations=depth)
        else:
            x, y, life_cycle, color, is_pass, policy, best_node = agent.search(root=best_node, num_simulations=depth)

        current_state_tensor = game.get_network_input_pytorch()

        history.append({
            'state_tensor': current_state_tensor.copy(),
            'mcts_policy': policy,
            'player': game.current_player,
            'move': move_to_dict(x, y, life_cycle, color, is_pass),
            'deep_search': is_deep
        })

        game = best_node.game_state

        if not till_the_end:
            if (best_node.mean_value > CONCEDE_AT) and (
                    game.get_current_leader() != game.current_player):  # У нас инверсия откуда-то яхз взялась, поэтому знак больше
                game.game_over = True

    if visualise:
        clear_output(wait=True)
        print("=" * 60)
        if extra_text is not None:
            print(extra_text)
        print(f"Игра окончена за {game.current_move} ходов!")
        print(game.get_winner_text())
        print("=" * 60)
        print(game.board.board_with_permissions_as_text(game.current_player, game.get_legal_moves_mask()))
        print()

        # Подсчет очков
        game.print_score()

    return history, game

def generate_self_play_games(
        games_to_generate: int, games_batch: int,
        rl_agent: MCTS_Agent, max_moves: int,
        file_path: str):

    start_time = time.time()
    for k in range(int(games_to_generate / games_batch) if games_batch < games_to_generate else 1):

        for i in range(games_batch if games_batch < games_to_generate else games_to_generate):

            text = f"ИГРА {i + 1 + k * games_batch}/{games_to_generate} | ПРОШЛО {time.time() - start_time}c."

            history, game = self_play(agent=rl_agent, max_moves=max_moves, extra_text=text)

            game_data = process_game_history(
                history=history,
                final_board_state=game.board.current_state,
                score=game.get_score(),
            )

            game_id = f"game_{i:05d}_{datetime.now()}_{uuid.uuid4().hex[:8]}"

            save_game_to_hdf5(file_path, game_data, game_id=game_id)

            rl_agent.flush()
