
import time
from datetime import datetime
from typing import Tuple

from IPython.display import clear_output
from numpy.random import uniform

from game import Game
from game_data_sender import GameDataSender
from mcts_agent import MCTS_Agent
from move import move_to_dict
from config import EMPTY, DEEP_DEPTH, SHALLOW_DEPTH, UNTIL_THE_END_CHANCE, CONCEDE_AT, \
    DEEP_SEARCH_CHANCE, BOARD_SIZE, possible_moves_total, IN_CHANNELS
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
        move_dict = turn_data['move']
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
        game_id = generate_name()

    with h5py.File(h5_file_path, 'a', libver=('v110','latest')) as f:
        if 'state_tensor' not in f:
            # Задаем maxshape=(None, ...) чтобы они могли расти бесконечно.
            # chunks обязательно нужно настроить (здесь для примера беру (128, ...))
            f.create_dataset('state_tensor', shape=(0, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE), maxshape=(None, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE), chunks=(4, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE),
                             dtype='float32', compression='lzf')
            f.create_dataset('mcts_policy', shape=(0, possible_moves_total), maxshape=(None, possible_moves_total), chunks=(32, possible_moves_total), dtype='float32',
                             compression='lzf')
            f.create_dataset('territories', shape=(0, BOARD_SIZE, BOARD_SIZE), maxshape=(None, BOARD_SIZE, BOARD_SIZE), chunks=(64, BOARD_SIZE, BOARD_SIZE),
                             dtype='int8', compression='lzf')
            f.create_dataset('value', shape=(0,), maxshape=(None,), chunks=(1024,), dtype='float32')
            f.create_dataset('score', shape=(0,), maxshape=(None,), chunks=(1024,), dtype='int64')
            f.create_dataset('turn', shape=(0,), maxshape=(None,), chunks=(1024,), dtype='int32')
            f.create_dataset('move_meta', shape=(0,), maxshape=(None,), chunks=(1024,), dtype='S32')
            f.create_dataset('game_id', shape=(0,), maxshape=(None,), chunks=(1024,), dtype='S32')

        d_state = f['state_tensor']
        d_policy = f['mcts_policy']
        d_terr = f['territories']
        d_value = f['value']
        d_score = f['score']
        d_turn = f['turn']
        d_meta = f['move_meta']
        d_gameid = f['game_id']

        total_samples = d_state.shape[0]

        added_samples = len(game_data['turn'])

        if added_samples == 0:
            return

        new_total = total_samples + added_samples

        for dset in [d_state, d_policy, d_terr, d_value, d_score, d_turn, d_meta, d_gameid]:
            dset.resize((new_total,) + dset.shape[1:])

        d_state[total_samples:new_total] = game_data['state_tensor']
        d_policy[total_samples:new_total] = game_data['mcts_policy']
        d_terr[total_samples:new_total] = game_data['territories']
        d_value[total_samples:new_total] = game_data['value']
        d_score[total_samples:new_total] = game_data['score']
        d_turn[total_samples:new_total] = game_data['turn']
        d_meta[total_samples:new_total] = np.array(game_data['move_meta']).astype('S32')

        d_gameid[total_samples:new_total] = np.array([game_id] * added_samples).astype('S32')

        for dset in [d_state, d_policy, d_terr, d_value, d_score, d_turn, d_meta, d_gameid]:
            dset.flush()



def self_play(agent: MCTS_Agent, max_moves: int = 300, visualise : bool = True, extra_text : str = None) -> Tuple[list, Game]:

    game : Game = agent.game_state

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

        game.apply_delta(best_node.delta_game)

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
        games_to_generate: int,
        rl_agent: MCTS_Agent, max_moves: int,
        file_path: str, log: bool):

    start_time = time.time()

    for index in range(games_to_generate):

            text = f"ИГРА {index + 1}/{games_to_generate} | ПРОШЛО {time.time() - start_time}c."

            history, game = self_play(agent=rl_agent, max_moves=max_moves, extra_text=text, visualise=log)

            game_data = process_game_history(
                history=history,
                final_board_state=game.board.current_state,
                score=game.get_score(),
            )

            samples = len(game_data['turn'])
            if samples > 0:
                save_game_to_hdf5(file_path, game_data, generate_name())

            rl_agent.flush()


