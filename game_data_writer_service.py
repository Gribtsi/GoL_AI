import zmq

import h5py

import msgpack
import msgpack_numpy as m
import os

from config import IN_CHANNELS, BOARD_SIZE, possible_moves_total, BLACK, WHITE

m.patch()
import numpy as np
from addresses_config import DATASETS_FILEPATH, WRITER_ADDRESS, DATASET_COUNT_ADDRESS

import signal


running = True

def shutdown(signum, frame):
    print("Shutdown command")
    global running
    running = False

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

def game_data_writer_service():

    context = zmq.Context.instance()

    # 1. Сокет для приема игр от воркеров
    pull_socket = context.socket(zmq.PULL)
    pull_socket.bind(WRITER_ADDRESS)

    # 2. НОВЫЙ сокет для ответа Тренеру о количестве сэмплов
    info_socket = context.socket(zmq.REP)
    info_socket.bind(DATASET_COUNT_ADDRESS)

    # 3. Настраиваем Poller для прослушивания обоих сокетов
    poller = zmq.Poller()
    poller.register(pull_socket, zmq.POLLIN)
    poller.register(info_socket, zmq.POLLIN)

    white_wins = 0
    black_wins = 0
    draws = 0
    total_games = 0
    agent_samples = 0
    current_agent_name = ""

    print(f"HDF5 Writer запущен. Слушаю \n{WRITER_ADDRESS}\n{DATASET_COUNT_ADDRESS}")
    with h5py.File(DATASETS_FILEPATH, 'a', libver='latest') as f:

        if 'state_tensor' not in f:
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
        total_samples = d_state.shape[0]

# Начинаем с того места, где закончили в прошлый раз

    with h5py.File(DATASETS_FILEPATH, 'a', libver='latest') as f:
        f.swmr_mode = True

        # Сохраняем ссылки на датасеты для быстрой работы
        d_state = f['state_tensor']
        d_policy = f['mcts_policy']
        d_terr = f['territories']
        d_value = f['value']
        d_score = f['score']
        d_turn = f['turn']
        d_meta = f['move_meta']
        d_gameid = f['game_id']

        while running:

            socks = dict(poller.poll(timeout=500))

            if not running:
                break

            # А. Тренер спрашивает количество ходов
            if info_socket in socks and socks[info_socket] == zmq.POLLIN:
                msg = info_socket.recv_json()
                if msg.get("command") == "GET_TOTAL_SAMPLES":
                    info_socket.send_json({"total_samples": total_samples})
                else:
                    info_socket.send_json({"error": "Unknown command"})

            # Б. Воркер прислал игру
            if pull_socket in socks and socks[pull_socket] == zmq.POLLIN:
                msg_parts = pull_socket.recv_multipart()
                game_id = msg_parts[0].decode('utf-8')
                encoded_data = msg_parts[1]

                try:
                    game_data = msgpack.unpackb(encoded_data, raw=False)
                    added_samples = len(game_data['turn'])


                    if added_samples == 0:
                        continue

                    winner = game_data['winner_meta']
                    agent_name = game_data['agent_meta']

                    if winner == BLACK:
                        black_wins += 1
                    elif winner == WHITE:
                        white_wins += 1
                    else:
                        draws += 1

                    new_total = total_samples + added_samples
                    agent_samples += added_samples

                    # 4. Расширяем датасеты (resize)
                    for dset in [d_state, d_policy, d_terr, d_value, d_score, d_turn, d_meta, d_gameid]:
                        dset.resize((new_total,) + dset.shape[1:])

                    # 5. Записываем данные в хвост датасетов
                    d_state[total_samples:new_total] = game_data['state_tensor']
                    d_policy[total_samples:new_total] = game_data['mcts_policy']
                    d_terr[total_samples:new_total] = game_data['territories']
                    d_value[total_samples:new_total] = game_data['value']
                    d_score[total_samples:new_total] = game_data['score']
                    d_turn[total_samples:new_total] = game_data['turn']
                    d_meta[total_samples:new_total] = np.array(game_data['move_meta']).astype('S32')

                    # Массив game_id (размножаем id игры на количество ходов)
                    d_gameid[total_samples:new_total] = np.array([game_id] * added_samples).astype('S32')

                    # 6. СБРАСЫВАЕМ данные на диск, чтобы читатели их увидели!
                    for dset in [d_state, d_policy, d_terr, d_value, d_score, d_turn, d_meta, d_gameid]:
                        dset.flush()

                    total_samples = new_total

                    total_games += 1


                    if current_agent_name != agent_name:

                        line = (
                            f"Агент {current_agent_name} создал {total_games} игр средней длиной "
                            f"{agent_samples / total_games:.1f} ходов "
                            f"Счет Ч{black_wins}/Б{white_wins} Н{draws}"
                        )
                        print(line)

                        black_wins = 0
                        white_wins = 0
                        draws = 0
                        total_games = 0
                        agent_samples = 0
                        current_agent_name = agent_name

                except Exception as e:
                    print(f"Ошибка при записи {game_id}: {e}")

        print("Shutdown done")


if __name__ == "__main__":
    # Папка должна быть примонтирована как Docker Volume
    game_data_writer_service()
