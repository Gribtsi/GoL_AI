import zmq

import h5py
import argparse
import msgpack
import msgpack_numpy as m
import os
from save_data import _init_train_datasets, _init_log_datasets, _write_train_game, _write_game_log, _write_tournament_result
from config import IN_CHANNELS, BOARD_SIZE, possible_moves_total, BLACK, WHITE, DEEP_SEARCH_CHANCE
from tournament_manager import TOURNAMENT_RESULTS_PATH
import json

m.patch()
import numpy as np
from addresses_config import DATASETS_FILEPATH, WRITER_ADDRESS, DATASET_COUNT_ADDRESS, TOURNAMENT_DATASETS_FILEPATH

import signal

running = True

def shutdown(signum, frame):
    print("Shutdown command")
    global running
    running = False

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)
def game_data_writer_service(dataset_filepath = DATASETS_FILEPATH):
    context = zmq.Context.instance()

    pull_socket = context.socket(zmq.PULL)
    pull_socket.bind(WRITER_ADDRESS)

    info_socket = context.socket(zmq.REP)
    info_socket.bind(DATASET_COUNT_ADDRESS)

    poller = zmq.Poller()
    poller.register(pull_socket, zmq.POLLIN)
    poller.register(info_socket, zmq.POLLIN)

    white_wins = 0
    black_wins = 0
    draws = 0
    total_games = 0
    agent_samples = 0
    current_agent_name = ""

    def _decode_str(x):
        if isinstance(x, bytes):
            return x.decode("utf-8")
        return str(x)

    def _ensure_dataset(parent, name, **kwargs):
        if name not in parent:
            parent.create_dataset(name, **kwargs)
        return parent[name]

    print(f"HDF5 Writer запущен. Слушаю \n{WRITER_ADDRESS}\n{DATASET_COUNT_ADDRESS}")

    with h5py.File(dataset_filepath, 'a', libver='latest') as f:

        _init_train_datasets(f)
        _init_log_datasets(f)

        total_samples      = f['state_tensor'].shape[0]
        total_logged_games = f['game_logs/game_id'].shape[0]
        total_logged_moves = f['game_logs/turn_flat'].shape[0]

        f.swmr_mode = True

        while running:
            socks = dict(poller.poll(timeout=500))

            if not running:
                break

            if info_socket in socks and socks[info_socket] == zmq.POLLIN:
                msg = info_socket.recv_json()
                if msg.get("command") == "GET_TOTAL_SAMPLES":
                    info_socket.send_json({"total_samples": total_samples})
                else:
                    info_socket.send_json({"error": "Unknown command"})

            if pull_socket in socks and socks[pull_socket] == zmq.POLLIN:
                msg_parts = pull_socket.recv_multipart()

                if len(msg_parts) != 3:
                    print(f"Некорректное число фреймов: {len(msg_parts)}")
                    continue

                message_type = msg_parts[0].decode('utf-8')
                game_id      = msg_parts[1].decode('utf-8')
                encoded_data = msg_parts[2]


                try:


                    if message_type == "tournament_result":
                        payload = json.loads(encoded_data)
                        _write_tournament_result(
                            TOURNAMENT_RESULTS_PATH, payload, game_id)
                    else:
                        payload = msgpack.unpackb(encoded_data, raw=False)

                    # --------------------------------
                    # TRAIN GAME: старый путь обучения
                    # --------------------------------
                    if message_type == "train_game":
                        added = _write_train_game(f, payload, game_id)
                        if added < 0:
                            continue

                        total_samples += added
                        total_games   += 1
                        agent_samples += added
                        agent_name     = payload['agent_meta']

                        if payload['winner_meta'] == BLACK:
                            black_wins += 1
                        elif payload['winner_meta'] == WHITE:
                            white_wins += 1
                        else:
                            draws += 1

                        if current_agent_name and current_agent_name != agent_name:
                            print(
                                f"Агент {current_agent_name} создал {total_games} игр "
                                f"средней длиной "
                                f"{agent_samples / total_games * (1 / DEEP_SEARCH_CHANCE):.1f} ходов "
                                f"Счет Ч{black_wins}/Б{white_wins} Н{draws}"
                            )
                            black_wins = white_wins = draws = total_games = agent_samples = 0

                        current_agent_name = agent_name

                    elif message_type == "game_log":
                        added_games, added_turns = _write_game_log(f, payload, game_id)
                        if added_games < 0:
                            continue

                        total_logged_games += 1
                        total_logged_moves += added_turns



                    elif message_type != "tournament_result":
                        print(f"Неизвестный message_type={message_type}")
                        continue

                    f.flush()

                except Exception as e:
                    print(f"Ошибка при записи {game_id}: {e}")

        print("Shutdown done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="selfplay",
        choices=["selfplay", "tournament"],
        help="selfplay — обычный датасет; tournament — турнирный датасет + JSONL"
    )
    args = parser.parse_args()

    path = TOURNAMENT_DATASETS_FILEPATH if (args.mode == "tournament") else DATASETS_FILEPATH

    game_data_writer_service(dataset_filepath=path)