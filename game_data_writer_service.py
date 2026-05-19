import zmq

import h5py

import msgpack
import msgpack_numpy as m
import os

from numpy.conftest import dtype

from config import IN_CHANNELS, BOARD_SIZE, possible_moves_total, BLACK, WHITE, DEEP_SEARCH_CHANCE

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

    with h5py.File(DATASETS_FILEPATH, 'a', libver='latest') as f:
        # ---------------------------
        # TRAINING DATASETS (старые)
        # ---------------------------
        d_state = _ensure_dataset(
            f, 'state_tensor',
            shape=(0, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            maxshape=(None, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            chunks=(4, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype='float32',
            compression='lzf'
        )
        d_policy = _ensure_dataset(
            f, 'mcts_policy',
            shape=(0, possible_moves_total),
            maxshape=(None, possible_moves_total),
            chunks=(32, possible_moves_total),
            dtype='float32',
            compression='lzf'
        )
        d_terr = _ensure_dataset(
            f, 'territories',
            shape=(0, BOARD_SIZE, BOARD_SIZE),
            maxshape=(None, BOARD_SIZE, BOARD_SIZE),
            chunks=(64, BOARD_SIZE, BOARD_SIZE),
            dtype='int8',
            compression='lzf'
        )
        d_value = _ensure_dataset(
            f, 'value',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='float32'
        )
        d_score = _ensure_dataset(
            f, 'score',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int64'
        )
        d_turn = _ensure_dataset(
            f, 'turn',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int32'
        )
        d_meta = _ensure_dataset(
            f, 'move_meta',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='S64'
        )
        d_gameid = _ensure_dataset(
            f, 'game_id',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='S96'
        )
        d_noise_seed = _ensure_dataset(
            f, 'noise_seed',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int32'
        )

        total_samples = d_state.shape[0]

        # ---------------------------
        # FULL GAME LOG DATASETS
        # ---------------------------
        logs = f.require_group("game_logs")

        lg_game_id = _ensure_dataset(
            logs, 'game_id',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='S96'
        )
        lg_agent = _ensure_dataset(
            logs, 'agent_meta',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='S96'
        )
        lg_offset = _ensure_dataset(
            logs, 'offset',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int64'
        )
        lg_length = _ensure_dataset(
            logs, 'length',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int32'
        )
        lg_move_encoding = _ensure_dataset(
            logs, 'move_encoding',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='S16'
        )

        lg_winner = _ensure_dataset(
            logs, 'winner',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int8'
        )
        lg_margin = _ensure_dataset(
            logs, 'margin',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='float32'
        )
        lg_margin_no_komi = _ensure_dataset(
            logs, 'margin_no_komi',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='float32'
        )
        lg_black_total = _ensure_dataset(
            logs, 'black_total',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='float32'
        )
        lg_white_total = _ensure_dataset(
            logs, 'white_total',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='float32'
        )
        lg_neutral = _ensure_dataset(
            logs, 'neutral',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int32'
        )
        lg_black_stones = _ensure_dataset(
            logs, 'black_stones',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int32'
        )
        lg_white_stones = _ensure_dataset(
            logs, 'white_stones',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int32'
        )
        lg_black_territory = _ensure_dataset(
            logs, 'black_territory',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int32'
        )
        lg_white_territory = _ensure_dataset(
            logs, 'white_territory',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='int32'
        )
        lg_black_base = _ensure_dataset(
            logs, 'black_base',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='float32'
        )
        lg_white_base = _ensure_dataset(
            logs, 'white_base',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='float32'
        )
        lg_black_komi = _ensure_dataset(
            logs, 'black_komi',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='float32'
        )
        lg_white_komi = _ensure_dataset(
            logs, 'white_komi',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='float32'
        )
        lg_max_komi = _ensure_dataset(
            logs, 'max_komi',
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype='float32'
        )

        # flat arrays for moves
        lg_turn_flat = _ensure_dataset(
            logs, 'turn_flat',
            shape=(0,),
            maxshape=(None,),
            chunks=(4096,),
            dtype='int32'
        )
        lg_player_flat = _ensure_dataset(
            logs, 'player_flat',
            shape=(0,),
            maxshape=(None,),
            chunks=(4096,),
            dtype='int8'
        )
        lg_deep_flat = _ensure_dataset(
            logs, 'deep_search_flat',
            shape=(0,),
            maxshape=(None,),
            chunks=(4096,),
            dtype='bool'
        )
        lg_move_flat = _ensure_dataset(
            logs, 'move_flat',
            shape=(0,),
            maxshape=(None,),
            chunks=(4096,),
            dtype='int32'
        )

        total_logged_games = lg_game_id.shape[0]
        total_logged_moves = lg_turn_flat.shape[0]

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

                if len(msg_parts) == 3:
                    message_type = msg_parts[0].decode('utf-8')
                    game_id = msg_parts[1].decode('utf-8')
                    encoded_data = msg_parts[2]
                else:
                    print(f"Некорректное число фреймов: {len(msg_parts)}")
                    continue

                try:
                    payload = msgpack.unpackb(encoded_data, raw=False)

                    # --------------------------------
                    # TRAIN GAME: старый путь обучения
                    # --------------------------------
                    if message_type == "train_game":
                        added_samples = len(payload['turn'])

                        if added_samples == 0:
                            continue

                        winner = payload['winner_meta']
                        agent_name = payload['agent_meta']
                        noise_seed_base = payload['noise_seed_meta']


                        if winner == BLACK:
                            black_wins += 1
                        elif winner == WHITE:
                            white_wins += 1
                        else:
                            draws += 1

                        new_total = total_samples + added_samples
                        agent_samples += added_samples

                        for dset in [d_state, d_policy, d_terr, d_value, d_score, d_turn, d_meta, d_gameid]:
                            dset.resize((new_total,) + dset.shape[1:])

                        d_state[total_samples:new_total] = payload['state_tensor']
                        d_policy[total_samples:new_total] = payload['mcts_policy']
                        d_terr[total_samples:new_total] = payload['territories']
                        d_value[total_samples:new_total] = payload['value']
                        d_score[total_samples:new_total] = payload['score']
                        d_turn[total_samples:new_total] = payload['turn']
                        d_meta[total_samples:new_total] = np.array(payload['move_meta']).astype('S64')
                        d_gameid[total_samples:new_total] = np.array([game_id] * added_samples).astype('S96')
                        d_noise_seed[total_samples:new_total] = np.array([noise_seed_base] * added_samples).astype('int32')

                        total_samples = new_total
                        total_games += 1

                        if current_agent_name and current_agent_name != agent_name:
                            line = (
                                f"Агент {current_agent_name} создал {total_games} игр средней длиной "
                                f"{agent_samples / total_games * (1 / DEEP_SEARCH_CHANCE):.1f} ходов "
                                f"Счет Ч{black_wins}/Б{white_wins} Н{draws}"
                            )
                            print(line)

                            black_wins = 0
                            white_wins = 0
                            draws = 0
                            total_games = 0
                            agent_samples = 0

                        current_agent_name = agent_name

                    # --------------------------------
                    # FULL GAME LOG: все ходы партии
                    # --------------------------------
                    elif message_type == "game_log":
                        game_len = int(payload['game_length'])
                        if game_len == 0:
                            continue

                        new_game_count = total_logged_games + 1
                        flat_start = total_logged_moves
                        flat_end = flat_start + game_len

                        # metadata row resize
                        for dset in [
                            lg_game_id, lg_agent, lg_offset, lg_length, lg_move_encoding,
                            lg_winner, lg_margin, lg_margin_no_komi,
                            lg_black_total, lg_white_total, lg_neutral,
                            lg_black_stones, lg_white_stones,
                            lg_black_territory, lg_white_territory,
                            lg_black_base, lg_white_base,
                            lg_black_komi, lg_white_komi, lg_max_komi
                        ]:
                            dset.resize((new_game_count,))

                        # flat resize
                        for dset in [lg_turn_flat, lg_player_flat, lg_deep_flat, lg_move_flat]:
                            dset.resize((flat_end,))

                        agent_meta = payload.get('agent_meta', b'')

                        lg_game_id[total_logged_games] = np.bytes_(game_id)
                        lg_agent[total_logged_games] = np.bytes_(_decode_str(agent_meta))
                        lg_offset[total_logged_games] = flat_start
                        lg_length[total_logged_games] = game_len

                        lg_winner[total_logged_games] = payload.get('winner', 0)
                        lg_margin[total_logged_games] = payload.get('margin', 0)
                        lg_margin_no_komi[total_logged_games] = payload.get('margin_no_komi', 0)
                        lg_black_total[total_logged_games] = payload.get('black_total', 0)
                        lg_white_total[total_logged_games] = payload.get('white_total', 0)
                        lg_neutral[total_logged_games] = payload.get('neutral', 0)
                        lg_black_stones[total_logged_games] = payload.get('black_stones', 0)
                        lg_white_stones[total_logged_games] = payload.get('white_stones', 0)
                        lg_black_territory[total_logged_games] = payload.get('black_territory', 0)
                        lg_white_territory[total_logged_games] = payload.get('white_territory', 0)
                        lg_black_base[total_logged_games] = payload.get('black_base', 0)
                        lg_white_base[total_logged_games] = payload.get('white_base', 0)
                        lg_black_komi[total_logged_games] = payload.get('black_komi', 0)
                        lg_white_komi[total_logged_games] = payload.get('white_komi', 0)
                        lg_max_komi[total_logged_games] = payload.get('max_komi', 0)

                        turns = np.asarray(payload['turn'], dtype=np.int32)
                        players = np.asarray(payload['player'], dtype=np.int8)
                        deep_search = np.asarray(payload['deep_search'], dtype=np.bool_)
                        move = np.asarray(payload['move'], dtype=np.int32)

                        lg_turn_flat[flat_start:flat_end] = turns
                        lg_player_flat[flat_start:flat_end] = players
                        lg_deep_flat[flat_start:flat_end] = deep_search
                        lg_move_flat[flat_start:flat_end] = move

                        total_logged_games = new_game_count
                        total_logged_moves = flat_end

                    else:
                        print(f"Неизвестный message_type={message_type}")
                        continue

                    f.flush()

                except Exception as e:
                    print(f"Ошибка при записи {game_id}: {e}")

        print("Shutdown done")


if __name__ == "__main__":
    # Папка должна быть примонтирована как Docker Volume
    game_data_writer_service()
