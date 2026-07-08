from typing import Tuple, Union

import h5py
import numpy as np
from config import IN_CHANNELS, BOARD_SIZE, possible_moves_total
import uuid
import json

def _ensure_dataset(parent, name, **kwargs):
    if name not in parent:
        parent.create_dataset(name, **kwargs)
    return parent[name]

def _init_train_datasets(f: h5py.File):
    in_ch = IN_CHANNELS
    bs    = BOARD_SIZE
    n_mv  = possible_moves_total

    _ensure_dataset(f, 'state_tensor',   shape=(0,in_ch,bs,bs), maxshape=(None,in_ch,bs,bs), chunks=(4,in_ch,bs,bs),   dtype='float32', compression='lzf')
    _ensure_dataset(f, 'mcts_policy',    shape=(0,n_mv),        maxshape=(None,n_mv),         chunks=(32,n_mv),          dtype='float32', compression='lzf')
    _ensure_dataset(f, 'mcts_opp_policy',shape=(0,n_mv),        maxshape=(None,n_mv),         chunks=(32,n_mv),          dtype='float32', compression='lzf')
    _ensure_dataset(f, 'territories',    shape=(0,bs,bs),       maxshape=(None,bs,bs),        chunks=(64,bs,bs),         dtype='float32',    compression='lzf')
    _ensure_dataset(f, 'value',      shape=(0,), maxshape=(None,), chunks=(1024,), dtype='int64')
    _ensure_dataset(f, 'score',      shape=(0,), maxshape=(None,), chunks=(1024,), dtype='int64')
    _ensure_dataset(f, 'turn',       shape=(0,), maxshape=(None,), chunks=(1024,), dtype='int32')
    _ensure_dataset(f, 'move_meta',  shape=(0,), maxshape=(None,), chunks=(1024,), dtype='S64')
    _ensure_dataset(f, 'game_id',    shape=(0,), maxshape=(None,), chunks=(1024,), dtype='S96')
    _ensure_dataset(f, 'noise_seed', shape=(0,), maxshape=(None,), chunks=(1024,), dtype='int32')


def _init_log_datasets(f: h5py.File):
    logs = f.require_group("game_logs")
    for name, dt in [
        ('game_id','S96'), ('agent_meta','S96'), ('length','int32'),
        ('winner','int8'), ('margin','float32'), ('margin_no_komi','float32'),
        ('black_stones','int32'), ('white_stones','int32'),
        ('black_territory','int32'), ('white_territory','int32'),
        ('black_komi','float32'), ('white_komi','float32'),
    ]:
        _ensure_dataset(logs, name, shape=(0,), maxshape=(None,), chunks=(1024,), dtype=dt)

    for name, dt in [('turn_flat','int32'), ('deep_search_flat','bool'), ('move_flat','int32')]:
        _ensure_dataset(logs, name, shape=(0,), maxshape=(None,), chunks=(4096,), dtype=dt)



def save_game_to_hdf5(
    h5_file_path: str,
    game_data: dict,
    game_id: str | None = None,
) -> Tuple[int, int]:
    """
    Standalone-запись одной партии в HDF5 с полным эксклюзивным локом.
    Используется вне сервиса: ручное тестирование, отладка, офлайн-вставка.

    Возвращает:
        train_game → base_index (сэмпли до вставки)
        game_log   → game_index (строка лога до вставки)
        -1         → пустые данные, ничего не записано
    """
    if game_id is None:
        game_id = str(uuid.uuid4())

    message_type = game_data.get("message_type", "train_game")

    with h5py.File(h5_file_path, 'a', libver='latest') as f:

        if message_type == "train_game":
            return _write_train_game(f, game_data, game_id), -1

        elif message_type == "game_log":
            return _write_game_log(f, game_data, game_id)

        else:
            raise ValueError(f"Неизвестный message_type: {message_type!r}")

def _write_train_game(f: h5py.File, payload: dict, game_id: str) -> int:
    """
    Возвращает количество добавленных сэмплов, либо -1 если данных нет.
    Работает с уже открытым f (без open/close).
    """
    turns = np.asarray(payload['turn'], dtype=np.int32)
    added = len(turns)
    if added == 0:
        return -1

    base      = f['state_tensor'].shape[0]
    new_total = base + added

    for ds in ['state_tensor', 'mcts_policy', 'mcts_opp_policy', 'territories',
               'value', 'score', 'turn', 'move_meta', 'game_id', 'noise_seed']:
        f[ds].resize((new_total,) + f[ds].shape[1:])

    f['state_tensor']   [base:new_total] = np.asarray(payload['state_tensor'],    dtype=np.float32)
    f['mcts_policy']    [base:new_total] = np.asarray(payload['mcts_policy'],     dtype=np.float32)
    f['mcts_opp_policy'][base:new_total] = np.asarray(payload['mcts_opp_policy'], dtype=np.float32)
    f['territories']    [base:new_total] = np.asarray(payload['territories'],     dtype=np.float32)
    f['value']          [base:new_total] = np.asarray(payload['value'],           dtype=np.int64)
    f['score']          [base:new_total] = np.asarray(payload['score'],           dtype=np.int64)
    f['turn']           [base:new_total] = turns
    f['move_meta']      [base:new_total] = np.array(payload['move_meta']).astype('S64')
    f['game_id']        [base:new_total] = np.array([game_id] * added).astype('S96')
    f['noise_seed']     [base:new_total] = np.full(added, payload['noise_seed_meta'], dtype=np.int32)

    return added


def _write_game_log(f: h5py.File, payload: dict, game_id: str) -> Tuple[int, int]:
    """
    Возвращает индекс вставленной строки лога, либо -1 если данных нет.
    Работает с уже открытым f (без open/close).
    """
    game_len = int(payload.get('game_length', 0))
    if game_len == 0:
        return -1, -1

    logs       = f['game_logs']
    game_idx   = logs['game_id'].shape[0]
    flat_start = logs['turn_flat'].shape[0]
    flat_end   = flat_start + game_len

    for ds in ['game_id', 'agent_meta', 'length', 'winner',
               'margin', 'margin_no_komi',
               'black_stones', 'white_stones',
               'black_territory', 'white_territory',
               'black_komi', 'white_komi']:
        logs[ds].resize((game_idx + 1,))

    for ds in ['turn_flat', 'deep_search_flat', 'move_flat']:
        logs[ds].resize((flat_end,))

    agent_raw = payload.get('agent_meta', b'')
    agent_str = agent_raw.decode('utf-8') if isinstance(agent_raw, bytes) else str(agent_raw)

    logs['game_id']        [game_idx] = np.bytes_(game_id)
    logs['agent_meta']     [game_idx] = np.bytes_(agent_str)
    logs['length']         [game_idx] = game_len
    logs['winner']         [game_idx] = payload.get('winner', 0)
    logs['margin']         [game_idx] = payload.get('margin', 0.0)
    logs['margin_no_komi'] [game_idx] = payload.get('margin_no_komi', 0.0)
    logs['black_stones']   [game_idx] = payload.get('black_stones', 0)
    logs['white_stones']   [game_idx] = payload.get('white_stones', 0)
    logs['black_territory'][game_idx] = payload.get('black_territory', 0)
    logs['white_territory'][game_idx] = payload.get('white_territory', 0)
    logs['black_komi']     [game_idx] = payload.get('black_komi', 0.0)
    logs['white_komi']     [game_idx] = payload.get('white_komi', 0.0)

    logs['turn_flat']       [flat_start:flat_end] = np.asarray(payload['turn'],        dtype=np.int32)
    logs['deep_search_flat'][flat_start:flat_end] = np.asarray(payload['deep_search'], dtype=np.bool_)
    logs['move_flat']       [flat_start:flat_end] = np.asarray(payload['move'],        dtype=np.int32)

    return 1, game_len



def _write_tournament_result(jsonl_path: str, payload: dict, game_id: str) -> Tuple[int, int]:
    """
    Дописывает одну строку в JSONL-файл турнирных результатов.
    payload уже является dict (десериализован из JSON воркером).
    """
    record = {
        "game_id":      game_id,
        "agent_black":      payload.get("agent_black"),
        "agent_white":      payload.get("agent_white"),
        "rules_black":      payload.get("rules_black"),
        "rules_white":      payload.get("rules_white"),
        "result_for_black": payload.get("result_for_black"),   # 1.0 / 0.0 / -1
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return 1,1