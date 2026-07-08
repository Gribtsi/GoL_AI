import os
import h5py
from joblib import Parallel, delayed
from pathlib import Path
from typing import List, Optional

from addresses_config import MODELS_DIR
from config import LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, TRAINING_STEPS, BATCH_SIZE
from self_play_service import run_game_pool_worker, debug_worker, WorkerConfig
from random_network import RandomNetwork, RandomNetworkClient
from train_epochs import train_network_epochs
from model_manager import ModelManager
from rl_agent import RLAgent, NewRLAgent
import torch
import multiprocessing as mp
from game_data_sender import DebugDataSender, DebugDataSenderWrapper
from save_data import _init_log_datasets, _init_train_datasets
import gc
import re
import glob

from train_network import train_network_steps_2



def create_dataset_file(n_thread: int) -> str:
    file_name = f"dataset_{n_thread}.h5"
    full_path = os.path.join(RANDOM_SAVE_DIR, file_name)

    # Добавляем параметр libver='latest' для поддержки формата 1.10+
    with h5py.File(full_path, 'w', libver='latest') as f:
        print(f"HDF5 файл успешно создан по пути: {full_path}")
        _init_log_datasets(f)
        _init_train_datasets(f)

    return full_path


def create_random_dataset(count : int = 1):
    print("Начинаем генерацию...")

    processes = []
    for i in range(count):
        create_dataset_file(i)

    configs = [
        WorkerConfig(
            process_id=i,
            games_per_process=1,
            sender_cls=DebugDataSender,
            sender_args=(f"{RANDOM_SAVE_DIR}dataset_{i}.h5",),
            sender_kwargs={},
            client_cls=RandomNetworkClient,
            client_args=(),
            client_kwargs={"name": f"rng_net"},
        ) for i in range(count)]


    for i in range(count):
        p = mp.Process(target=run_game_pool_worker, args=(configs[i],))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"Генерация завершена!")

def train_test():
    model_manager = ModelManager(NewRLAgent, save_dir=".\\shared_models\\", device="cuda")

    model = model_manager.model_class().to(model_manager.device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,  # Nesterov momentum — стандартная практика для SGD в DL
    )
    model_manager.load_checkpoint("agent_v0.pth", model, optimizer)

    # Обучаем! (Ваш метод train_network_steps)
    train_network_steps_2(model, optimizer, steps=TRAINING_STEPS, datasets_filepath=".\\shared_data\\dataset.h5", batch_size=BATCH_SIZE, device="cuda")

    # Сохраняем новую версию
    version = 1
    current_model_name = f"agent_v{version}.pth"
    model_manager.save_checkpoint(model, optimizer, {'version': version}, current_model_name)

TRAIN_DATASETS = [
    'state_tensor', 'mcts_policy', 'mcts_opp_policy', 'territories',
    'value', 'score', 'turn', 'move_meta', 'game_id', 'noise_seed',
]

LOG_DATASETS = [
    'game_id', 'agent_meta', 'length', 'winner', 'margin', 'margin_no_komi',
    'black_stones', 'white_stones', 'black_territory', 'white_territory',
    'black_komi', 'white_komi',
]

LOG_FLAT_DATASETS = ['turn_flat', 'deep_search_flat', 'move_flat']


def merge_datasets(
    source_dir: str,
    dest_dir: str,
    output_filename: str = "dataset.h5",
    chunk_size: int = 512,
    verbose: bool = True,
) -> Optional[str]:
    """
    Сливает все HDF5-датасеты из source_dir в один файл в dest_dir.

    Args:
        source_dir:      Папка с исходными датасетами (*.h5).
        dest_dir:        Папка, куда будет создан результирующий файл.
        output_filename: Имя итогового файла (default: merged.h5).
        chunk_size:      Размер чанка при покусочном копировании (по первой оси).
        verbose:         Печатать прогресс.

    Returns:
        Полный путь к созданному файлу, или None если не найдено источников.
    """
    source_files = sorted(glob.glob(os.path.join(source_dir, "*.h5")))
    if not source_files:
        print(f"[merge] Нет .h5 файлов в {source_dir}")
        return None

    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, output_filename)

    if verbose:
        print(f"[merge] Источников: {len(source_files)}, цель: {dest_path}")

    with h5py.File(dest_path, 'w', libver='latest') as dst:
        _init_train_datasets(dst)
        _init_log_datasets(dst)

        total_samples = 0
        total_games   = 0

        for src_path in source_files:
            if verbose:
                print(f"  [merge] Читаем {os.path.basename(src_path)} ...", end=" ", flush=True)

            with h5py.File(src_path, 'r', swmr=True, libver='latest') as src:
                n = src['state_tensor'].shape[0]
                if n == 0:
                    if verbose:
                        print("пусто, пропуск")
                    continue

                # ── Train datasets ────────────────────────────────────────────
                base      = dst['state_tensor'].shape[0]
                new_total = base + n

                for ds_name in TRAIN_DATASETS:
                    dst[ds_name].resize((new_total,) + dst[ds_name].shape[1:])

                # Копируем чанками чтобы не грузить весь датасет в память
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    for ds_name in TRAIN_DATASETS:
                        dst[ds_name][base + start: base + end] = src[ds_name][start:end]

                total_samples += n

                # ── Log datasets ──────────────────────────────────────────────
                if 'game_logs' not in src:
                    if verbose:
                        print(f"({n} сэмплов, нет логов)")
                    continue

                src_logs = src['game_logs']
                dst_logs = dst['game_logs']

                n_games    = src_logs['game_id'].shape[0]
                n_flat     = src_logs['turn_flat'].shape[0]
                game_base  = dst_logs['game_id'].shape[0]
                flat_base  = dst_logs['turn_flat'].shape[0]

                if n_games > 0:
                    for ds_name in LOG_DATASETS:
                        dst_logs[ds_name].resize((game_base + n_games,))
                    for ds_name in LOG_FLAT_DATASETS:
                        dst_logs[ds_name].resize((flat_base + n_flat,))

                    for start in range(0, n_games, chunk_size):
                        end = min(start + chunk_size, n_games)
                        for ds_name in LOG_DATASETS:
                            dst_logs[ds_name][game_base + start: game_base + end] = \
                                src_logs[ds_name][start:end]

                    if n_flat > 0:
                        for start in range(0, n_flat, chunk_size):
                            end = min(start + chunk_size, n_flat)
                            for ds_name in LOG_FLAT_DATASETS:
                                dst_logs[ds_name][flat_base + start: flat_base + end] = \
                                    src_logs[ds_name][start:end]

                    total_games += n_games

                dst.flush()
                if verbose:
                    print(f"({n} сэмплов, {n_games} игр)")

        if verbose:
            print(f"[merge] Готово: {total_samples} сэмплов, {total_games} игр → {dest_path}")

    return dest_path

RANDOM_MERGED_DIR = ".\\test1\\random_merged\\"
RANDOM_SAVE_DIR = ".\\test1\\random\\"
RANDOM_MODEL_DIR = ".\\test1\\random_model\\"


if __name__ == "__main__":
    train_test()


