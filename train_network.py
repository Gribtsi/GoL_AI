
import torch.nn.functional as F
import numpy as np

from config import BOARD_SIZE

import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class TrainingDataset(Dataset):
    def __init__(self, h5_file_path, board_size=19):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.board_size = board_size
        self.board_size_sqr = board_size * board_size

        # Мы не открываем файл здесь, чтобы избежать конфликтов при multiprocessing
        self.file = None

        # Сканируем файл, чтобы узнать общее количество позиций и где они лежат
        self.index_map = []
        with h5py.File(self.h5_file_path, 'r') as f:
            for game_id in f.keys():
                game_group = f[game_id]
                num_turns = len(game_group['value'])
                # Сохраняем "координаты" каждого хода: (game_id, индекс_внутри_игры)
                for i in range(num_turns):
                    self.index_map.append((game_id, i))

        self.total_samples = len(self.index_map)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Ленивое открытие файла (каждый воркер DataLoader'а откроет свой)
        if self.file is None:
            # swmr=True позволяет читать файл, пока в него пишет другой процесс (воркер-генератор)
            self.file = h5py.File(self.h5_file_path, 'r', swmr=True)

        game_id, turn_idx = self.index_map[idx]
        group = self.file[game_id]

        # Читаем данные конкретного хода (numpy arrays)
        state = group['state_tensor'][turn_idx]
        policy = group['mcts_policy'][turn_idx]
        territory = group['territories'][turn_idx]
        value = group['value'][turn_idx]
        score = group['score'][turn_idx]

        # Применяем аугментацию (ваша существующая функция)
        aug_state, aug_policy, aug_territory = augment_data(
            state, policy, territory
        )

        # Подготавливаем скаляры
        # Сдвигаем score чтобы он был >= 0 для кросс-энтропии, как у вас в коде
        score = score + self.board_size_sqr

        # Возвращаем тензоры
        return (
            torch.from_numpy(aug_state).float(),
            torch.from_numpy(aug_policy).float(),
            torch.tensor([value], dtype=torch.float32),  # unsqueeze(1) эквивалент
            torch.from_numpy(aug_territory).float().unsqueeze(0),  # (1, H, W)
            torch.tensor(score, dtype=torch.long)
        )

def augment_data(state: np.ndarray, policy: np.ndarray, territory: np.ndarray):
    """
    Применяет случайное вращение (0, 90, 180, 270 градусов) и
    случайное отражение (по горизонтали) к данным состояния, политики и территории.

    Args:
        state: 3D массив (C, board_size, board_size)
        policy: 1D массив размером (board_size * board_size * 2 + 1)
        territory: 2D массив (board_size, board_size)
        board_size: Размер доски (например, 19)

    Returns:
        Кортеж из аугментированных (state, policy, territory)
    """

    k = np.random.randint(0, 4)
    flip = np.random.choice([True, False])

    # 1. Трансформация State (C, H, W)
    # Вращаем последние две оси (H=1, W=2)
    aug_state = np.rot90(state, k=k, axes=(1, 2))
    if flip:
        # Отражаем по горизонтали (по оси W, то есть axis=2)
        aug_state = np.flip(aug_state, axis=2)

    # 2. Трансформация Territory (H, W)
    aug_territory = np.rot90(territory, k=k, axes=(0, 1))
    if flip:
        aug_territory = np.flip(aug_territory, axis=1)

    # 3. Трансформация Policy
    # Policy имеет структуру [life_cycle_0_moves, life_cycle_1_moves, pass_move]
    board_sqr = BOARD_SIZE * BOARD_SIZE
    aug_policy = np.zeros_like(policy)
    aug_policy[-1] = policy[-1]

    for i in range(2):
        start_idx = i * board_sqr
        end_idx = start_idx + board_sqr

        # Извлекаем кусок политики и превращаем в 2D доску (H, W)
        policy_board = policy[start_idx:end_idx].reshape((BOARD_SIZE, BOARD_SIZE))

        aug_policy_board = np.rot90(policy_board, k=k, axes=(0, 1))
        if flip:
            aug_policy_board = np.flip(aug_policy_board, axis=1)

        # flatten reshape соответствуют нынешнему подходу по кодировке ходов
        aug_policy[start_idx:end_idx] = aug_policy_board.flatten()

    return aug_state.copy(), aug_policy.copy(), aug_territory.copy()


def train_network_epochs(
        model, optimizer, h5_file_path,  # Теперь передаем путь к HDF5
        epochs=1, batch_size=256, device="cpu",
        w_terr=0.03, w_score=0.02
):
    model.train()
    model.to(device)

    # Инициализируем наш новый Dataset
    dataset = TrainingDataset(h5_file_path)

    # num_workers=4 (или больше) значительно ускорит загрузку с диска
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"Начинаем обучение на {len(dataset)} примерах из HDF5...")

    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0
        total_terr_loss = 0
        total_score_loss = 0

        for state_batch, policy_batch, value_batch, terr_batch, score_batch in dataloader:
            state_batch = state_batch.to(device)
            policy_batch = policy_batch.to(device)
            terr_batch = terr_batch.to(device)
            value_batch = value_batch.to(device)
            score_batch = score_batch.to(device)

            optimizer.zero_grad()

            pred_policy_logits, pred_value, pred_territory, pred_score_logits = model(state_batch)

            value_loss = F.mse_loss(pred_value, value_batch)

            pred_log_probs = F.log_softmax(pred_policy_logits, dim=1)
            policy_loss = F.kl_div(pred_log_probs, policy_batch, reduction='batchmean')

            terr_loss = F.mse_loss(pred_territory, terr_batch)
            score_loss = F.cross_entropy(pred_score_logits, score_batch)

            loss = value_loss + policy_loss + (w_terr * terr_loss) + (w_score * score_loss)

            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item() * state_batch.size(0)
            total_value_loss += value_loss.item() * state_batch.size(0)
            total_terr_loss += terr_loss.item() * state_batch.size(0)
            total_score_loss += score_loss.item() * state_batch.size(0)

        avg_policy_loss = total_policy_loss / len(dataset)
        avg_value_loss = total_value_loss / len(dataset)
        avg_terr_loss = total_terr_loss / len(dataset)
        avg_score_loss = total_score_loss / len(dataset)

        print(f"Эпоха {epoch + 1}/{epochs}: "
              f"Pol = {avg_policy_loss:.4f}, Val = {avg_value_loss:.4f}, "
              f"Terr = {avg_terr_loss:.4f}, Score = {avg_score_loss:.4f}")

def train_network_steps(
        model, optimizer, h5_file_path, # Передаем путь
        steps=50, batch_size=256, device="cpu",
        entropy_beta=0.01, w_terr=0.03, w_score=0.02
):
    model.train()
    model.to(device)

    dataset = TrainingDataset(h5_file_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"Начинаем обучение. Позиций на диске: {len(dataset)}, Шагов: {steps}...")

    total_policy_loss, total_value_loss = 0.0, 0.0
    total_terr_loss, total_score_loss = 0.0, 0.0
    total_entropy = 0.0
    processed_samples, step_count = 0, 0

    data_iter = iter(dataloader)

    for _ in range(steps):
        try:
            state_batch, policy_batch, value_batch, terr_batch, score_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            state_batch, policy_batch, value_batch, terr_batch, score_batch = next(data_iter)

        state_batch = state_batch.to(device)
        policy_batch = policy_batch.to(device)
        value_batch = value_batch.to(device)
        terr_batch = terr_batch.to(device)
        score_batch = score_batch.to(device)

        optimizer.zero_grad()

        pred_policy_logits, pred_value, pred_territory, pred_score_logits = model(state_batch)

        value_loss = F.mse_loss(pred_value, value_batch)
        terr_loss = F.mse_loss(pred_territory, terr_batch)
        score_loss = F.cross_entropy(pred_score_logits, score_batch)

        pred_log_probs = F.log_softmax(pred_policy_logits, dim=1)
        policy_loss = -torch.mean(torch.sum(policy_batch * pred_log_probs, dim=1))

        pred_probs = torch.exp(pred_log_probs)
        entropy = -torch.mean(torch.sum(pred_probs * pred_log_probs, dim=1))

        loss = value_loss + policy_loss + (w_terr * terr_loss) + (w_score * score_loss) - (entropy_beta * entropy)

        loss.backward()
        optimizer.step()

        batch_size_actual = state_batch.size(0)
        total_policy_loss += policy_loss.item() * batch_size_actual
        total_value_loss += value_loss.item() * batch_size_actual
        total_terr_loss += terr_loss.item() * batch_size_actual
        total_score_loss += score_loss.item() * batch_size_actual
        total_entropy += entropy.item() * batch_size_actual

        processed_samples += batch_size_actual
        step_count += 1

    avg_policy_loss = total_policy_loss / processed_samples
    avg_value_loss = total_value_loss / processed_samples
    avg_terr_loss = total_terr_loss / processed_samples
    avg_score_loss = total_score_loss / processed_samples
    avg_entropy = total_entropy / processed_samples

    print(f"Шаги: {step_count} | Pol: {avg_policy_loss:.4f} | Val: {avg_value_loss:.4f} | "
          f"Terr: {avg_terr_loss:.4f} | Score: {avg_score_loss:.4f} | Ent: {avg_entropy:.4f}")
