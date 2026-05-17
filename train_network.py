
import torch.nn.functional as F
import numpy as np

from addresses_config import DATASETS_FILEPATH
from config import BOARD_SIZE, board_size_sqr, REPLAY_BUFFER_LENGTH

import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class TrainingDataset(Dataset):
    def __init__(self, datasets_filepath=DATASETS_FILEPATH, buffer_length=REPLAY_BUFFER_LENGTH):
        super().__init__()
        self.datasets_filepath = datasets_filepath
        self.buffer_length = buffer_length
        self.file = None
        self.augment = True

        # Узнаем текущий размер файла ОДИН раз при инициализации
        with h5py.File(self.datasets_filepath, 'r', swmr=True, libver='latest') as f:
            self.total_samples = f['state_tensor'].shape[0]

    def __len__(self):
        # Если вы хотите возвращать фиксированный буфер:
        return min(self.total_samples, self.buffer_length)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.datasets_filepath, 'r', swmr=True, libver='latest')
            self.d_state = self.file['state_tensor']
            self.d_policy = self.file['mcts_policy']
            self.d_value = self.file['value']
            self.d_terr = self.file['territories']
            self.d_score = self.file['score']

        if idx >= self.d_state.shape[0]:
            self.d_state.refresh()
            self.d_policy.refresh()
            self.d_value.refresh()
            self.d_terr.refresh()
            self.d_score.refresh()
            self.total_samples = self.d_state.shape[0]

        if self.total_samples > self.buffer_length:
            actual_idx = self.total_samples - self.buffer_length + idx
        else:
            actual_idx = idx

        # Чтение данных из HDF5 (возвращаются numpy-массивы или скаляры)
        state = self.d_state[actual_idx]
        policy = self.d_policy[actual_idx]
        value = self.d_value[actual_idx]
        terr = self.d_terr[actual_idx]
        score = self.d_score[actual_idx] + board_size_sqr

        if self.augment:
            state, policy, terr = augment_data(state, policy, terr)


        value = np.array([value], dtype=np.float32)

        terr = np.expand_dims(terr, axis=0).astype(np.float32)

        return state, policy, value, terr, score


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


def train_network_steps(
        model, optimizer, datasets_filepath=DATASETS_FILEPATH,
        steps=50, batch_size=256, device="cpu",
        entropy_beta=0.01, w_terr=0.03, w_score=0.02
):
    model.train()
    model.to(device)

    dataset = TrainingDataset(datasets_filepath)
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


        value_loss = F.mse_loss(pred_value.squeeze(-1), value_batch.squeeze(-1))
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
