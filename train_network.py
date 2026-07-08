
import torch.nn.functional as F
import numpy as np

from addresses_config import DATASETS_FILEPATH
from config import BOARD_SIZE, board_size_sqr, REPLAY_BUFFER_LENGTH, W_TERR, W_SCORE, W_OPP, C_VALUE

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
            self.d_opp_policy = self.file['mcts_opp_policy']
            self.d_value = self.file['value']
            self.d_terr = self.file['territories']
            self.d_score = self.file['score']

        if idx >= self.d_state.shape[0]:
            self.d_state.refresh()
            self.d_policy.refresh()
            self.d_opp_policy.refresh()
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
        opp_policy = self.d_opp_policy[actual_idx]
        value = self.d_value[actual_idx]
        terr = self.d_terr[actual_idx]
        score = self.d_score[actual_idx] + board_size_sqr

        if self.augment:
            state, policy, terr, opp_policy = augment_data(state, policy, terr, opp_policy)


        value = np.array([value], dtype=np.int64)

        terr = np.expand_dims(terr, axis=0).astype(np.float32)

        return state, policy, opp_policy, value, terr, score


def augment_data(state: np.ndarray, policy: np.ndarray, territory: np.ndarray, opp_policy):
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
    aug_opp_policy = np.zeros_like(opp_policy)


    aug_opp_policy[-1] = opp_policy[-1]
    aug_opp_policy[-2] = opp_policy[-2]
    aug_policy[-1] = policy[-1]
    aug_policy[-2] = policy[-2]

    for i in range(2):
        start_idx = i * board_sqr
        end_idx = start_idx + board_sqr

        # Извлекаем кусок политики и превращаем в 2D доску (H, W)
        policy_board = policy[start_idx:end_idx].reshape((BOARD_SIZE, BOARD_SIZE))
        opp_policy_board = opp_policy[start_idx:end_idx].reshape((BOARD_SIZE, BOARD_SIZE))

        aug_policy_board = np.rot90(policy_board, k=k, axes=(0, 1))
        aug_opp_policy_board = np.rot90(opp_policy_board, k=k, axes=(0, 1))
        if flip:
            aug_policy_board = np.flip(aug_policy_board, axis=1)
            aug_opp_policy_board = np.flip(aug_opp_policy_board, axis=1)

        # flatten reshape соответствуют нынешнему подходу по кодировке ходов
        aug_policy[start_idx:end_idx] = aug_policy_board.flatten()
        aug_opp_policy[start_idx:end_idx] = aug_policy_board.flatten()

    return aug_state.copy(), aug_policy.copy(), aug_territory.copy(), aug_opp_policy.copy()


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
            state_batch, policy_batch, opp_policy_batch, value_batch, terr_batch, score_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            state_batch, policy_batch, opp_policy_batch, value_batch, terr_batch, score_batch = next(data_iter)

        state_batch = state_batch.to(device)
        policy_batch = policy_batch.to(device)


        value_batch = value_batch.to(device)
        terr_batch = terr_batch.to(device)
        score_batch = score_batch.to(device)
        opp_policy_batch = opp_policy_batch.to(device)

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

def train_network_steps_2(
        model, optimizer, datasets_filepath=DATASETS_FILEPATH,
        steps=50, batch_size=256, device="cuda",
):
    model.train()
    model.to(device)

    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            print(f"NaN/Inf в параметре: {name}")


    dataset    = TrainingDataset(datasets_filepath)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Начинаем обучение. Позиций: {len(dataset)}, Шагов: {steps}...")

    totals            = dict(policy=0., opp_policy=0., value=0., terr=0., score=0., total=0.)
    processed_samples = 0
    step_count        = 0
    data_iter         = iter(dataloader)

    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch     = next(data_iter)

        (state_batch,          # [B, C, H, W]
         policy_batch,         # [B, H*W*2+2]   — политика текущего игрока
         opp_policy_batch,     # [B, H*W*2+2]   — политика оппонента
         outcome_idx_batch,        # [B, 1]          — one-hot: idx win/draw/loss
         terr_batch,           # [B, 1, H, W]   — территория {0, 0.5, 1}
         score_batch         # [B, H*W*2+1]   — распределение счёта
         ) = [t.to(device) for t in batch]

        for name, t in [
            ("state", state_batch),
            ("policy", policy_batch),
            ("opp_policy", opp_policy_batch),
            ("outcome", outcome_idx_batch),
            ("terr", terr_batch),
            ("score", score_batch),
        ]:
            if not torch.isfinite(t).all():
                print(f"NaN/Inf в батче {name}!")


        optimizer.zero_grad()


        policy_self_logits, policy_opp_logits, outcome_logits, territory_logits, score_logits = \
            model(state_batch)

        # ── 1. Value loss ──────────────────────────────────────────────
        # cross-entropy между one-hot исходом и предсказанием outcome головы
        # outcome_batch: [B, 3],  outcome_logits: [B, 3]
        #outcome_batch = F.one_hot(outcome_idx_batch, num_classes=3).float()  # [B, 3]
        #log_z_hat   = F.log_softmax(outcome_logits, dim=1)          # [B, 3]
        #value_loss  = C_VALUE * -(outcome_batch * log_z_hat).sum(dim=1).mean()
        value_targets = outcome_idx_batch.view(-1).long()
        value_loss = C_VALUE * F.cross_entropy(outcome_logits, value_targets)

        # ── 2. Policy loss (текущий игрок) ────────────────────────────
        log_pi_hat  = F.log_softmax(policy_self_logits, dim=1)      # [B, A]
        policy_ce   = -(policy_batch * log_pi_hat).sum(dim=1)       # [B]
        policy_loss = policy_ce.mean()

        # ── 3. Opponent policy loss ───────────────────────────────────
        log_pi_opp_hat  = F.log_softmax(policy_opp_logits, dim=1)   # [B, A]
        opp_ce          = -(opp_policy_batch * log_pi_opp_hat).sum(dim=1)
        opp_policy_loss = W_OPP * opp_ce.mean()

        # ── 4. Territory loss ─────────────────────────────────────────
        # o(l,p) ∈ {0, 0.5, 1} — мягкий таргет → cross-entropy через log_softmax
        # territory_logits: [B, 1, H, W],  terr_batch: [B, 1, H, W]
        # Трактуем как бинарную классификацию с soft target через BCE
        terr_log_probs = F.logsigmoid(territory_logits)             # log(σ(x))
        terr_log_neg   = F.logsigmoid(-territory_logits)            # log(1 - σ(x))
        terr_loss      = W_TERR * -(
            terr_batch       * terr_log_probs +
            (1 - terr_batch) * terr_log_neg
        ).mean()

        # ── 5. Score probabilities loss ───────────────────────────────
        score_loss   = W_SCORE * F.cross_entropy(score_logits, score_batch)

        # ── Итоговый лосс ─────────────────────────────────────────────
        loss = value_loss + policy_loss + opp_policy_loss + terr_loss + score_loss

        for name, val in [
            ("value_loss", value_loss),
            ("policy_loss", policy_loss),
            ("opp_policy_loss", opp_policy_loss),
            ("terr_loss", terr_loss),
            ("score_loss", score_loss),
        ]:
            if not torch.isfinite(val):
                print(f"Shit in step {step}")
                print(f"  outcome_logits min/max: {outcome_logits.min():.2f}/{outcome_logits.max():.2f}")
                print(f"  policy_self min/max:    {policy_self_logits.min():.2f}/{policy_self_logits.max():.2f}")
                print(f"  score_logits min/max:   {score_logits.min():.2f}/{score_logits.max():.2f}")
                raise ValueError(f"NaN в {name}")

        loss.backward()
        optimizer.step()

        B = state_batch.size(0)
        totals['policy']     += policy_loss.item()     * B
        totals['opp_policy'] += opp_policy_loss.item() * B
        totals['value']      += value_loss.item()      * B
        totals['terr']       += terr_loss.item()       * B
        totals['score']      += score_loss.item()      * B
        totals['total']      += loss.item()            * B
        processed_samples    += B
        step_count           += 1

    avgs = {k: v / processed_samples for k, v in totals.items()}
    print(
        f"Шаги: {step_count} | "
        f"Pol: {avgs['policy']:.4f} | OppPol: {avgs['opp_policy']:.4f} | "
        f"Val: {avgs['value']:.4f} | Terr: {avgs['terr']:.4f} | "
        f"Score: {avgs['score']:.4f} | Total: {avgs['total']:.4f}"
    )
    return avgs
