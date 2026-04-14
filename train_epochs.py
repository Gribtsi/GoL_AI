from torch.utils.data import DataLoader
import torch.nn.functional as F

from addresses_config import DATASETS_FILEPATH
from config import REPLAY_BUFFER_LENGTH
from train_network import TrainingDataset


def train_network_epochs(
        model, optimizer, datasets_filepath=DATASETS_FILEPATH,
        epochs=1, batch_size=256, device="cuda",
        w_terr=0.03, w_score=0.02, buffer_length = REPLAY_BUFFER_LENGTH
):
    model.train()
    model.to(device)

    # Инициализируем наш новый Dataset
    dataset = TrainingDataset(datasets_filepath, buffer_length=buffer_length)

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
