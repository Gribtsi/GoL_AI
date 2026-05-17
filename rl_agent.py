import torch
import torch.nn as nn
import torch.nn.functional as F

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

from config import BOARD_SIZE, IN_CHANNELS, CONV_CHANNELS, FEATURES, NUM_RES_BLOCKS

class ResBlock(nn.Module):
    """
    Остаточный блок (Residual Block), ключевой компонент архитектуры.
    """
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out, inplace=True)
        return out


class RLAgent(nn.Module):

    def __init__(self):
        super(RLAgent, self).__init__()

        # --- 1. Общий ствол (Shared Body) ---
        self.conv_in = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, CONV_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(CONV_CHANNELS),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(CONV_CHANNELS) for _ in range(NUM_RES_BLOCKS)]
        )

        # --- 2. Голова Политики (Policy Head) ---
        self.policy_head = nn.Sequential(
            nn.Conv2d(CONV_CHANNELS, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE * 2 + 1)
        )

        # --- 3. Голова Оценки (Value Head) ---
        self.value_head = nn.Sequential(
            nn.Conv2d(CONV_CHANNELS, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, FEATURES),
            nn.ReLU(inplace=True),
            nn.Linear(FEATURES, 1),
            nn.Tanh()
        )

        # --- 4. Голова Территории (Ownership Head) ---
        self.territory_head = nn.Sequential(
            # Здесь bias=True (по умолчанию) оставляем, т.к. нет BatchNorm!
            nn.Conv2d(CONV_CHANNELS, 1, kernel_size=1),
            nn.Tanh()
        )

        # --- 5. Голова Счета (Score Head) ---
        self.score_head = nn.Sequential(
            nn.Conv2d(CONV_CHANNELS, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, FEATURES),
            nn.ReLU(inplace=True),
            nn.Linear(FEATURES, BOARD_SIZE * BOARD_SIZE * 2 + 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv_in(x)
        x = self.res_blocks(x)  # Заменили цикл на вызов Sequential

        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        territory = self.territory_head(x)
        score_logits = self.score_head(x)

        return policy_logits, value, territory, score_logits