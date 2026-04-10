import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from board import BOARD_SIZE, IN_CHANNELS


class ResBlock(nn.Module):
    """
    Остаточный блок (Residual Block), ключевой компонент архитектуры.
    """

    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Ключевой элемент - сложение с входом
        out = F.relu(out)
        return out


# --- Определение архитектуры агента ---


CONV_CHANNELS = 64
FEATURES = 64
NUM_RES_BLOCKS = 7

class RLAgent(nn.Module):
    """
    Нейросетевой агент в стиле AlphaGo Zero.

    Состоит из общего свёрточного "тела" и двух "голов":
    1. Policy Head: предсказывает вероятность каждого хода.
    2. Value Head: оценивает вероятность победы из текущей позиции.
    """

    def __init__(self):
        super(RLAgent, self).__init__()

        # Входной канал = 33, доска 19x19

        # --- 1. Общий ствол (Shared Body) ---
        self.conv_in = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, CONV_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(CONV_CHANNELS),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList(
            [ResBlock(CONV_CHANNELS) for _ in range(NUM_RES_BLOCKS)]
        )

        # --- 2. Голова Политики (Policy Head) ---
        self.policy_head = nn.Sequential(
            nn.Conv2d(CONV_CHANNELS, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE * 2 + 1)
        )

        # --- 3. Голова Оценки (Value Head) ---
        self.value_head = nn.Sequential(
            nn.Conv2d(CONV_CHANNELS, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, FEATURES),
            nn.ReLU(),
            nn.Linear(FEATURES, 1),
            nn.Tanh()  # Выход в диапазоне [-1, 1]
        )

        # --- 4. Голова Территории (Ownership Head) ---
        # Предсказывает принадлежность каждой клетки на доске.
        # В отличие от других голов, здесь мы сохраняем пространственную структуру.
        self.territory_head = nn.Sequential(
            nn.Conv2d(CONV_CHANNELS, 1, kernel_size=1),
            # Не используем ReLU и BatchNorm перед Tanh, чтобы не искажать центровку значений
            nn.Tanh()  # Выход (batch, 1, BOARD_SIZE, BOARD_SIZE) в диапазоне [-1, 1]
        )

        # --- 5. Голова Счета (Score Head) ---
        # Предсказывает категориальное распределение счета от -BOARD_SIZE до BOARD_SIZE
        # Количество классов = BOARD_SIZE * 2 + 1
        self.score_head = nn.Sequential(
            nn.Conv2d(CONV_CHANNELS, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, FEATURES),
            nn.ReLU(),
            nn.Linear(FEATURES, BOARD_SIZE * BOARD_SIZE * 2 + 1)  # Выдаем логиты для кросс-энтропии
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Прямой проход через сеть.

        Args:
            x (torch.Tensor): Входной тензор состояния доски (batch, 33, BOARD_SIZE, BOARD_SIZE).

        Returns:
            policy_logits (torch.Tensor): Логиты вероятностей ходов (batch, BOARD_SIZE*BOARD_SIZE*2 + 1).
            value (torch.Tensor): Оценка позиции (batch, 1).
            territory (torch.Tensor): Предсказание принадлежности клеток (batch, 1, BOARD_SIZE, BOARD_SIZE).
            score_logits (torch.Tensor): Логиты распределения финального счета (batch, BOARD_SIZE*2 + 1).
        """
        # Пропускаем через общий ствол
        x = self.conv_in(x)
        for block in self.res_blocks:
            x = block(x)

        # Получаем выходы от всех четырех голов
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        territory = self.territory_head(x)
        score_logits = self.score_head(x)

        # Возвращаем логиты (без Softmax), так как это стандарт для nn.CrossEntropyLoss
        return policy_logits, value, territory, score_logits