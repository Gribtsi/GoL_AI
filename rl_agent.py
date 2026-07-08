import torch
import torch.nn as nn
import torch.nn.functional as F

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

from config import BOARD_SIZE, IN_CHANNELS, CONV_CHANNELS, FEATURES, NUM_RES_BLOCKS, possible_moves_total, \
    CHANNELS_HEAD, GP_BLOCKS, CHANNELS_GLOBAL_POOLING, board_size_sqr


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

class NewResBlock(nn.Module):
    def __init__(self, num_channels):
        super(NewResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        out = out + residual
        return out

class GlobalPoolingVector(nn.Module):
    """
    Вычисляет вектор глобального пулинга из входного тензора.
    BN + ReLU → mean + max → конкатенация.

    Input:  [B, C, H, W]
    Output: [B, 2*C]
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn(x), inplace=True)
        g_mean = y.mean(dim=(2, 3))               # [B, C]
        g_max  = y.amax(dim=(2, 3))               # [B, C]
        return torch.cat([g_mean, g_max], dim=1)  # [B, 2*C]


class GlobalPoolingBias(nn.Module):
    """
    Применяет сдвиг, полученный из вектора глобального пулинга,
    к последним num_bias_channels каналам тензора x.

    Принимает:
        x:      [B, num_channels, H, W]
        gp_vec: [B, 2*num_pooled_channels]

    Возвращает: [B, num_channels, H, W]

    Ограничения:
        0 < num_pooled_channels <= num_channels
        0 < num_bias_channels   <= num_channels
    """

    def __init__(
        self,
        num_channels: int,
        num_pooled_channels: int,
        num_bias_channels: int,
    ):
        super().__init__()

        assert 0 < num_pooled_channels <= num_channels
        assert 0 < num_bias_channels   <= num_channels

        self.num_channels      = num_channels
        self.num_bias_channels = num_bias_channels

        self.fc = nn.Linear(2 * num_pooled_channels, num_bias_channels)

    def forward(self, x: torch.Tensor, gp_vec: torch.Tensor) -> torch.Tensor:
        bias = self.fc(gp_vec)                     # [B, num_bias_channels]
        bias = bias.unsqueeze(-1).unsqueeze(-1)    # [B, num_bias_channels, 1, 1]

        pad = x.new_zeros(
            x.size(0),
            self.num_channels - self.num_bias_channels,
            1, 1,
        )
        bias_full = torch.cat([pad, bias], dim=1)  # [B, num_channels, 1, 1]
        return x + bias_full

class NewGlobalPoolingResBlock(nn.Module):
    def __init__(self, num_channels: int, num_pooled_channels: int):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(
            num_channels, num_channels,
            kernel_size=3, padding=1, bias=False
        )

        self.gvec = GlobalPoolingVector(num_channels=num_pooled_channels)

        self.gpool = GlobalPoolingBias(
            num_channels=num_channels,
            num_pooled_channels=num_pooled_channels,
            num_bias_channels=num_channels - num_pooled_channels
        )

        self.p_conv = nn.Conv2d(num_channels, num_channels,        kernel_size=1, bias=False)
        self.g_conv = nn.Conv2d(num_channels, num_pooled_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels,
            kernel_size=3, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)

        p = self.p_conv(out)
        g = self.g_conv(out)

        out = self.gpool(p, self.gvec(g))          # P + bias из G

        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        out = out + residual
        return out

class NewPolicyHead(nn.Module):

    def __init__(self, in_channels: int, c_head: int):
        super().__init__()

        self.p_conv = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)
        self.g_conv = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)

        # G-ветвь: вычисляет GP-вектор
        self.gp_vec = GlobalPoolingVector(c_head)

        # P-ветвь: применяет сдвиг из GP-вектора ко всем c_head каналам
        self.gpb = GlobalPoolingBias(
            num_channels=c_head,
            num_pooled_channels=c_head,
            num_bias_channels=c_head,
        )

        self.bn = nn.BatchNorm2d(c_head)

        self.board_conv  = nn.Conv2d(c_head, 4, kernel_size=1)
        self.scalar_fc   = nn.Linear(2 * c_head, 4)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, in_channels, H, W]

        Returns:
            board_logits:  [B, 4, H, W]
            scalar_logits: [B, 4]
        """
        p = self.p_conv(x)   # [B, c_head, H, W]
        g = self.g_conv(x)   # [B, c_head, H, W]

        # GP-вектор из G-ветви — используется дважды
        gp_vec = self.gp_vec(g)              # [B, 2*c_head]

        # Сдвигаем P-ветвь
        p = self.gpb(p, gp_vec)             # [B, c_head, H, W]
        p = F.relu(self.bn(p), inplace=True)

        board_logits  = self.board_conv(p)  # [B, 4, H, W]
        scalar_logits = self.scalar_fc(gp_vec)  # [B, 4]

        B = board_logits.size(0)

        self_board = board_logits[:, :2, :, :]  # [B, 2, H, W]
        opp_board = board_logits[:, 2:, :, :]  # [B, 2, H, W]

        policy_self = torch.cat([
            self_board[:, 0, :, :].reshape(B, -1),  # [B, H*W]  без цикла
            self_board[:, 1, :, :].reshape(B, -1),  # [B, H*W]  с циклом
            scalar_logits[:, :2],
        ], dim=1)  # [B, H*W*2]

        policy_opp = torch.cat([
            opp_board[:, 0, :, :].reshape(B, -1),
            opp_board[:, 1, :, :].reshape(B, -1),
            scalar_logits[:, 2:],
        ], dim=1)

        return policy_self, policy_opp

class NewValueHead(nn.Module):
    def __init__(self, in_channels: int, c_head: int):
        """
        Args:
            in_channels: число каналов трунка
            c_head:      число каналов V-ветви
            board_size:  размер доски (одна сторона), board_size_sqr = board_size**2
        """
        super().__init__()

        # 1. Проекция трунка в V-ветвь
        self.v_conv = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)

        # 2. Global Pooling Vector из V
        self.gpv = GlobalPoolingVector(c_head)

        # 3. Предсказание исхода: V_pooled → c_head → ReLU → 3
        self.outcome_fc1 = nn.Linear(2 * c_head, c_head)
        self.outcome_fc2 = nn.Linear(c_head, 3)

        # 4. Предсказание территории: V → 1 канал
        self.territory_conv = nn.Conv2d(c_head, 1, kernel_size=1)

        # 5. Предсказание счёта: V_pooled → c_head → ReLU → board_size_sqr*2+1
        self.score_fc1 = nn.Linear(2 * c_head, c_head)
        self.score_fc2 = nn.Linear(c_head, board_size_sqr * 2 + 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, in_channels, H, W]

        Returns:
            outcome_logits:   [B, 3]          — победа / поражение / ничья
            territory_logits: [B, 1, H, W]    — территория по позициям
            score_logits:     [B, H*W*2 + 1]  — распределение по счёту
        """
        # 1. Проекция в V
        v = self.v_conv(x)                          # [B, c_head, H, W]

        # 2. GP-вектор
        v_pooled = self.gpv(v)                      # [B, 2*c_head]

        # 3. Исход
        outcome = F.relu(self.outcome_fc1(v_pooled), inplace=True)
        outcome_logits = self.outcome_fc2(outcome)  # [B, 3]

        # 4. Территория
        territory_logits = self.territory_conv(v)   # [B, 1, H, W]

        # 5. Счёт
        score = F.relu(self.score_fc1(v_pooled), inplace=True)
        score_logits = self.score_fc2(score)        # [B, board_size_sqr*2 + 1]

        return outcome_logits, territory_logits, score_logits

class NewRLAgent(nn.Module):
    def __init__(self):
        super(NewRLAgent, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, CONV_CHANNELS, kernel_size=5, padding=2, bias=False),
        )

        gp_indices = set(
            round(i)
            for i in torch.linspace(0, NUM_RES_BLOCKS - 1, GP_BLOCKS).tolist()
        )

        self.res_blocks = nn.Sequential(
            *[
                NewGlobalPoolingResBlock(CONV_CHANNELS, CHANNELS_GLOBAL_POOLING)
                if i in gp_indices
                else NewResBlock(CONV_CHANNELS)
                for i in range(NUM_RES_BLOCKS)
            ]
        )
        self.policy_head = NewPolicyHead(CONV_CHANNELS, CHANNELS_HEAD)

        self.value_head = NewValueHead(CONV_CHANNELS, CHANNELS_HEAD)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv_in(x)
        x = self.res_blocks(x)  # Заменили цикл на вызов Sequential

        policy_logits_self, policy_logits_opp = self.policy_head(x)

        outcomes, territory, score_logits = self.value_head(x)

        return policy_logits_self, policy_logits_opp, outcomes, territory, score_logits

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
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, possible_moves_total)
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