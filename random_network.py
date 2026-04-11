import abc

from numpy.random import random, uniform
from torch.distributions import Dirichlet

import numpy as np

from board import Board
from config import BOARD_SIZE, BLACK, WHITE, board_size_sqr, IN_CHANNELS


class NetworkBase:
    @abc.abstractmethod
    def predict(self, state_numpy: np.ndarray, **kwargs) -> (np.ndarray, np.float32, np.ndarray):
        pass

class RandomNetwork(NetworkBase):
    """
    Заглушка нейросети, возвращающая случайные policy и value.

    Policy имеет небольшой случайный шум, чтобы избежать равных вероятностей
    и bias к меньшим индексам действий.
    """
    def __init__(self):
        self.rollout_games = 20
        self.rollout_turns = 20
        self.randomise_value = False
        self.randomise_score = True
        self.value_modifier = 1
        self.local_board = Board()


    def predict(self, state: np.ndarray, **kwargs):
        """
        Предсказание для состояния.

        Args:
            state: Тензор состояния (игнорируется)

        Returns:
            (policy, value):
                - policy: numpy array (723,) со случайными вероятностями
                - value: случайное значение от -1 до 1
        """

        #policy = np.ones(BOARD_SIZE * BOARD_SIZE * 2 + 1) / (BOARD_SIZE * BOARD_SIZE * 2 + 1)

        action_space_size = BOARD_SIZE * BOARD_SIZE * 2 + 1
        dirichlet_alpha = 0.2
        policy = np.random.dirichlet([dirichlet_alpha] * action_space_size).astype(np.float32)




        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[1, i, j] > 0:
                    self.local_board.current_state[i, j] = BLACK
                elif state[17, i, j] > 0:
                    self.local_board.current_state[i, j] = WHITE
        territories = self.local_board.get_territories()
        my_stones = (territories==BLACK).astype(np.float32)
        enemy_stones =  (territories==WHITE).astype(np.float32)

        if self.randomise_score:
            alpha_tensor = torch.full((board_size_sqr * 2 + 1,), dirichlet_alpha, dtype=torch.float32)
            scores = Dirichlet(alpha_tensor).sample()
        else:
            scores = torch.full((board_size_sqr * 2 + 1,), 0, dtype=torch.float32)
            scores[my_stones - enemy_stones + board_size_sqr] = 1

        if self.randomise_value:
            value = uniform(-1,1) * self.value_modifier
        else:
            raw_value = (np.sum(my_stones) - np.sum(enemy_stones)) * 3 / board_size_sqr
            value = max(min(raw_value, 0.9), -0.9)

        return policy.astype(np.float32), np.float32(value), scores

import torch
import torch.nn.functional as F
from rl_agent import RLAgent


class PytorchAgentWrapper(NetworkBase):
    """
    Обертка для PyTorch-модели RLAgent, предоставляющая интерфейс,
    аналогичный классу RandomNetwork.
    """

    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Инициализация обертки.
        """
        self.device = device

        # 1. Гарантируем, что модель в channels_last
        self.model = model.to(device, memory_format=torch.channels_last)
        self.model.eval()

        # 2. Компиляция графа (PyTorch 2.0+). mode="reduce-overhead" идеально для MCTS,
        # так как минимизирует задержки вызова на маленьких батчах.
        if hasattr(torch, 'compile') and device == 'cuda':
            self.model = torch.compile(self.model)

        # 3. ZERO-ALLOCATION НА GPU:
        # Вместо того чтобы каждый раз создавать тензор через .to(device),
        # мы один раз выделяем память под него в правильном формате.
        self.gpu_buffer = torch.empty(
            (1, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float32,
            device=self.device,
            memory_format=torch.channels_last
        )

    def predict(self, state_numpy: np.ndarray, **kwargs) -> (np.ndarray, np.float32, np.ndarray):
        """
        Делает предсказание с помощью нейросети.
        """
        # 1. Zero-allocation копирование.
        # torch.from_numpy не выделяет память (это view),
        # а .copy_() просто переливает биты в уже существующий GPU-буфер.
        state_view = torch.from_numpy(state_numpy).unsqueeze(0)
        self.gpu_buffer.copy_(state_view, non_blocking=True)

        # 2. inference_mode() - это более строгая и быстрая версия no_grad()
        with torch.inference_mode():
            # enabled=(self.device=='cuda') защитит от падений, если запустишь на CPU
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device == 'cuda')):
                policy_logits, value_tensor, _, score_logits = self.model(self.gpu_buffer)

        # 3. Обработка результатов
        # Обязательно кастуем к float32 перед softmax, иначе в fp16 возможны underflow/overflow
        policy_probabilities = F.softmax(policy_logits.to(torch.float32), dim=1)
        policy_probabilities = policy_probabilities.squeeze(0).cpu().numpy()

        value_scalar = value_tensor.item()

        # 4. ИСПРАВЛЕНИЕ УТЕЧКИ: score_logits нужно перенести на CPU и в numpy!
        # В старом коде он возвращался как GPU-тензор, из-за чего MCTS узлы
        # могли вечно хранить ссылки на VRAM.
        score_numpy = score_logits.squeeze(0).cpu().numpy()

        return policy_probabilities, np.float32(value_scalar), score_numpy


def create_and_get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Создаем экземпляр вашей нейросети
    pytorch_model = RLAgent()

    # (Опционально) Загружаем веса обученной модели
    # pytorch_model.load_state_dict(torch.load("best_model.pth"))

    # Создаем агента с нужным интерфейсом
    network_agent = PytorchAgentWrapper(pytorch_model, device=device)

    return network_agent
