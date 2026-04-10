import abc
from abc import abstractclassmethod
from math import gamma

from numpy.random import random, uniform
from scipy.signal import ellip
from torch.distributions import Dirichlet
from torch.nn.parallel.scatter_gather import scatter_kwargs

import board
from game import Game, decode_move, board_size_sqr

import numpy as np

from board import BOARD_SIZE, BLACK, Board, WHITE


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

        Args:
            model (torch.nn.Module): Экземпляр обученной модели (RLAgent).
            device (str): Устройство для вычислений ('cpu' или 'cuda').
        """
        self.model = model.to(device)
        self.device = device

        # Переключаем модель в режим оценки. Это важно, так как отключает
        # слои вроде Dropout и меняет поведение BatchNorm.
        self.model.eval()

    def predict(self, state_numpy: np.ndarray, **kwargs) -> (np.ndarray, np.float32, np.ndarray):
        """
        Делает предсказание с помощью нейросети.

        Args:
            state_numpy (np.ndarray): Входной тензор состояния игры
                                      в формате PyTorch (17, 19, 19).

        Returns:
            (policy, value):
                - policy (np.ndarray): Массив вероятностей ходов (723,).
                - value (np.float32): Оценка позиции от -1 до 1.
        """
        # 1. Преобразование numpy-массива в тензор PyTorch
        #    - Добавляем batch-измерение: (17, 19, 19) -> (1, 17, 19, 19)
        #    - Устанавливаем тип данных float32
        #    - Перемещаем на нужное устройство
        input_tensor = torch.from_numpy(state_numpy).unsqueeze(0).to(self.device, dtype=torch.float32)

        # 2. Выполнение предсказания
        #    Используем torch.no_grad() для отключения расчета градиентов,
        #    что ускоряет вычисления и экономит память.
        with torch.no_grad():
            policy_logits, value_tensor, _, score_logits = self.model(input_tensor)

        # 3. Обработка результатов
        #    - Преобразуем логиты в вероятности с помощью softmax
        #    - Убираем batch-измерение с помощью squeeze(0)
        #    - Перемещаем на CPU и конвертируем в numpy
        policy_probabilities = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # Извлекаем скалярное значение из тензора оценки
        value_scalar = value_tensor.item()

        return policy_probabilities, np.float32(value_scalar), score_logits


def create_and_get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Создаем экземпляр вашей нейросети
    pytorch_model = RLAgent()

    # (Опционально) Загружаем веса обученной модели
    # pytorch_model.load_state_dict(torch.load("best_model.pth"))

    # Создаем агента с нужным интерфейсом
    network_agent = PytorchAgentWrapper(pytorch_model, device=device)

    return network_agent
