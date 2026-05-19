import abc
import zmq
import pickle
import numpy as np

from numpy.random import random, uniform

from addresses_config import get_inference_service_address
from board import Board
from config import BOARD_SIZE, BLACK, WHITE, board_size_sqr, IN_CHANNELS, NN_BATCH_SIZE, possible_moves_total


class NetworkBase:

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def predict(self, state_numpy: np.ndarray, **kwargs) -> (np.ndarray, np.float32, np.ndarray):
        pass




class ZMQNetworkClient(NetworkBase):
    def __init__(self, host=get_inference_service_address(0), name="ZMQClientAgent"):
        super().__init__(name)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        # Подключаемся к сервису в Docker
        self.socket.connect(host)

    def predict(self, state_numpy: np.ndarray, **kwargs) -> tuple:
        # Сериализуем numpy-массив (pickle делает это максимально эффективно для np.ndarray)
        payload = pickle.dumps(state_numpy)

        # Отправляем запрос
        self.socket.send(payload)

        # Блокируем выполнение, пока не придет ответ от инференс-сервера
        reply = self.socket.recv()


        # Распаковываем результат
        policy, value, score, name = pickle.loads(reply)


        self.name = name

        return policy, value, score


class RandomNetwork(NetworkBase):
    """
    Заглушка нейросети, возвращающая случайные policy и value.

    Policy имеет небольшой случайный шум, чтобы избежать равных вероятностей
    и bias к меньшим индексам действий.
    """
    def __init__(self, name="RandomAgent"):
        super().__init__(name)
        self.rollout_games = 20
        self.rollout_turns = 20
        self.randomise_value = False
        self.randomise_score = False
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

        action_space_size = possible_moves_total
        dirichlet_alpha = 0.2
        policy = np.random.dirichlet([dirichlet_alpha] * action_space_size).astype(np.float32)


        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[1, i, j] > 0:
                    self.local_board.current_state[i, j] = BLACK
                elif state[17, i, j] > 0:
                    self.local_board.current_state[i, j] = WHITE

        my_stones, enemy_stones = self.local_board.fast_territories()

        if self.randomise_score:
            alpha_array = np.full((board_size_sqr * 2 + 1,), dirichlet_alpha, dtype=np.float32)
            scores = np.random.dirichlet(alpha_array).astype(np.float32)
        else:
            scores = np.zeros((board_size_sqr * 2 + 1,), dtype=np.float32)
            scores[my_stones - enemy_stones + board_size_sqr] = 1.0

        if self.randomise_value:
            value = uniform(-1,1) * self.value_modifier
        else:
            raw_value = (my_stones - enemy_stones) * 3 / board_size_sqr
            value = max(min(raw_value, 0.9), -0.9)

        return policy.astype(np.float32), np.float32(value), scores

import torch
import torch.nn.functional as F

class PytorchAgentWrapper(NetworkBase):
    """
    Обертка для PyTorch-модели RLAgent, предоставляющая интерфейс,
    аналогичный классу RandomNetwork.
    """

    def __init__(self, model: torch.nn.Module, name="AgentWrapper", device: str = 'cpu'):
        """
        Инициализация обертки.
        """
        super().__init__(name)
        self.device = device

        # 1. Гарантируем, что модель в channels_last
        self.model = model.to(device, memory_format=torch.channels_last)
        self.model.eval()

        # 2. Компиляция графа (PyTorch 2.0+). mode="reduce-overhead" только для линуха

        if hasattr(torch, 'compile') and device == 'cuda':
            self.model = torch.compile(self.model)

        # 3. меньше выделений НА GPU:
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
            # enabled=(self.device=='cuda') защитит от падений, если на CPU
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device == 'cuda')):
                policy_logits, value_tensor, _, score_logits = self.model(self.gpu_buffer)

        # 3. Обработка результатов
        # Обязательно кастуем к float32 перед softmax, иначе в fp16 возможны underflow/overflow
        policy_probabilities = F.softmax(policy_logits.to(torch.float32), dim=1)
        policy_probabilities = policy_probabilities.squeeze(0).cpu().numpy()

        value_scalar = value_tensor.item()

        score_numpy = score_logits.squeeze(0).cpu().numpy()

        return policy_probabilities, np.float32(value_scalar), score_numpy


class BatchedPytorchAgentWrapper:
    """
    Батчевая обертка для PyTorch-модели RLAgent.
    Поддерживает zero-allocation и защиту от рекомпиляций torch.compile.
    """

    def __init__(self, model: torch.nn.Module, device: str = 'cpu', name="BatchedAgent"):
        self.name=name
        self.device = device
        self.max_batch_size = NN_BATCH_SIZE

        # 1. Гарантируем, что модель в channels_last
        self.model = model.to(device, memory_format=torch.channels_last)
        self.model.eval()

        # 2. Компиляция графа (рекомендуется max-autotune для инференса)
        if hasattr(torch, 'compile') and device == 'cuda':
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # 3. Выделяем Pinned Memory в RAM. Это позволяет делать неблокирующее копирование
        # на GPU. Получаем из нее numpy-view, куда будем складывать массивы от воркеров.
        self.pinned_cpu_tensor = torch.zeros(
            (NN_BATCH_SIZE, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32
        ).pin_memory()
        self.cpu_buffer = self.pinned_cpu_tensor.numpy()

        # 4. Выделяем память на GPU один раз (всегда фиксированного размера!)
        self.gpu_buffer = torch.empty(
            (NN_BATCH_SIZE, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float32,
            device=self.device,
            memory_format=torch.channels_last
        )

    def add_input(self, index: int, state_numpy: np.ndarray):
        """
        Заполняет слот батча данными от воркера.
        Не выделяет память, просто перезаписывает значения в RAM.
        """
        self.cpu_buffer[index] = state_numpy

    def predict(self, actual_batch_size: int) -> tuple:
        """
        Отправляет собранный батч на GPU и делает предсказание.
        """
        # 1. Zero-allocation копирование по PCI-e
        # Копируем только фактически заполненную часть, чтобы не гонять мусор по шине
        self.gpu_buffer[:actual_batch_size].copy_(
            self.pinned_cpu_tensor[:actual_batch_size], non_blocking=True
        )
        torch.cuda.synchronize()

        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device == 'cuda')):
                # ВАЖНО: Всегда подаем полный self.gpu_buffer (размер max_batch_size).
                # Это гарантирует, что torch.compile не будет рекомпилировать граф.
                policy_logits, value_tensor, _, score_logits = self.model(self.gpu_buffer)

        # 3. Обработка результатов (берем только полезную часть до actual_batch_size)
        valid_policy_logits = policy_logits[:actual_batch_size].to(torch.float32)
        policy_probabilities = F.softmax(valid_policy_logits, dim=1).cpu().numpy()

        # value_tensor обычно имеет размер (B, 1), flatten() сделает (B,)
        values = value_tensor[:actual_batch_size].cpu().numpy().flatten()

        scores = score_logits[:actual_batch_size].cpu().numpy()

        return policy_probabilities, values, scores
