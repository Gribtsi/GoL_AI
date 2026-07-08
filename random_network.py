import abc
from typing import Tuple, Callable, Union

import zmq
import pickle
import numpy as np

from numpy.random import random, uniform

from numba import njit

from game import GameData
from addresses_config import get_inference_service_address
from board import build_board_data, build_board_temp, fast_territories_numba
from config import BOARD_SIZE, BLACK, WHITE, board_size_sqr, IN_CHANNELS, NN_BATCH_SIZE, possible_moves_total, pass_code
from mcts_tree import MCTS_Tree
from rollouts import Rollout


class NetworkBase:

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def predict(self, state_numpy: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.float32, np.ndarray]:
        pass

class AsyncNetworkBase(NetworkBase):
    def __init__(self, name: str):
        super().__init__(name)

    def send_request(self, state_numpy: np.ndarray, **kwargs) -> None:
        pass

    def result_ready(self) -> bool:
        pass

    def get_result(self, block: bool = False):
        pass



class RandomNetworkClient(AsyncNetworkBase):
    def __init__(self, name: str):
        super().__init__(name)
        self.network = RandomNetwork()
        self.input_cache = None

    def send_request(self, state_numpy: np.ndarray, **kwargs) -> None:
        self.input_cache = state_numpy

    def result_ready(self) -> bool:
        return True

    def get_result(self, block: bool = False):
        return self.network.predict(self.input_cache)


class ZMQNetworkClient(AsyncNetworkBase):
    def __init__(self, host=get_inference_service_address(0), name="ZMQClientAgent"):
        super().__init__(name)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        # Подключаемся к сервису в Docker
        self.socket.connect(host)

        # Защита от нарушения паттерна REQ-REP (строгое чередование send и recv)
        self._request_pending = False

    def send_request(self, state_numpy: np.ndarray, **kwargs) -> None:
        """Асинхронно отправляет запрос (не блокирует)"""
        if self._request_pending:
            raise RuntimeError("ZMQ REQ Socket Error: Нельзя отправить новый запрос, не получив результат предыдущего.")

        self.socket.send(state_numpy.tobytes(), copy=False)
        self._request_pending = True

    def result_ready(self) -> bool:
        """Мгновенно проверяет, пришел ли ответ, без блокировки потока"""
        if not self._request_pending:
            return False

        # timeout=0 означает мгновенный возврат (опрашивает наличие POLLIN событий)
        return self.socket.poll(timeout=0) != 0

    def get_result(self, block: bool = False):
        """
        Забирает результат.
        Если block=False и ответ еще не готов, возвращает None (чтобы не упасть с ошибкой).
        Если block=True, будет ждать ответа синхронно.
        """
        if not self._request_pending:
            raise RuntimeError("Нет ожидающего запроса для получения результата.")

        flags = 0 if block else zmq.NOBLOCK

        try:
            frames = self.socket.recv_multipart(flags=flags)
        except zmq.error.Again:
            # Срабатывает, если мы вызвали recv с NOBLOCK, а данных еще нет
            return None

        self._request_pending = False

        policy = np.frombuffer(frames[0], dtype=np.float32)  # shape восстанавливается из размера
        value = np.frombuffer(frames[1], dtype=np.float32)[0]  # скаляр
        score = np.frombuffer(frames[2], dtype=np.float32)
        name = frames[3].decode('utf-8')

        self.name = name
        return policy, value, score

    def predict(self, state_numpy: np.ndarray, **kwargs) -> tuple:
        """Старый блокирующий метод для обратной совместимости"""
        self.send_request(state_numpy, **kwargs)
        # Устанавливаем block=True, чтобы дождаться ответа
        return self.get_result(block=True)


class RandomNetwork(NetworkBase):
    """
    Заглушка нейросети, возвращающая случайные policy и value.

    Policy имеет небольшой случайный шум, чтобы избежать равных вероятностей
    и bias к меньшим индексам действий.
    """
    def __init__(self, name="HeuristicsAgent", random_seed: int = 42, mode: int = 0):
        super().__init__(name)
        self.rng = np.random.default_rng(random_seed)
        self.rollout = Rollout()
        self.mode = mode # 0 = full rng move, 1 = rng rollout, 2 = no eyes rollout, 3 = no_terr_dec_rollout, 4 = no_eyes_no_terr_dec_rollout;
        self.scores = np.zeros(board_size_sqr * 2 + 1 ,dtype=np.float32)

    def _dirichlet_policy(self) -> np.ndarray:
        alpha = 0.3
        policy = self.rng.dirichlet(np.full(possible_moves_total, alpha, dtype=np.float64))
        return policy.astype(np.float32)

    def _apply_policy_mask(self, policy: np.ndarray, mask: np.ndarray) -> np.ndarray:


        policy = policy * mask
        s = np.sum(policy)

        if s > 0.0:
            policy = policy / s
        else:
            policy.fill(0.0)
            policy[pass_code] = 1.0

        return policy.astype(np.float32)

    def _apply_mask_to_tree(self, tree: MCTS_Tree, mask: np.ndarray,  idx: int):
        np.copyto(tree.delta_games.next_mask[idx], mask)
        np.copyto(tree.data.game_state.legal_mask, mask)



    def predict(
        self,
        state_numpy: np.ndarray = None,
        game: MCTS_Tree = None,
        node_idx = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.float32, np.ndarray]:

        if game is None:
            self.mode = 0

        policy = self._dirichlet_policy()
        self.scores.fill(0.0)

        if self.mode == 0:
            print("No GameData provided! Fallback to random value")
            value = np.float32(self.rng.uniform(-1.0, 1.0))

        elif self.mode == 1:
            value = np.float32(self.rollout.rollout(game.data.game_state, False, False, self.rng))

        elif self.mode == 2:
            mask = self.rollout.build_policy_mask(game.data.game_state, True, False)
            policy = self._apply_policy_mask(policy, mask)
            self._apply_mask_to_tree(game, mask, node_idx)
            value = np.float32(self.rollout.rollout(game.data.game_state, True, False, self.rng))

        elif self.mode == 3:
            mask = self.rollout.build_policy_mask(game.data.game_state, False, True)
            policy = self._apply_policy_mask(policy, mask)
            self._apply_mask_to_tree(game, mask, node_idx)
            value = np.float32(self.rollout.rollout(game.data.game_state, False, True, self.rng))

        elif self.mode == 4:
            mask = self.rollout.build_policy_mask(game.data.game_state, True, True)
            policy = self._apply_policy_mask(policy, mask)
            self._apply_mask_to_tree(game, mask, node_idx)
            value = np.float32(self.rollout.rollout(game.data.game_state, True, True, self.rng))

        else:
            value = np.float32(0.0)

        return policy, value, self.scores


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
                policy_self, _, outcome_logits, territory, score_logits = self.model(self.gpu_buffer)

                # Берём только актуальную часть батча
        policy_self = policy_self[0].to(torch.float32)
        outcome_logits = outcome_logits[0].to(torch.float32)
        score_logits = score_logits[0].to(torch.float32)

        # Политики → вероятности
        policy_self_probs = F.softmax(policy_self, dim=0).cpu().numpy()

        # Исход → скалярное value ∈ (-1, +1)
        outcome_probs = torch.softmax(outcome_logits, dim=0)
        weights = outcome_logits.new_tensor([1.0, 0.0, -1.0])  # win, draw, loss
        values = (outcome_probs * weights).sum(dim=0).cpu().numpy()  # [B]

        # Счёт → вероятности по классам
        score_probs = F.softmax(score_logits, dim=0).cpu().numpy()  # [B, H*W*2+1]

        return policy_self_probs, values, score_probs


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
                policy_self, _, outcome_logits, territory, score_logits = self.model(self.gpu_buffer)

        # Берём только актуальную часть батча
        policy_self = policy_self[:actual_batch_size].to(torch.float32)
        outcome_logits = outcome_logits[:actual_batch_size].to(torch.float32)
        score_logits = score_logits[:actual_batch_size].to(torch.float32)

        # Политики → вероятности
        policy_self_probs = F.softmax(policy_self, dim=1).cpu().numpy()

        # Исход → скалярное value ∈ (-1, +1)
        outcome_probs = torch.softmax(outcome_logits, dim=1)
        weights = outcome_logits.new_tensor([1.0, 0.0, -1.0])  # win, draw, loss
        values = (outcome_probs * weights).sum(dim=1).cpu().numpy()  # [B]

        # Счёт → вероятности по классам
        score_probs = F.softmax(score_logits, dim=1).cpu().numpy()  # [B, H*W*2+1]

        return policy_self_probs, values, score_probs
