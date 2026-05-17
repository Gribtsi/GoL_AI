from typing import Tuple, Union
import gc

import zmq
import pickle
import time
import numpy as np
import os
import tempfile
import multiprocessing as mp
import torch

from addresses_config import MODEL_PROVIDER_ADDRESS, get_inference_service_address
from config import INFERENCE_TIMEOUT_MS, DEEP_DEPTH, SHALLOW_DEPTH, DEEP_SEARCH_CHANCE, MAX_MOVES_PER_GAME, \
    INFERENCE_SERVICES_COUNT, NN_BATCH_SIZE
from model_manager import ModelManager
from random_network import BatchedPytorchAgentWrapper
from rl_agent import RLAgent


class ModelClient:
    """Запрашивает актуальную модель у Провайдера через ZMQ REQ с защитой от зависаний."""

    def __init__(self, provider_address=MODEL_PROVIDER_ADDRESS, save_dir="/tmp"):
        self.context = zmq.Context.instance()
        self.provider_address = provider_address
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.socket = self._create_socket()

    def _create_socket(self):
        """Создает новый сокет REQ с таймаутом на чтение."""
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # Таймаут ожидания ответа: 5000 мс (5 секунд)
        socket.connect(self.provider_address)
        return socket

    def _reset_socket(self):
        """Закрывает зависший сокет и переподключается."""
        # Устанавливаем нулевой LINGER, чтобы сокет закрылся мгновенно, не пытаясь дослать пакеты
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.socket = self._create_socket()

    def get_latest_model(self, current_version=None) -> Tuple[Union[str, None], bool]:
        """
        Спрашивает провайдера.
        Возвращает (model_name, is_new). Если is_new == True, файл уже сохранен на диск.
        """
        try:
            self.socket.send_json({"command": "GET_LATEST_MODEL"})
        except zmq.ZMQError as e:
            print(f"Ошибка отправки запроса: {e}. Переподключение...")
            self._reset_socket()
            return None, False

        try:
            # Если провайдер не ответит за 5 секунд, выбросится zmq.error.Again
            response = self.socket.recv_multipart()
        except zmq.error.Again:
            print("Провайдер не ответил (таймаут). Сервер еще не готов или упал.")
            self._reset_socket()
            return None, False

        except Exception as e:
            print(f"Неизвестная ошибка ZMQ: {e}")
            self._reset_socket()
            return None, False

        if response[0] == b"ERROR":
            return None, False

        # Провайдер присылает имя файла (например, "agent_v3.pth")
        model_name = os.path.basename(response[0].decode('utf-8'))

        # Если у нас уже есть эта версия, не тратим время на перезапись
        if current_version == model_name:
            return model_name, False

        # Сохраняем новую версию на диск
        model_bytes = response[1]
        local_path = os.path.join(self.save_dir, model_name)
        with open(local_path, "wb") as f:
            f.write(model_bytes)

        print(f"Скачана новая модель: {model_name} ({len(model_bytes) // 1024 // 1024} MB)")
        return model_name, True

class ZMQInferenceServer:
    def __init__(self, timeout_ms=INFERENCE_TIMEOUT_MS, process_id = 0, device = 'cuda'):

        self.process_id = process_id
        self.device = device
        self.timeout_ms = timeout_ms
        self.max_batch_size = NN_BATCH_SIZE

        self.check_for_updates_interval = 10
        self.last_update_time = 0

        self.current_model_version = None
        self.model_client = None
        self.model_manager = None
        self.model = None

        self.context = None
        self.socket = None
        self.poller = None

    def check_for_updates(self) -> bool:
        new_version, is_updated = self.model_client.get_latest_model(self.current_model_version)
        if new_version is None:
            return False

        if is_updated or self.current_model_version is None:
            print(f"=== Инициализация агента {new_version} ===")
            best_model = self.model_manager.load_model_weights(new_version)

            if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
                torch.compiler.reset()
            elif hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
                torch._dynamo.reset()

            self.model = BatchedPytorchAgentWrapper(best_model, device=self.device)
            self.max_batch_size = self.model.max_batch_size


            self.current_model_version = new_version

            return True
        return False

    def real_init(self):
        self.check_for_updates_interval = 10
        self.last_update_time = 0

        self.current_model_version = None

        temp_dir = tempfile.gettempdir()
        process_model_dir = os.path.join(temp_dir, f"inference_models_p{self.process_id}")
        os.makedirs(process_model_dir, exist_ok=True)

        self.model_client = ModelClient(save_dir=process_model_dir)
        self.model_manager = ModelManager(RLAgent, save_dir=process_model_dir, device=self.device)


        cache_root = tempfile.gettempdir()
        os.environ["TRITON_CACHE_DIR"] = f"{cache_root}/triton_cache/p{self.process_id}"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{cache_root}/torchinductor_cache/p{self.process_id}"

        os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
        os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)


        self.model = None


        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)

        self.socket.bind(get_inference_service_address(self.process_id))

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def run(self):

        gc.enable()

        self.real_init()

        # Сюда нормальный адрес
        print(f"Inference server listening on {get_inference_service_address(self.process_id)}... Max batch: {self.max_batch_size}")

        while self.current_model_version is None:
            self.check_for_updates()
            if self.current_model_version is None:
                print("Модель еще не готова, ждем 5 секунд...")
                time.sleep(5)

        print(f"Получена модель {self.current_model_version}")

        while True:
            identities = []
            empty_frames = []
            actual_batch_size = 0

            # 1. Ждем первый запрос (блокирующе, без таймаута)
            socks = dict(self.poller.poll(timeout=None))
            if self.socket in socks:
                req = self.socket.recv_multipart()
                identities.append(req[0])
                empty_frames.append(req[1])
                # Сразу записываем в pinned память по нулевому индексу
                self.model.add_input(actual_batch_size, pickle.loads(req[2]))
                actual_batch_size += 1

            # 2. Собираем остальные запросы в окно таймаута
            start_time = time.time()
            while actual_batch_size < self.max_batch_size:
                elapsed_ms = (time.time() - start_time) * 1000
                remaining_time = max(0, self.timeout_ms - elapsed_ms)

                if remaining_time <= 0:
                    break

                socks = dict(self.poller.poll(timeout=remaining_time))
                if self.socket in socks:
                    req = self.socket.recv_multipart()
                    identities.append(req[0])
                    empty_frames.append(req[1])

                    self.model.add_input(actual_batch_size, pickle.loads(req[2]))
                    actual_batch_size += 1
                else:
                    break

            # 3. Делаем 1 прямой проход на собранном батче
            policies, values, scores = self.model.predict(actual_batch_size)

            # 4. Рассылаем результаты воркерам
            for i in range(actual_batch_size):
                # Приводим к форматам, которые ожидает клиент (MCTS)
                # policy_probabilities[i] уже np.ndarray, score_numpy тоже.
                result = (policies[i], np.float32(values[i]), scores[i], self.current_model_version)
                reply_payload = pickle.dumps(result)

                self.socket.send_multipart([identities[i], empty_frames[i], reply_payload])


            if time.time() > self.last_update_time + self.check_for_updates_interval:
                self.check_for_updates()
                self.last_update_time = time.time()


if __name__ == "__main__":
    mp.set_start_method('fork')
    gc.freeze()


    processes = []
    for i in range(INFERENCE_SERVICES_COUNT):
        service = ZMQInferenceServer(process_id=i)
        p = mp.Process(target=service.run)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

