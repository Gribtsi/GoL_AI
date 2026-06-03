# trainer_service.py
import h5py
import time
import os
import zmq
import torch
import re

from addresses_config import DATASETS_FILEPATH, MODELS_DIR, MODEL_PROVIDER_ADDRESS, WRITER_ADDRESS, \
    DATASET_COUNT_ADDRESS
from config import SAMPLES_PER_TRAINING, TRAINING_STEPS, BATCH_SIZE, LEARNING_RATE
from model_manager import ModelManager
from rl_agent import RLAgent
from train_network import train_network_steps


def get_total_samples() -> int:
    """Спрашивает у Писателя текущее количество сэмплов через ZMQ."""
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 2000)  # Таймаут 2 секунды
    socket.connect(DATASET_COUNT_ADDRESS)

    try:
        socket.send_json({"command": "GET_TOTAL_SAMPLES"})
        response = socket.recv_json()
        return response.get("total_samples", 0)
    except zmq.error.Again:
        print("data_writer не ответил (таймаут).")
        return -1
    except Exception as e:
        print(f"Ошибка запроса сэмплов: {e}")
        return -1
    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()


def notify_provider(model_name: str, provider_address=MODEL_PROVIDER_ADDRESS, max_retries=3):
    """
    Отправляет команду Провайдеру начать раздавать новую версию.
    Использует механизм Timeout и Retry для защиты от зависаний (Deadlocks).
    """
    context = zmq.Context.instance()

    for attempt in range(max_retries):
        # 1. Создаем сокет с таймаутом
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 2000)  # Ждем ответа максимум 2 секунды
        socket.connect(provider_address)

        try:
            # 2. Отправляем команду
            socket.send_json({"command": "UPDATE_MODEL", "model_name": model_name})

            # 3. Пытаемся прочитать ответ
            response = socket.recv_string()

            if response == "OK":
                print(f"Провайдер успешно переключился на {model_name}")
                socket.close()
                return True
            else:
                print(f"Провайдер ответил ошибкой: {response}")

        except zmq.error.Again:
            print(f"Таймаут (попытка {attempt + 1}/{max_retries}): Провайдер не ответил.")
        except Exception as e:
            print(f"Ошибка связи с Провайдером: {e}")

        finally:
            #4. Проверяем, жив ли еще сокет, прежде чем его трогать
            if socket and not socket.closed:
                try:
                    socket.setsockopt(zmq.LINGER, 0)
                except zmq.error.ZMQError:
                    pass
                socket.close()

    print("ВНИМАНИЕ: Не удалось уведомить Провайдера после всех попыток.")
    return False


def trainer_service():
    model_manager = ModelManager(RLAgent, save_dir=MODELS_DIR, device="cuda")

    # Получаем все файлы в папке моделей, которые подходят под шаблон agent_vX.pth
    model_files = [f for f in os.listdir(MODELS_DIR) if re.match(r'^agent_v\d+\.pth$', f)]

    if not model_files:
        # Если файлов вообще нет, создаем нулевую (инициализационную) модель
        version = 0
        current_model_name = f"agent_v{version}.pth"
        initial_model = model_manager.create_new_model()
        model_manager.save_checkpoint(initial_model, None, {'version': version}, current_model_name)
        print(f"Создана базовая модель: {current_model_name}")
    else:
        versions = [int(re.findall(r'\d+', f)[0]) for f in model_files]

        version = max(versions)
        current_model_name = f"agent_v{version}.pth"
        print(f"Найдена существующая модель: {current_model_name}")



    notify_provider(current_model_name)

    last_trained_samples = -1

    while last_trained_samples == -1:
        last_trained_samples = get_total_samples()



    print("Trainer запущен. Ожидание данных...")

    skips = 6*5
    current_skips = 0

    while True:
        current_samples = get_total_samples()
        new_samples = current_samples - last_trained_samples

        if new_samples >= SAMPLES_PER_TRAINING:
            print(f"Накоплено {new_samples} новых данных. Запускаем обучение v{version + 1}...")

            # Загружаем последнюю модель для дообучения
            model = model_manager.model_class().to(model_manager.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            model_manager.load_checkpoint(current_model_name, model, optimizer)

            # Обучаем! (Ваш метод train_network_steps)
            train_network_steps(model, optimizer, steps=TRAINING_STEPS, batch_size=BATCH_SIZE, device="cuda")

            # Сохраняем новую версию
            version += 1
            current_model_name = f"agent_v{version}.pth"
            model_manager.save_checkpoint(model, optimizer, {'version': version}, current_model_name)

            # Уведомляем провайдера
            notify_provider(current_model_name)

            last_trained_samples = current_samples
            print(f"Модель v{version} успешно выложена в продакшен.")

            current_skips = 0
        else:
            # Спим и ждем, пока воркеры нагенерируют данные
            current_skips += 1

            if current_skips >= skips:
                print(f"Ожидание новых данных {new_samples}/{SAMPLES_PER_TRAINING} собрано за {skips // 6} мин")
                current_skips = 0
            time.sleep(10)


if __name__ == "__main__":
    trainer_service()