# model_provider.py
import zmq
import os

from addresses_config import MODEL_PROVIDER_ADDRESS, MODELS_DIR


def model_provider_service():
    """
    Раздает актуальные веса агентам и принимает команды от Тренера на обновление.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP (Reply) сокет для паттерна Запрос-Ответ
    socket.bind(MODEL_PROVIDER_ADDRESS)

    os.makedirs(MODELS_DIR, exist_ok=True)
    current_model_name = "agent_v0.pth"
    current_model_path = os.path.join(MODELS_DIR, current_model_name)


    # Если начальной модели нет, ждем пока Тренер или другой скрипт ее создаст
    print(f"Model Provider запущен на {MODEL_PROVIDER_ADDRESS}")

    while True:
        # Ждем запроса от любого клиента (Worker или Trainer)
        message = socket.recv_json()
        command = message.get("command")

        if command == "GET_LATEST_MODEL":
            if os.path.exists(current_model_path):
                with open(current_model_path, "rb") as f:
                    model_bytes = f.read()
                # Отправляем путь (версию) и сами байты файла
                print(f"Провайдер передает {current_model_path}")
                socket.send_multipart([current_model_path.encode('utf-8'), model_bytes])

            else:
                socket.send_multipart([b"ERROR", b"Model not found yet"])

        elif command == "GET_MODEL_NAME":
            socket.send_multipart([current_model_name.encode('utf-8')])

        elif command == "UPDATE_MODEL":
            # Команда от Тренера о том, что появилась новая модель
            new_model_name = message.get("model_name")
            current_model_path = os.path.join(MODELS_DIR, new_model_name)
            print(f"Провайдер переключился на раздачу: {current_model_path}")
            socket.send_string("OK")

        else:
            socket.send_string("UNKNOWN_COMMAND")


if __name__ == "__main__":
    model_provider_service()