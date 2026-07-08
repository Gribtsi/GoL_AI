# model_provider.py
import zmq
import os
import argparse

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



def model_provider_tournament():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.bind(MODEL_PROVIDER_ADDRESS)

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_a_name:  str   = None
    model_b_name:  str   = None
    model_a_bytes: bytes = None
    model_b_bytes: bytes = None

    print(f"[ModelProvider/tournament] Listening on {MODEL_PROVIDER_ADDRESS}")
    print(f"[ModelProvider/tournament] Waiting for SET_TOURNAMENT_MODELS...")

    while True:
        try:
            message = socket.recv_json()
        except zmq.error.Again:
            continue

        command = message.get("command")

        # ── Новая команда от TournamentManager ──────────────────────
        if command == "SET_TOURNAMENT_MODELS":
            name_a = message.get("model_a")
            name_b = message.get("model_b")
            path_a = os.path.join(MODELS_DIR, name_a)
            path_b = os.path.join(MODELS_DIR, name_b)

            try:
                with open(path_a, "rb") as f:
                    model_a_bytes = f.read()
                with open(path_b, "rb") as f:
                    model_b_bytes = f.read()
                model_a_name = name_a
                model_b_name = name_b

                print(
                    f"[ModelProvider] A={name_a} ({len(model_a_bytes)//1024//1024}MB), "
                    f"B={name_b} ({len(model_b_bytes)//1024//1024}MB)"
                )
                socket.send_string("OK")

            except FileNotFoundError as e:
                print(f"[ModelProvider] ERROR: {e}")
                socket.send_multipart([b"ERROR", str(e).encode()])

        # ── Инференс запрашивает модель ──────────────────────────────
        elif command == "GET_LATEST_MODEL":
            if model_a_bytes is None:
                socket.send_multipart([b"ERROR", b"models not set yet"])
                continue

            inference_index = message.get("inference_index", 0)
            if inference_index % 2 == 0:
                name, data = model_a_name, model_a_bytes
            else:
                name, data = model_b_name, model_b_bytes

            socket.send_multipart([name.encode("utf-8"), data])

        elif command == "GET_MODEL_NAME":
            inference_index = message.get("inference_index", 0)
            if model_a_name is None:
                socket.send_multipart([b"ERROR", b"models not set yet"])
            elif inference_index % 2 == 0:
                socket.send_multipart([model_a_name.encode("utf-8")])
            else:
                socket.send_multipart([model_b_name.encode("utf-8")])

        else:
            socket.send_string("UNKNOWN_COMMAND")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="selfplay",
        choices=["selfplay", "tournament"],
    )
    args = parser.parse_args()

    if args.mode == "tournament":
        model_provider_tournament()
    else:
        model_provider_service()