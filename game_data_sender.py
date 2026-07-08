
import zmq

from addresses_config import WRITER_ADDRESS

import msgpack
import msgpack_numpy as m

from save_data import save_game_to_hdf5

m.patch()

class GameDataSender:
    def __init__(self, writer_address=WRITER_ADDRESS):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(writer_address)

    def _send_typed(self, message_type: str, game_id: str, payload: dict):
        encoded_data = msgpack.packb(payload, use_bin_type=True)
        self.socket.send_multipart([
            message_type.encode('utf-8'),
            game_id.encode('utf-8'),
            encoded_data
        ])

    def send_game(self, game_id: str, game_data: dict):
        self._send_typed("train_game", game_id, game_data)

    def send_game_log(self, game_id: str, game_log: dict):
        self._send_typed("game_log", game_id, game_log)


class DebugDataSender:
    def __init__(self, h5_file_path: str):
        self.h5_file_path = h5_file_path

    def _send_typed(self, message_type: str, game_id: str, payload: dict):
        payload["message_type"] = message_type
        result, _ = save_game_to_hdf5(self.h5_file_path, payload, game_id=game_id)
        if result == -1:
            print(f"[DebugDataSender] Пустые данные, game_id={game_id}, type={message_type}")
        else:
            print(f"[DebugDataSender] Записано: type={message_type}, game_id={game_id}, index={result}")

    def send_game(self, game_id: str, game_data: dict):
        self._send_typed("train_game", game_id, game_data)

    def send_game_log(self, game_id: str, game_log: dict):
        self._send_typed("game_log", game_id, game_log)


class DebugDataSenderWrapper:
    def __init__(self, sender: DebugDataSender):
        self.sender = sender

    def send_game(self, game_id: str, game_data: dict):
        self.sender.send_game(game_id, game_data)

    def send_game_log(self, game_id: str, game_log: dict):
        self.sender.send_game_log(game_id, game_log)