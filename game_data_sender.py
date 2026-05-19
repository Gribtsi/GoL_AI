
import zmq

from addresses_config import WRITER_ADDRESS

import msgpack
import msgpack_numpy as m


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