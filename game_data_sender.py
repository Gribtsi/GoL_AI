
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

    def send_game(self, game_id: str, game_data: dict):
        """
        Упаковывает NumPy массивы и отправляет по ZMQ.
        """
        # Теперь пакуем словарь стандартным способом (msgpack_numpy сам разберется с массивами)
        encoded_data = msgpack.packb(game_data, use_bin_type=True)

        # Отправляем фреймы: [ID игры, Байты данных]
        self.socket.send_multipart([game_id.encode('utf-8'), encoded_data])

        #print(f"Партия {game_id} отправлена на запись.")
