# client_handler.py
import socket
import pickle
import logging # debug, info, warning, error, critical
from server import HEADERSIZE
from controller import Controller_RPi

class Client():

    def __init__(self, ip, port: int, controller):
        self.server_socket = self._connect_socket(ip, port)
        self.controller = controller

    def _connect_socket(self, ip:str, port:int):
        server_socket = None
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.connect((ip, port))
            logging.info(f'client is connected to server {ip}:{port}.')
        except socket.error as err:
            logging.error(f'socket {ip}:{port} creation failed with error {err}.')
        return server_socket

    def start(self) -> None:
        logging.info('starting client.')
        try:
            while True:
                image = self.controller.get_image()
                self._send(image)
                action = int(self._receive())
                self.controller.do_action(action)
        except:
            logging.warning(f'client stopped, exiting clean.')
            self.exit_clean()

    def _send(self, image) -> None:
        msg = pickle.dumps(image)
        msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
        self.server_socket.send(msg)

    def _receive() -> str:
        got_full_msg = False
        is_new_msg = True
        full_msg = b''
        while not got_full_msg:
            part_msg = self.server_socket.recv(16)
            if is_new_msg:
                len_msg = int(part_msg[:HEADERSIZE])
                is_new_msg = False
                logging.debug(f"got new message length: {len_msg}")

            full_msg += part_msg

            if len(full_msg)-HEADERSIZE == len_msg:
                msg = full_msg[HEADERSIZE:]
                got_full_msg = True
                logging.debug(f'full msg received: {}')

        msg = pickle.loads(msg)
        return msg

    def exit_clean() -> None:
        self.controller.exit_clean()

if __name__ == '__main__':
    ip = '192.168.1.106'
    port = 8003
    client = Client(ip, port, controller=Controller_RPi())
    client.start()