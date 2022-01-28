# client_simple.py
import socket
import logging # debug, info, warning, error, critical
from controller import Controller, ControllerSimple
from comunication import HEADERSIZE, receive, send

class Client():

    def __init__(self, ip, port: int, controller: Controller):
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
                image = self.controller.get_image() # (type <class 'numpy.ndarray'>)
                send(self.server_socket, image)
                action = int(receive(self.server_socket))
                self.controller.do_action(action)
        except Exception as e:
            # WARNING:root:client stopped, exiting clean, exception
            # invalid literal for int() with base 10: b''
            logging.warning(f'client stopped, exiting clean, exception {e}')

            self.controller.exit_clean()



if __name__ == '__main__':
    ip = '192.168.1.106'
    port = 8003
    client = Client(ip, port, controller=ControllerSimple())
    client.start()