# client.py
import traceback
import socket
import logging # debug, info, warning, error, critical
from utils.logging_level import LOGGING_LEVEL
logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('client')
log.setLevel(LOGGING_LEVEL)

from client.controller import Controller
from utils.comunication import receive, send

class Client():

    def __init__(self, ip, port: int, controller: Controller):
        self.server_socket = self._connect_socket(ip, port)
        self.controller = controller

    def _connect_socket(self, ip:str, port:int):
        server_socket = None
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.connect((ip, port))
            log.info(f'client is connected to server {ip}:{port}.')
        except socket.error as err:
            log.error(f'socket {ip}:{port} creation failed with error {err}.')
        return server_socket

    def start(self) -> None:
        log.info('starting client.')
        try:
            while True:
                print(f'getting image')
                image = self.controller.get_image()
                print(f'sending image (type {type(image)}).')
                send(self.server_socket, image)
                print(f'waiting for server action.')
                action = int(receive(self.server_socket))
                print(f'got from server action {action}')
                self.controller.do_action(action)
        except Exception as e:
            log.warning(f'client stopped, exiting clean, exception {e}')
            traceback.print_exception(type(e), e, e.__traceback__)
            self.controller.exit_clean()



if __name__ == '__main__':
    ip = '192.168.1.106' # home
    #ip = '172.20.85.180' # be-all
    port = 8003

    from client.controller_raspi import Controller_RPi
    client = Client(ip, port, controller=Controller_RPi())

    #from controller import ControllerSimple
    #client = Client(ip, port, controller=ControllerSimple())

    client.start()
