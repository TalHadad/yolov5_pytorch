# server.py
import traceback
import socket as s
import logging
from utils.logging_level import LOGGING_LEVEL
logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('server')
log.setLevel(LOGGING_LEVEL)

import gym

from server.detector import Detector, Detector_Yolov5
from server.agent import Agent, AgentDDPG, AgentEnv
from utils.comunication import receive, send


class Server():
    def __init__(self, ip, port: int, detector: Detector, agent: Agent):
        self.socket = self._bind_socket(ip, port)
        self.detector = detector
        self.agent = agent

    def _bind_socket(self, ip: str, port: int):
        socket = None
        try:
            socket = s.socket(s.AF_INET, s.SOCK_STREAM)
            socket.bind((ip, port))
            socket.listen(5)  # param = number of clients
            log.info(f'server is binded to {ip}:{port}.')
        except s.error as err:
            log.error(f'socket {ip}:{port} creation failed with error {err}')
        return socket

    def start(self) -> None:
        log.info('starting server.')
        self._wait_for_client()
        self.agent.load_models()
        try:
            while True:
                image = receive(self.client_socket)
                log.info(f'received from client image (type {type(image)}).')
                location = self.detector.get_location(image)
                log.info(f'got location {location}.')
                self.detector.render()
                action = self.agent.choose_action_and_prep_with_env_simple(location)
                log.info(f'selected action {action}.')
                send(self.client_socket, action)
                log.info(f'sent to client.')
        except Exception as e:
            log.warning(f'server stopped, exiting clean, exception {e}')
            traceback.print_exception(type(e), e, e.__traceback__)
            self.exit_clean()

    def _wait_for_client(self):
        self.client_socket, address = self.socket.accept()
        log.info(f'Connection from {address} has been established!')

    def virtual_start(self):
        log.info('virtual starting server.')
        env = gym.make('Mouse-v0')
        env.reset()
        # self.agent.load_models()
        try:
            while True:
                location = env.detector_get_location()
                log.info(f'got location {location}.')
                #env.detector_render()
                action = self.agent.choose_action_and_prep_with_env_simple(location)
                log.info(f'selected action {action}.')
                env.controller_do_action(action)
        except Exception as e:
            log.warning(f'server stopped, exiting clean, exception {e}')
            self.exit_clean()

    def exit_clean(self):
        self.detector.exit_clean()
        self.agent.exit_clean()
        self.socket.close()

if __name__ == '__main__':
    # ip = socket.gethostname()
    ip = '192.168.1.106' # home
    #ip = '172.20.85.180' # be-all
    port = 8003
    target = 'person'

    input_dims = (2,)
    n_actions = 7
    env = AgentEnv(input_dims, n_actions)
    server = Server(ip, port, Detector_Yolov5(target), AgentDDPG(env=env))

    server.start()
    #server.virtual_start()
