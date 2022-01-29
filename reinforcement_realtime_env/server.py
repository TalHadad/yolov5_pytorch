# server.py
import socket as s
import logging
logging.basicConfig(level=logging.INFO)

import gym
import gym_mouse_lib.gym_mouse

from detector import Detector, Detector_Yolov5
from agent import Agent, AgentDDPG, AgentEnv
from comunication import receive, send


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
            logging.info(f'server is binded to {ip}:{port}.')
        except s.error as err:
            logging.error(f'socket {ip}:{port} creation failed with error {err}')
        return socket

    def start(self) -> None:
        logging.info('starting server.')
        self._wait_for_client()
        #self.agent.load_models()
        try:
            while True:
                image = receive(self.client_socket)
                logging.info(f'received from client image (type {type(image)}).')
                location = self.detector.get_location(image)
                logging.info(f'got location {location}.')
                self.detector.render()
                action = self.agent.choose_action_and_prep_with_env_simple(location)
                logging.info(f'selected action {action}.')
                send(self.client_socket, action)
                logging.info(f'sent to client.')
        except Exception as e:
            logging.warning(f'server stopped, exiting clean, exception {e}')
            self.detector.exit_clean()
            self.agent.exit_clean()

    def _wait_for_client(self):
        self.client_socket, address = self.socket.accept()
        logging.info(f'Connection from {address} has been established!')

    def virtual_start(self):
        logging.info('virtual starting server.')
        env = gym.make('Mouse-v0')
        env.reset()
        # self.agent.load_models()
        try:
            while True:
                location = env.detector_get_location()
                logging.info(f'got location {location}.')
                #env.detector_render()
                action = self.agent.choose_action_and_prep_with_env_simple(location)
                logging.info(f'selected action {action}.')
                env.controller_do_action(action)
        except Exception as e:
            logging.warning(f'server stopped, exiting clean, exception {e}')
            self.agent.exit_clean()

if __name__ == '__main__':
    # ip = socket.gethostname()
    ip = '192.168.1.106'
    port = 8003
    target = 'person'

    input_dims = (2,)
    n_actions = 7
    env = AgentEnv(input_dims, n_actions)
    server = Server(ip, port, Detector_Yolov5(target), AgentDDPG(env=env))

    #server.start()
    server.virtual_start()
