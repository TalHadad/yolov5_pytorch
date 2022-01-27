# server.py
from abc import ABC, abstractmethod

from enum import Enum
import logging
import socket as s
logging.basicConfig(level=logging.INFO)
from detector import Detector, Detector_Yolov5
from agent import Agent, Agent_DDPG
from comunication import HEADERSIZE, receive, send


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
                image = receive(self.socket)
                location = self.detector.get_location(image)
                # self.detector.render()
                action = self.agent.choose_action_and_prep(location)
                send(self.client_socket, action)
        except:
            logging.warning(f'server stopped, exiting clean')
            self.detector.exit_clean()
            self.agent.exit_clean()

    def _wait_for_client(self):
        self.client_socket, address = self.socket.accept()
        logging.info(f'Connection from {address} has been established!')


if __name__ == '__main__':
    # ip = socket.gethostname()
    ip = '192.168.1.106'
    port = 8003
    target = 'person'
    server = Server(ip, port, Detector_Yolov5(target), Agent_DDPG())
    server.start()
