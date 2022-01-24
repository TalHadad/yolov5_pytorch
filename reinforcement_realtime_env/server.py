# server.py
from abc import ABC, abstractmethod

from enum import Enum
import cv2
import socket
import time
import pickle
import numpy as np
from detector import Detector, Detector_Yolov5
from agent import Agent, Agent_DDPG

HEADERSIZE = 10
def receive(socket) -> str:
    got_full_msg = False
    is_new_msg = True
    full_msg = b''
    while not got_full_msg:
        part_msg = socket.recv(16)
        if is_new_msg:
            len_msg = int(part_msg[:HEADERSIZE])
            is_new_msg = False
            logging.debug(f"got new message length: {len_msg}")

        full_msg += part_msg

        if len(full_msg)-HEADERSIZE == len_msg:
            msg = full_msg[HEADERSIZE:]
            got_full_msg = True

    msg = pickle.loads(msg)
    logging.debug(f'full msg received: {msg}')
    return msg

def send(socket, msg) -> None:
    msg = pickle.dumps(msg)
    msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
    socket.send(msg)

class Server():
    def __init__(self, ip, port: int, detector: Detector, agent: Agent):
        self.socket = self._bind_socket(ip, port)
        self.detector = detector
        self.agent = agent

    def _bind_socket(self, ip: str, port: int):
        socket = None
        try:
            socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socket.bind((ip, port))
            socket.listen(5) # param = number of clients
            logging.info(f'server is binded to {ip}:{port}.')
        except socket.error as err:
            logging.error(f'socket {ip}:{port} creation failed with error {err}')
        return socket

    def start(self) -> None:
        logging.info('starting server.')
        self._wait_for_client()
        try:
            while True:
                image = receive(self.socket)
                location = self.detector.get_location(image)
                action = self.agent.get_action(location)
                send(self.client_socket, action)
        except:
            logging.warning(f'server stopped, exiting clean')
            self.detector.exit_clean()
            self.agent.exit_clean()

    def _wait_for_client():
        self.client_socket, address = self.socket.accept()
        logging.info(f'Connection from {address} has been established!')

if __name__ == '__main__':
    #ip = socket.gethostname()
    ip = '192.168.1.106'
    port = 8003
    target = 'person'
    server = Server(ip, port, Detector_Yolov5(target), Agent_DDPG())
    server.start()
