# client_pc.py
from enum import Enum
import socket
import pickle
import cv2
import time
import numpy as np

from client_handler import Client, Controller


class Client_PC(Client):

    def __init__(self, ip, port: int):
        super().__init__(ip, port, controller=Controller_PC())

class Controller_PC(Controller):
    pass

if __name__ == '__main__':
    ip = '192.168.1.106'
    port = 8005
    client = Client_PC(ip, port)
    client.start()