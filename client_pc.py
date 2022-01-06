# client_pc.py
from enum import Enum
import socket
import pickle
import cv2
import time
import numpy as np

import client_handler

class Client_PC(client_handler.Client):

    def __init__(self, ip, port: int):
        super().__init__(ip, port, controller=Controller_PC())

class Controller_PC(client_handler.Controller):
    pass

if __name__ == '__main__':
    ip = '192.168.1.106'
    port = 8003
    client = Client_PC(ip, port)
    client.start()