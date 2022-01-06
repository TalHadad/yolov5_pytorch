# server_raspi.py
from enum import Enum
import cv2
import socket
import time
import pickle
import torch
import numpy as np

from server_handler import Server, Analyzer

class Server_RASPI(Server):
    def __init__(self, ip, port: int, target):
        super().__init__(ip, port, analyzer=Analyzer_RASPI(target))

class Analyzer_RASPI(Analyzer):
      pass
if __name__ == '__main__':
    #ip = socket.gethostname()
    ip = '192.168.1.106'
    port = 8003
    target = 'person'
    server = Server_RASPI(ip, port, target)
    server.start()