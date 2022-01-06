# server_handler.py
from enum import Enum
import cv2
import socket
import time
import pickle
import numpy as np

HEADERSIZE = 10

TARGET = 'person'

class MessageType(Enum):
    WELCOME = "welcome to the server"
    CLOSE = "closing connection"

class Server():
    def __init__(self, ip, port: int, analyzer):
        self.bind_socket(ip, port)
        self.analyzer = analyzer

    def bind_socket(self, ip, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((ip, port))

    def start(self):
        self.socket.listen(5) # arg = number of clients

        while True:
            client_socket, address = self.socket.accept()
            print(f'Connection from {address} has been established!')

            msg = pickle.dumps(MessageType.WELCOME)
            print(f'sending.')

            msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
            client_socket.send(msg)

            new_msg = True
            full_msg = b''
            while True:
                msg = client_socket.recv(16)
                if new_msg:
                    print(f"new message length: {msg[:HEADERSIZE]}")
                    msglen = int(msg[:HEADERSIZE])
                    new_msg = False

                #full_msg += msg.decode("utf-8")
                full_msg += msg

                if len(full_msg)-HEADERSIZE == msglen:
                    #print(f"full msg recvd, size {len(full_msg[HEADERSIZE:])}")
                    #print(full_msg[HEADERSIZE:])

                    message = pickle.loads(full_msg[HEADERSIZE:])
                    ret = self.analyzer.process_image(message)
                    msg = pickle.dumps(ret)
                    msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
                    client_socket.send(msg)

                    new_msg = True
                    full_msg = b''

class Analyzer():
    def __init__(self):
        self.target = TARGET
        print(f'target is : {self.target}')

    def process_image(self, frame):
        ret = (0, 0, 0, 0)
        start = time.time()
        cv2.imshow('frame', frame)

        ret = self.analyze(frame)

        self.print_time(start)

        return ret

    def analyze(self, frame):
        # Should be implemented in child class
        return (0, 0, 0, 0)

    def print_time(self, start):

        stop = time.time()
        seconds = stop - start
        print(f'Time taken : {seconds} seconds')
        # Calcutate frames per seconds
        fps = 1 / seconds
        print(f'Estimated frames per second : {fps}')


if __name__ == '__main__':
    #ip = socket.gethostname()
    ip = '192.168.1.106'
    port = 8003
    target = 'person'
    server = Server(ip, port, target)
    server.start()