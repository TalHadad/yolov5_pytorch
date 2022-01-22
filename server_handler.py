# server_handler.py
from abc import ABC, abstractmethod

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
    FOR

class Server():
    def __init__(self, ip, port: int, analyzer):
        print(f'binding server to {ip}:{port}')
        self._bind_socket(ip, port)
        self.analyzer = analyzer

    def _bind_socket(self, ip: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((ip, port))

    def _pack_message(self, msg: str) -> bytes:
        msg = pickle.dumps(msg)
        msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
        return msg

    def _unpack_message(self, msg: bytes) -> str:
        msg = pickle.loads(msg[HEADERSIZE:])
        return msg

    def start(self):
        self.socket.listen(5) # arg = number of clients

        while True:
            client_socket, address = self.socket.accept()
            print(f'Connection from {address} has been established!')

            msg = self._pack_message(MessageType.WELCOME)
            print(f'sending welcome message.')
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

                    image = self._unpack_message(full_msg)
                    cat_coordinates = self.analyzer.process_image(image)
                    action = self.agent.get w
                    send_msg = self._pack_message(response)
                    client_socket.send(send_msg)

                    new_msg = True
                    full_msg = b''

class Analyzer(ABC):
    def __init__(self):
        self.target = TARGET
        print(f'target is : {self.target}')

        self.model = self.get_model()
        print(f'build model')

        self.x = 0
        self.y = 0

    @abstractmethod
    def get_model(self):
        pass

    def process_image(self, frame) -> tuple:
        coords_diff = (0, 0, 0, 0)
        start = time.time()

        labeled_img = self.get_labeled_image(frame)
        cv2.imshow('result', np.asarray(labeled_img, dtype=np.uint8))
        #if cv2.waitKey(1) == ord('q'):
        #    cv2.destroyAllWindows()
        cv2.waitKey(1)

        current_x, current_y = self.get_target_coords()
        if current_x!=0 or current_y!=0:
            print(f'found {self.target} in x={current_x} and y={current_y}')
            coords_diff = self.x, self.y, current_x, current_y
            self.x = current_x
            self.y = current_y

        self.print_time(start)

        return coords_diff

    @abstractmethod
    def get_labeled_image(self, frame):
        pass
    @abstractmethod
    def get_target_coords(self):
        pass

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
