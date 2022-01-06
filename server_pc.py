
# server.py
from enum import Enum
import cv2
import socket
import time
import pickle
import torch
import numpy as np

HEADERSIZE = 10

class MessageType(Enum):
    WELCOME = "welcome to the server"
    CLOSE = "closing connection"

class Server_PC():
    def __init__(self, ip, port: int, target: str):
        self.bind_socket(ip, port)
        self.analyzer = Analyzer(target)

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
    def __init__(self, target: str):

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.previous_x = 0
        self.previous_y = 0
        self.target = target
        print(f'target is : {self.target}')

    def process_image(self, frame):
        ret = (0, 0, 0, 0)
        start = time.time()
        cv2.imshow('frame', frame)


        results = self.model(frame)
        cv2.imshow('detection result', np.asarray(results.imgs[0], dtype=np.uint8))
        cv2.imshow('result', np.asarray(results.xyxy[0], dtype=np.uint8))
        #results.show()
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()

        results_df = results.pandas().xyxy[0]
        print(f'results_df:\n{results_df}')  # img1 predictions (pandas)

        target_df = results_df[results_df['name'].isin([target])] # filter on spesific values
        print(f'target_df.size is {target_df.shape[0]}')
        print(f'target_df:\n{target_df}')
        if target_df.shape[0] > 1:
            print(f'Error: found multipule ({target_df.size}) targets')
        elif target_df.shape[0] == 1:
            current_x = target_df["xmin"].values[0]
            current_y = target_df["ymin"].values[0]
            print(f'found {target} in x={current_x} and y={current_y}')
            ret = (self.previous_x, self.previous_y, current_x, current_y)
            self.previous_x = current_x
            self.previous_y = current_y

            self.print_time(start)

        return ret

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
    server = Server_PC(ip, port, target)
    server.start()
