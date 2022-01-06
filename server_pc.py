# server_pc.py
from enum import Enum
import cv2
import socket
import time
import pickle
import torch
import numpy as np

from server_handler import Server, Analyzer

class Server_PC(Server):
    def __init__(self, ip, port: int, target):
        super().__init__(ip, port, analyzer=Analyzer_PC(target))

class Analyzer_PC(Analyzer):
    def __init__(self, target: str):
        super().__init(target)

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.previous_x = 0
        self.previous_y = 0

    def analyze(self, frame):
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
        return ret

if __name__ == '__main__':
    #ip = socket.gethostname()
    ip = '192.168.1.106'
    port = 8003
    target = 'person'
    server = Server_PC(ip, port, target)
    server.start()
