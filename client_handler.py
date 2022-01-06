# client_handler.py
from enum import Enum
import socket
import pickle
import cv2
import time
import numpy as np

from server_handler import MessageType, HEADERSIZE

class Client():

    def __init__(self, ip, port: int, controller):
        self.connect_socket(ip, port)
        self.controller = controller

    def process_message(self, msg):
        if msg == MessageType.WELCOME:
            print(msg)
        elif msg == MessageType.CLOSE:
            print(msg)
            self.controller.clean_exit()
        else:
            previous_x, previous_y, current_x, current_y = msg
            try:
                self.controller.move_car(previous_x, previous_y, current_x, current_y)
            except:
                self.controller.clean_exit()

    def connect_socket(self, ip:str, port:int ):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.connect((ip, port))
            print ("Socket successfully created")
        except socket.error as err:
            print ("socket creation failed with error %s" %(err))

    def start(self):
        new_msg = True
        full_msg = b''
        while True:
            msg = self.server_socket.recv(16)
            if new_msg:
                print(f"new message length: {msg[:HEADERSIZE]}")
                msglen = int(msg[:HEADERSIZE])
                new_msg = False

            #full_msg += msg.decode("utf-8")
            full_msg += msg

            if len(full_msg)-HEADERSIZE == msglen:
                print("full msg recvd")
                #print(full_msg[HEADERSIZE:])

                message = pickle.loads(full_msg[HEADERSIZE:])
                #print(f'{message}')
                #print(message)
                self.process_message(message)

                image = self.controller.take_a_picture()
                msg = pickle.dumps(image)
                msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg
                self.server_socket.send(msg)

                new_msg = True
                full_msg = b''


class Controller():

    def clean_exit(self):
        print("clean exit.")

    def move_car(self, previous_x, previous_y, current_x, current_y):
        # first measurment
        if previous_x==0 and previous_y==0:
            print("first measurment: doing nothing")

        # stop (do nothing)
        if previous_x==current_x and previous_y==current_y:
            print("target isn't moving: doing nothing.")

        # Forward
        elif previous_x==current_x and previous_y>current_y:
            print("target moved backward: moving forward.")

            # forward_left
        elif previous_x>current_x and previous_y>current_y:
            print('target moved left and backward: moving left and forward.')

            # forward_right
        elif previous_x<current_x and previous_y>current_y:
            print('target moved right and backward: moving right and forward.')

            # backward
        elif previous_x==current_x and previous_y<current_y:
            print('target moved forward: moving backward.')

            # backward_left
        elif previous_x<current_x and previous_y<current_y:
            print('target moved right and forward: moving left and backward.')
            # backward_right

        elif previous_x>current_x and previous_y<current_y:
            print('target moved left and forward: moving right and backward.')

    def take_a_picture(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            print('Could not read frame.')
        cap.release()
        return frame

if __name__ == '__main__':
    ip = '192.168.1.106'
    port = 8003
    client = Client(ip, port, controller=Controller())
    client.start()