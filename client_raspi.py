# client_raspi.py
from enum import Enum
import socket
import pickle
import cv2
import time
import numpy as np

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)

import client_handler

class Client_RPI(client_handler.Client):

    def __init__(self, ip, port: int):
        super().__init__(ip, port, controller=Controller_RPI())


class Controller_RPI(client_handler.Controller):

    def __init__(self):
        self.backward_pin = 11 # white_pin
        self.forward_pin = 12 # purple_pin
        self.right_pin = 13 # green_pin
        self.left_pin = 15 # blue_pin
        self.action_and_color_pwm = {self.backward_pin: "backward, white", self.forward_pin: "forward, purple", self.right_pin: "right, green", self.left_pin: "left, blue"}

        self.wait_seconds = 2
        self.pwm_frequency = 100
        self.pwm_duty_cycle = 19

        self.pwms = {}

        self.pwm_setup()

    def pwm_setup(self):
        print('setting GPIOs.')
        for num in self.action_and_color_pwm:
            GPIO.setup(num, GPIO.OUT)
            pwm = GPIO.PWM(num, self.pwm_frequency)
            self.pwms[num] = pwm

    def move_forward(self):
        self.pwms[self.forward_pin].start(self.pwm_duty_cycle)
    def stop_forward(self):
        self.pwms[self.forward_pin].stop()

    def move_left(self):
        self.pwms[self.left_pin].start(self.pwm_duty_cycle)
    def stop_left(self):
        self.pwms[self.left_pin].stop()

    def move_right(self):
        self.pwms[self.right_pin].start(self.pwm_duty_cycle)
    def stop_right(self):
        self.pwms[self.right_pin].stop()

    def move_backward(self):
        self.pwms[self.backward_pin].start(self.pwm_duty_cycle)
    def stop_backward(self):
        self.pwms[self.backward_pin].stop()

    def wait(self):
        time.sleep(self.wait_seconds)

    def clean_exit(self):
        for pwm_num in self.pwms:
            self.pwms[pwm_num].stop()
            GPIO.cleanup()
            print("clean exit.")

    def move_car(self, previous_x, previous_y, current_x, current_y):
        # first measurment
        if previous_x==0 and previous_y==0:
            print("first measurment: doing nothing")
            pass

        # stop (do nothing)
        if previous_x==current_x and previous_y==current_y:
            print("target isn't moving: doing nothing.")
            pass

        # Forward
        elif previous_x==current_x and previous_y>current_y:
            print("target moved backward: moving forward.")
            self.move_forward()
            self.wait()
            self.stop_forward()

            # forward_left
        elif previous_x>current_x and previous_y>current_y:
            print('target moved left and backward: moving left and forward.')
            self.move_left()
            self.move_forward()
            self.wait()
            self.stop_forward()
            self.stop_left()

            # forward_right
        elif previous_x<current_x and previous_y>current_y:
            print('target moved right and backward: moving right and forward.')
            self.move_right()
            self.move_forward()
            self.wait()
            self.stop_forward()
            self.stop_right()

            # backward
        elif previous_x==current_x and previous_y<current_y:
            print('target moved forward: moving backward.')
            self.move_backward()
            self.wait()
            self.stop_backward()

            # backward_left
        elif previous_x<current_x and previous_y<current_y:
            print('target moved right and forward: moving left and backward.')
            self.move_left()
            self.move_backward()
            self.wait()
            self.stop_left()
            self.stop_backward()

            # backward_right
        elif previous_x>current_x and previous_y<current_y:
            print('target moved left and forward: moving right and backward.')
            self.move_right()
            self.move_backward()
            self.wait()
            self.stop_right()
            self.stop_backward()


if __name__ == '__main__':
    ip = '192.168.1.106'
    port = 8003
    client = Client_RPI(ip, port)
    client.start()
