# controller.py
import time
from abc import ABC, abstractmethod
import logging
import cv2
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)

class Controller(ABC):

    @abstractmethod
    def do_action(self, action: int) -> None:
        pass

    def get_image(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            logging.error('Could not read frame.')
        cap.release()
        return frame

    @abstractmethod
    def exit_clean(self):
        pass

class Controller_RPi(Controller):
    '''
    Raspberry pi controller of pwms and camera
    '''

    def __init__(self):
        self.pins = {'forward': 11, # 11 = backward (white)
                     'backward': 12, # 12 = forward (purple)
                     'right': 13, # 13 = right (green)
                     'left': 15} # 15 = left (blue)
        self.wait_seconds = 2
        self.pwm_frequency = 100
        self.pwm_duty_cycle = 100
        self.pwms =  self._pwm_setup(self.pins)

    def _pwm_setup(self, pins: dict) -> list:
        logging.info('setting GPIOs.')
        pwms = []
        for pin in pins:
            GPIO.setup(pins[pin], GPIO.OUT)
            pwms.append(GPIO.PWM(pins[pin], self.pwm_frequency))
        return pwms

    def do_action(self, action: int) -> None:
        '''
        Move the car acording to action number:
           0 = stay
           1 = forward
           2 = forward left
           3 = forward right
           4 = backward
           5 = backward left
           6 = backward right
        '''
        # stop (do nothing)
        if action == 0:
            logging.debug("target isn't moving: doing nothing.")
            pass

        # forward
        elif action == 1:
            logging.debug("target moved backward: moving forward.")
            self._move_forward()
            self._wait()
            self._stop_forward()

        # forward_left
        elif action == 2:
            logging.debug('target moved left and backward: moving left and forward.')
            self._move_left()
            self._move_forward()
            self._wait()
            self._stop_forward()
            self._stop_left()

        # forward_right
        elif action == 3:
            logging.debug('target moved right and backward: moving right and forward.')
            self._move_right()
            self._move_forward()
            self._wait()
            self._stop_forward()
            self._stop_right()

        # backward
        elif action == 4:
            logging.debug('target moved forward: moving backward.')
            self._move_backward()
            self._wait()
            self._stop_backward()

        # backward_left
        elif action == 5:
            logging.debug('target moved right and forward: moving left and backward.')
            self._move_left()
            self._move_backward()
            self._wait()
            self._stop_left()
            self._stop_backward()

        # backward_right
        elif action == 6:
            logging.debug('target moved left and forward: moving right and backward.')
            self._move_right()
            self._move_backward()
            self._wait()
            self._stop_right()
            self._stop_backward()

    def exit_clean(self) -> None:
        logging.info(f'exiting clean.')
        for pwm in self.pwms:
            pwm.stop()
        GPIO.cleanup()

    def _wait(self) -> None:
        time.sleep(self.wait_seconds)

    def _move_forward(self) -> None:
        self.pwms[self.pins['forward']].start(self.pwm_duty_cycle)
    def _stop_forward(self) -> None:
        self.pwms[self.pins['forward']].stop()

    def _move_left(self) -> None:
        self.pwms[self.pins['left']].start(self.pwm_duty_cycle)
    def _stop_left(self) -> None:
        self.pwms[self.pins['left']].stop()

    def _move_right(self) -> None:
        self.pwms[self.pins['right']].start(self.pwm_duty_cycle)
    def _stop_right(self) -> None:
        self.pwms[self.pins['right']].stop()

    def _move_backward(self) -> None:
        self.pwms[self.pins['backward']].start(self.pwm_duty_cycle)
    def _stop_backward(self) -> None:
        self.pwms[self.pins['backward']].stop()
