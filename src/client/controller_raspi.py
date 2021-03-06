# controller_raspi.py
import traceback
import zmq
from utils_2.comunication import receive
from utils_2.config_parser import ConfigReader
import logging
import time
from utils_2.logging_level import LOGGING_LEVEL, MAX_ITER
import RPi.GPIO as GPIO
from .controller import Controller

GPIO.setmode(GPIO.BOARD)

logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('controller_raspi')
log.setLevel(LOGGING_LEVEL)

class ControllerRPi(Controller):
    '''
    Raspberry pi controller of pwms and camera
    '''

    def __init__(self, conf):
        super(ControllerRPi, self).__init__(conf=conf)
        self.pins = {'forward': 12,#11, # 11 = backward (white)
                     'backward': 11,#12, # 12 = forward (purple)
                     'right': 15,#13, # 13 = right (green)
                     'left': 13}#15} # 15 = left (blue)
        self.wait_seconds = 2
        self.pwm_frequency = 100
        self.pwm_duty_cycle = 100
        self.pwms = self._pwm_setup(self.pins)

    def _pwm_setup(self, pins: dict) -> list:
        logging.info('setting GPIOs.')
        pwms = {}
        for pin in pins:
            GPIO.setup(pins[pin], GPIO.OUT)
            pwms[pin] = GPIO.PWM(pins[pin], self.pwm_frequency)
            #pwms[pin].start(0)
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
        logging.info(f'controller raspi doing action {action}')
        # stop (do nothing) when target not moving or not found
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
        super().exit_clean()
        for pin in self.pwms:
            self.pwms[pin].stop()
        GPIO.cleanup()

    def _wait(self) -> None:
        time.sleep(self.wait_seconds)

    def _move_forward(self) -> None:
        #self.pwms['forward'].ChangeDutyCycle(self.pwm_duty_cycle)
        self.pwms['forward'].start(self.pwm_duty_cycle)
    def _stop_forward(self) -> None:
        self.pwms['forward'].ChangeDutyCycle(0)
        self.pwms['forward'].stop()

    def _move_left(self) -> None:
        self.pwms['left'].ChangeDutyCycle(self.pwm_duty_cycle)
    def _stop_left(self) -> None:
        self.pwms['left'].ChangeDutyCycle(0)

    def _move_right(self) -> None:
        self.pwms['right'].ChangeDutyCycle(self.pwm_duty_cycle)
    def _stop_right(self) -> None:
        self.pwms['right'].ChangeDutyCycle(0)

    def _move_backward(self) -> None:
        self.pwms['backward'].ChangeDutyCycle(self.pwm_duty_cycle)
    def _stop_backward(self) -> None:
        self.pwms['backward'].ChangeDutyCycle(0)


def main():
    conf = ConfigReader().get_params()
    controller = ControllerRPi(conf=conf)
    controller.run()

if __name__ == '__main__':
    conf = ConfigReader().get_params()
    controller = ControllerRPi(conf=conf)
    test_do_action_interactive_program(controller)

def test_do_action_interactive_program(controller: Controller):
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
    action_text_map = {
        0 : 'stay'
        1 : 'forward'
        2 : 'forward left'
        3 : 'forward right'
        4 : 'backward'
        5 : 'backward left'
        6 : 'backward right'
    }
    print(f'Enter action number to do (0-6 or -1 to exit): ')
    action = int(input())
    try:
        while action != -1:
            print(f'action {action} ({action_text_map[action]}) is been done.')
            controller.do_action(action)
            print(f'action {action} ({action_text_map[action]}) has ended.')
            print(f'Enter action number to do (0-6 or -1 to exit): ')
            pin = int(input())
    except:
        print('exiting.')
        controller.exit_clean()
    finally:
        print('exiting.')
        controller.exit_clean()
