# controller.py
import time
from abc import ABC, abstractmethod
import multiprocessing
import traceback
import logging
import zmq
import cv2
from utils_2.comunication import receive, send


class Camera(multiprocessing.Process):
    def __init__(self, conf: dict):
        super(Camera, self).__init__()
        self._conf = conf
        self.seconds = 2

    def run(self):
        # client camera
        logging.info(f"camera connection to detector queue {self._conf['Detector']['ip']}:{self._conf['Detector']['port']}")
        self._detector_context = zmq.Context()
        self._detector_socket = self._detector_context.socket(zmq.PUSH)
        self._detector_socket.connect(f"tcp://{self._conf['Detector']['ip']}:{self._conf['Detector']['port']}")

        try:
            while True:
                logging.info(f'camera taking image')
                image = self.get_image()

                logging.info(f'camera sending image')
                send(self._detector_socket, image)

                logging.info(f'camera waiting {self.seconds} seconds')
                time.sleep(self.seconds)

        except Exception as e:
            logging.warning(f'camera exitting clean, exception {e}')
            traceback.print_exception(type(e), e, e.__traceback__)
            self.exit_clean()

    def exit_clean(self):
        self._detector_context.destroy()
        self.terminate()

    def get_image(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            logging.error('Could not read frame.')
        cap.release()
        return frame


class Controller(ABC, multiprocessing.Process):
    def __init__(self, conf: dict):
        super(Controller, self).__init__()
        self._conf = conf

    def run(self):
        # server controller
        logging.info(f"controller binding to controller queue {self._conf['Controller']['ip']}:{self._conf['Controller']['port']}")
        self._controller_context = zmq.Context()
        self._controller_socket = self._controller_context.socket(zmq.PULL)
        self._controller_socket.connect(f"tcp://{self._conf['Controller']['ip']}:{self._conf['Controller']['port']}")

        try:
            while True:
                logging.info(f'controller getting action')
                action = int(receive(self._controller_socket))

                logging.info(f'controller doning action')
                self.do_action(action)

        except Exception as e:
            logging.warning(f'controller exitting clean, exception {e}')
            traceback.print_exception(type(e), e, e.__traceback__)
            self.exit_clean()

    @abstractmethod
    def do_action(self, action: int) -> None:
        pass

    def exit_clean(self):
        self._controller_context.destroy()
        self.terminate()


class ControllerSimple(Controller):
    def do_action(self, action):
        pass
