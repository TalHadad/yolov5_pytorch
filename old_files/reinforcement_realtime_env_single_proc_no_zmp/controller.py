# controller.py
import time
from abc import ABC, abstractmethod
import logging
import cv2

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

class ControllerSimple(Controller):
    def do_action(self, action):
        pass
    def exit_clean(self):
        pass
