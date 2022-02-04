# controller.py
from abc import ABC, abstractmethod
import multiprocessing
import traceback
import logging
import zmq
from utils_2.comunication import receive
from utils_2.config_parser import ConfigReader
from utils_2.logging_level import LOGGING_LEVEL, MAX_ITER

logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('controller')
log.setLevel(LOGGING_LEVEL)

class Controller(ABC, multiprocessing.Process):
    def __init__(self, conf: dict):
        super(Controller, self).__init__()
        self._conf = conf

    def run(self):
        # server controller
        logging.info(f"controller SUB binding to controller queue {self._conf['Controller']['ip']}:{self._conf['Controller']['port']}")
        self._controller_context = zmq.Context()
        self._controller_socket = self._controller_context.socket(zmq.SUB)
        self._controller_socket.connect(f"tcp://{self._conf['Controller']['ip']}:{self._conf['Controller']['port']}")
        self._controller_socket.subscribe("")

        try:
            iter = 0
            while iter < MAX_ITER:
                iter += 1
                logging.info(f'controller getting action')
                action = int(self._controller_socket.recv(copy=False, flags=0))

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


def main():
    conf = ConfigReader().get_params()
    controller = ControllerSimple(conf=conf)
    controller.run()
