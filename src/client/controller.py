# controller.py
import multiprocessing
import traceback
import logging
import zmq
import time
from utils_2.comunication import receive
from utils_2.config_parser import ConfigReader
from utils_2.logging_level import LOGGING_LEVEL, MAX_ITER

logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('controller')
log.setLevel(LOGGING_LEVEL)

class JobHandler(multiprocessing.Process):
    def __init__(self, controller, action=0):
        super(JobHandler, self).__init__()
        self.controller = controller
        self.action = action

    def run(self) -> None:
        self.controller.do_action(self.action)



class Controller(multiprocessing.Process):
    def __init__(self, conf: dict):
        super(Controller, self).__init__()
        self._conf = conf
        self.job_handler = JobHandler(self)

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
                action = int.from_bytes(self._controller_socket.recv(copy=False, flags=0), 'big')

                # The old code. I'm trying to do do_action in another process so the movement
                # will be more fulid. And check the boundaries of PWM access in raspi (no need for sudo and now check if different process can access it)
                #logging.info(f'controller doning action {action}')
                #self.do_action(action)

                # 1. testing multiprocessing package
                #p = multiprocessing.Process(target=self.do_action, args=(action,))
                #p.start()

                # 2. testing JobHandler(muliprocessing.Process)
                #if not self.job_handler.is_alive():
                #   logging.info(f'controller doning action {iter}: {action}')
                #   self.job_handler = JobHandler(self, action)
                #   self.job_handler.start()
                #else:
                #   logging.info(f'controller is not doning action {iter}: {action}, and throw it away.')

                # 3. testing subprocess (with and without sudo)
                # changing __main__ and call it from test_job_handler (should be called test_do_action)


        except Exception as e:
            logging.warning(f'controller exitting clean, exception {e}')
            traceback.print_exception(type(e), e, e.__traceback__)
            self.exit_clean()

        finally:
            logging.warning(f'controller exitting clean')
            self.exit_clean()


    def do_action(self, action: int) -> None:
        pass

    def exit_clean(self):
        logging.info(f'exiting clean.')
        self._controller_context.destroy()
        #self.terminate()


def main():
    conf = ConfigReader().get_params()
    controller = Controller(conf=conf)
    controller.run()
