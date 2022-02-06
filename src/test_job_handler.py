# test_job_handler.py
import logging
import time

from client.controller import JobHandler
from client.controller_raspi import ControllerRPi
from utils_2.config_parser import ConfigReader

def test_job_handler():
    conf = ConfigReader().get_params()
    controller_raspi = ControllerRPi(conf=conf)
    action = 1
    job_handler = JobHandler(controller_raspi, action)
    logging.info(f'controller doning action: {action}')
    #job_handler = JobHandler(action)
    job_handler.start()
    time.sleep(5)
    job_handler.join()

if __name__ == "__main__":
    test_job_handler()
