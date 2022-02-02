# client.py
import logging # debug, info, warning, error, critical
from utils_2.logging_level import LOGGING_LEVEL
logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('client')
log.setLevel(LOGGING_LEVEL)

from controller import ControllerSimple, Camera
#from controller import Controller_Raspi
from utils_2.config_parser import ConfigReader

if __name__ == '__main__':
    conf = ConfigReader().get_params()

    camera = Camera(conf=conf)

    controller = ControllerSimple(conf=conf)
    #controller = Controller_Raspi(conf)

    camera.start()
    controller.start()
