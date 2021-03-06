# client.py
import logging # debug, info, warning, error, critical
from utils_2.logging_level import LOGGING_LEVEL
logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('client')
log.setLevel(LOGGING_LEVEL)

from controller import ControllerSimple
from camera import Camera
#from .controller_raspi import ControllerRPi
from utils_2.config_parser import ConfigReader

#if __name__ == '__main__':
def main():
    conf = ConfigReader().get_params()

    camera = Camera(conf=conf)

    controller = ControllerSimple(conf=conf)
    #controller = ControllerRPi(conf)

    camera.start()
    controller.start()
