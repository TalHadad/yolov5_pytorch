# camera.py
import multiprocessing
import traceback
import logging
import zmq
import cv2
from utils_2.comunication import send, receive
from utils_2.config_parser import ConfigReader
from utils_2.logging_level import LOGGING_LEVEL, MAX_ITER

logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('camera')
log.setLevel(LOGGING_LEVEL)

class Camera(multiprocessing.Process):
    def __init__(self, conf: dict):
        super(Camera, self).__init__()
        self._conf = conf
        self.seconds = 10

    def run(self):
        # client camera
        logging.info(f"camera REQ connection to detector queue {self._conf['Detector']['ip']}:{self._conf['Detector']['port']}")
        self._detector_context = zmq.Context()
        self._detector_socket = self._detector_context.socket(zmq.REQ)
        self._detector_socket.connect(f"tcp://{self._conf['Detector']['ip']}:{self._conf['Detector']['port']}")

        try:
            iter = 0
            while iter < MAX_ITER:
                iter += 1
                logging.info(f'camera taking image')
                image = self.get_image()

                logging.info(f'camera sending image')
                send(self._detector_socket, image)

                logging.info(f'camera waiting for detector ack')
                ack = receive(self._detector_socket)
                logging.info(f'camera got {ack}')


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

def main():
    conf = ConfigReader().get_params()
    camera = Camera(conf=conf)
    camera.run()
