# detector.py
import logging
from utils_2.logging_level import LOGGING_LEVEL, MAX_ITER
import time
from abc import ABC, abstractmethod
import multiprocessing
import zmq
import traceback
import torch
from utils_2.comunication import receive, send, RenderMsg
from utils_2.config_parser import ConfigReader

logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('detector')
log.setLevel(LOGGING_LEVEL)

class Detector(multiprocessing.Process, ABC):
    def __init__(self, conf: dict, target: str):
        super(Detector, self).__init__()
        self._conf = conf
        self._target = target
        logging.info(f'target is : {self._target}.')

        logging.info(f'building model.')
        self._model = self._get_model()

    def run(self):
        # server detector
        logging.info(f"detector REP binding to detector queue {self._conf['Detector']['ip']}:{self._conf['Detector']['port']}")
        self._detector_context = zmq.Context()
        self._detector_socket = self._detector_context.socket(zmq.REP)
        self._detector_socket.bind(f"tcp://{self._conf['Detector']['ip']}:{self._conf['Detector']['port']}")

        # client render
        logging.info(f"detector PUB connecting to render queue {self._conf['Render']['ip']}:{self._conf['Render']['port']}")
        self._render_context = zmq.Context()
        self._render_socket = self._render_context.socket(zmq.PUB)
        self._render_socket.connect(f"tcp://{self._conf['Render']['ip']}:{self._conf['Render']['port']}")

        # client agent
        logging.info(f"detector PUB connecting to detector queue {self._conf['Agent']['ip']}:{self._conf['Agent']['port']}")
        self._agent_context = zmq.Context()
        self._agent_socket = self._agent_context.socket(zmq.PUB)
        self._agent_socket.connect(f"tcp://{self._conf['Agent']['ip']}:{self._conf['Agent']['port']}")

        try:
            iter = 0
            while iter < MAX_ITER:
                iter += 1
                logging.info(f'detector getting image')
                image = receive(self._detector_socket)

                logging.info(f'detector extracting location')
                location, results, image = self.get_location(image)

                logging.info(f'detector sending location {location} to agent')
                send(self._agent_socket, location)

                logging.info(f'detector sending results and image to render')
                send(self._render_socket, results)
                send(self._render_socket, image)

                logging.info(f'detector sending ack to camera')
                send(self._detector_socket, 'ack')

        except Exception as e:
            logging.warning(f'detector exitting clean, exception {e}')
            traceback.print_exception(type(e), e, e.__traceback__)
            self.exit_clean()

    @abstractmethod
    def _get_model(self):
        pass

    @abstractmethod
    def get_location(self, image) -> tuple:
        pass

    def exit_clean(self):
        self._detector_context.destroy()
        self._render_context.destroy()
        self._agent_context.destroy()
        self.terminate()


class Detector_Yolov5(Detector):
    def __init__(self, conf: dict, target: str, confidence_threshold: float = 0.6):
        super(Detector_Yolov5, self).__init__(conf=conf, target=target)
        # model label/class is:
        #     0 = person
        #     15 = cat
        #     16 = dog
        label_num_map = {'person': 0, 'cat': 15, 'dog': 16}
        self.target_num = label_num_map[self._target]
        self.confidence_threshold = confidence_threshold
        self._model = self._get_model()

    def exit_clean(self):
        super().exit_clean()

    def _get_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        return model

    def get_location(self, image) -> list:
        start = time.time()

        self.image = image
        self.results = self._model(image)
        self.mid_cords = self._get_target_mid_cords(self.results)
        norm_mid_cords = self._norm_to_nxn_grid(10, self.mid_cords, self.image)

        self._print_time(start)
        return norm_mid_cords, self.results, self.image

    def _norm_to_nxn_grid(self, n: int, cords: list, image) -> list:
        # TODO: chech if the image shape is ok, 1 is x and 0 is y (and not the other way around)
        x_shape, y_shape = image.shape[1], image.shape[0]
        x_norm = (cords[0] / x_shape) * n
        y_norm = (cords[1] / y_shape) * n
        return [x_norm, y_norm]

    def _get_target_mid_cords(self, results) -> list:
        cords = results.xyxy[0].numpy()

        # select target records
        target_cords = cords[cords[:, 5] == self.target_num]
        # target_cords = target_cords[target_cords[:, 4]>self.confidence_threshold]
        if len(target_cords) == 0:
            logging.debug(f'did not found {self._target}')
            # TODO not to return None (action will be nan)
            # TODO cont. realtime venv/gym return up to -1 and 11 with done is True
            return [0, 0]
        # select above confidence threshold

        # take x, y of the first match/row
        x1, y1, x2, y2 = target_cords[0][0], target_cords[0][1], target_cords[0][2], target_cords[0][3]
        x_mid = (abs(x2 - x1) / 2) + x1
        y_mid = (abs(y2 - y1) / 2) + y1
        logging.debug(f'found {self._target} in x={x_mid} and y={y_mid}')
        return [x_mid, y_mid]

    """
    The Function below oracestrates the entire operation and performs the real-time parsing for video stream.
    """

    def _print_time(self, start):
        stop = time.time()
        seconds = stop - start
        logging.debug(f'Time taken : {seconds} seconds')
        # Calcutate frames per seconds
        fps = 1 / seconds
        logging.debug(f'Estimated frames per second : {fps}')

def main():
    conf = ConfigReader().get_params()
    target = 'person'
    detector = Detector_Yolov5(conf=conf, target=target)
    detector.run()
