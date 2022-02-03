# detector.py
import cv2
import logging
from utils_2.logging_level import LOGGING_LEVEL

logging.basicConfig(level=LOGGING_LEVEL)
import time
from abc import ABC, abstractmethod
import multiprocessing
import zmq
import traceback
import numpy as np
import torch

from utils_2.comunication import receive, send


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
        logging.info(f"detector binding to detector queue {self._conf['Detector']['ip']}:{self._conf['Detector']['port']}")
        self._detector_context = zmq.Context()
        self._detector_socket = self._detector_context.socket(zmq.SUB)
        self._detector_socket.connect(f"tcp://{self._conf['Detector']['ip']}:{self._conf['Detector']['port']}")
        # client render
        logging.info(f"detector connecting to render queue {self._conf['Render']['ip']}:{self._conf['Render']['port']}")
        self._render_context = zmq.Context()
        self._render_socket = self._render_context.socket(zmq.PUB)
        self._render_socket.bind(f"tcp://{self._conf['Render']['ip']}:{self._conf['Render']['port']}")

        # client agent
        logging.info(f"detector connecting to detector queue {self._conf['Agent']['ip']}:{self._conf['Agent']['port']}")
        self._agent_context = zmq.Context()
        self._agent_socket = self._agent_context.socket(zmq.PUB)
        self._agent_socket.bind(f"tcp://{self._conf['Agent']['ip']}:{self._conf['Agent']['port']}")

        try:
            while True:
                logging.info(f'detector getting image')
                image = receive(self._detector_socket)

                logging.info(f'detector extracting location')
                location, results, image = self.get_location(image)

                logging.info(f'detector sending location to agent')
                send(self._agent_socket, location)

                logging.info(f'detector sending results and image to render')
                send(self._render_socket, (results, image))

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

    @abstractmethod
    def get_labels(self):
        pass


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

    def exit_clean(self):
        super().exit_clean()

    def get_labels(self):
        return self._model.names

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


class Render(multiprocessing.Process):
    def __init__(self, conf: dict, classes):
        super(Render, self).__init__()
        self._conf = conf
        self._classes = classes

    def run(self):
        # server render
        logging.info(f"render binding to render queue {self._conf['Render']['ip']}:{self._conf['Render']['port']}")
        self._render_context = zmq.Context()
        self._render_socket = self._render_context.socket(zmq.SUB)
        self._render_socket.connect(f"tcp://{self._conf['Render']['ip']}:{self._conf['Render']['port']}")

        try:
            while True:
                logging.info(f'render getting results and image')
                result, image = receive(self._render_socket)

                logging.info(f'render rendeting labeled image')
                self.render(result, image)

        except Exception as e:
            logging.warning(f'render exitting clean, exception {e}')
            traceback.print_exception(type(e), e, e.__traceback__)
            self.exit_clean()

    def exit_clean(self) -> None:
        self._render_context.destroy()
        self.terminate()

    def render(self, results, image) -> None:
        labeled_img = self._get_labeled_image(results, image)
        cv2.imshow('result', np.asarray(labeled_img, dtype=np.uint8))
        # if cv2.waitKey(1) == ord('q'):
        #    cv2.destroyAllWindows()
        cv2.waitKey(1)

    def _get_labeled_image(self, results, image):
        labels, cord = self._get_labels_and_cords(results)
        if len(cord) == 0:
            return image
        labeled_img = self._plot_boxes(labels, cord, image)
        return labeled_img

    """
    The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
    """

    def _get_labels_and_cords(self, results):
        # frame = [torch.tensor(frame)]
        # results = model(frame)
        # labels = results.xyxyn[0][:, -1].numpy() # all rows last column
        # cord = results.xyxyn[0][:, :-1].numpy() # all rows all columns except last
        labels = results.xyxy[0][:, -1].numpy()  # all rows last column
        cord = results.xyxy[0][:, :-1].numpy()  # all rows all columns except last
        return labels, cord

    """
    The function below takes the results and the frame as input and plots boxes over all the objects which have a score higer than our threshold.
    """

    def _plot_boxes(self, labels, cord, image):
        n = len(labels)
        # x_shape, y_shape = image.shape[1], image.shape[0]
        for i in range(n):
            row = cord[i]
            # If score is less than 0.2 we avoid making a prediction.
            if row[4] < 0.2:
                continue
            # x1 = int(row[0]*x_shape)
            # y1 = int(row[1]*y_shape)
            # x2 = int(row[2]*x_shape)
            # y2 = int(row[3]*y_shape)
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            bgr = (0, 255, 0)  # color of the box
            label_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for the label.
            cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 2)  # Plot the boxes
            cv2.putText(image, self._classes[int(labels[i])], (x1, y1), label_font, 0.9, bgr, 2)  # Put a label over box.
            return image
