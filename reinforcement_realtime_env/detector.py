# detector.py
import cv2
import logging
from reinforcement_realtime_env import LOGGING_LEVEL
logging.basicConfig(level=LOGGING_LEVEL)
import time
from abc import ABC, abstractmethod
import numpy as np
import torch


class Detector(ABC):
    def __init__(self, target: str):
        self.target = target
        logging.info(f'target is : {self.target}.')

        logging.info(f'building model.')
        self.model = self._get_model()

    @abstractmethod
    def _get_model(self):
        pass

    @abstractmethod
    def get_location(self, image) -> tuple:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def exit_clean(self):
        pass

class Detector_Yolov5(Detector):
    def __init__(self, target: str = 'cat', confidence_threshold: float = 0.6):
        super(Detector_Yolov5, self).__init__(target)
        # model label/class is:
        #     0 = person
        #     15 = cat
        #     16 = dog
        label_num_map = {'person': 0, 'cat': 15, 'dog': 16}
        self.target_num = label_num_map[self.target]
        self.confidence_threshold = confidence_threshold
    def exit_clean(self):
        pass
    def _get_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        return model

    def get_location(self, image) -> list:
        start = time.time()

        self.image = image
        self.results = self.model(image)
        self.mid_cords = self._get_target_mid_cords(self.results)
        norm_mid_cords = self._norm_to_nxn_grid(10, self.mid_cords, self.image)

        self._print_time(start)
        return norm_mid_cords

    def _norm_to_nxn_grid(self, n: int, cords: list, image) -> list:
        # TODO: chech if the image shape is ok, 1 is x and 0 is y (and not the other way around)
        x_shape, y_shape = image.shape[1], image.shape[0]
        x_norm = (cords[0]/x_shape) * n
        y_norm = (cords[1]/y_shape) * n
        return [x_norm, y_norm]




    def render(self):
        labeled_img = self._get_labeled_image(self.results, self.image)
        cv2.imshow('result', np.asarray(labeled_img, dtype=np.uint8))
        #if cv2.waitKey(1) == ord('q'):
        #    cv2.destroyAllWindows()
        cv2.waitKey(1)

    def _get_target_mid_cords(self, results) -> list:
        cords = results.xyxy[0].numpy()

        # select target records
        target_cords = cords[cords[:,5]==self.target_num]
        if len(target_cords) == 0:
            logging.debug(f'did not found {self.target}')
            # TODO not to return None (action will be nan)
            # TODO cont. realtime venv/gym return up to -1 and 11 with done is True
            return [0,0]
        # select above confidence threshold
        target_cords = target_cords[target_cords[:, 4]>self.confidence_threshold]
        # take x, y of the first match/row
        x1, y1, x2, y2 = target_cords[0][0], target_cords[0][1], target_cords[0][2], target_cords[0][3]
        x_mid = (abs(x2-x1)/2) + x1
        y_mid = (abs(y2-y1)/2) + y1
        logging.debug(f'found {self.target} in x={x_mid} and y={y_mid}')
        return [x_mid, y_mid]


    def _get_labeled_image(self, results, image):
        labels, cord = self._get_labels_and_cords(results)
        labeled_img = self._plot_boxes(labels, cord, image)
        return labeled_img

    """
    The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
    """
    def _get_labels_and_cords(self, results):
        #frame = [torch.tensor(frame)]
        #results = model(frame)
        #labels = results.xyxyn[0][:, -1].numpy() # all rows last column
        #cord = results.xyxyn[0][:, :-1].numpy() # all rows all columns except last
        labels = results.xyxy[0][:, -1].numpy() # all rows last column
        cord = results.xyxy[0][:, :-1].numpy() # all rows all columns except last
        return labels, cord

    """
    The function below takes the results and the frame as input and plots boxes over all the objects which have a score higer than our threshold.
    """
    def _plot_boxes(self, labels, cord, image):
        n = len(labels)
        #x_shape, y_shape = image.shape[1], image.shape[0]
        for i in range(n):
            row = cord[i]
            # If score is less than 0.2 we avoid making a prediction.
            if row[4] < 0.2:
                continue
            #x1 = int(row[0]*x_shape)
            #y1 = int(row[1]*y_shape)
            #x2 = int(row[2]*x_shape)
            #y2 = int(row[3]*y_shape)
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            bgr = (0, 255, 0) # color of the box
            classes = self.model.names # Get the name of label index
            label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
            cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 2) #Plot the boxes
            cv2.putText(image, classes[int(labels[i])], (x1, y1), label_font, 0.9, bgr, 2) #Put a label over box.
            return image

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
