# render.py
import cv2
import logging
from utils_2.logging_level import LOGGING_LEVEL, MAX_ITER
import multiprocessing
import zmq
import traceback
import numpy as np
from utils_2.comunication import receive
from utils_2.config_parser import ConfigReader

logging.basicConfig(level=LOGGING_LEVEL)
log = logging.getLogger('render')
log.setLevel(LOGGING_LEVEL)

class Render(multiprocessing.Process):
    def __init__(self, conf: dict, classes):
        super(Render, self).__init__()
        self._conf = conf
        self._classes = classes

    def run(self):
        # server render
        logging.info(f"render SUB binding to render queue {self._conf['Render']['ip']}:{self._conf['Render']['port']}")
        self._render_context = zmq.Context()
        self._render_socket = self._render_context.socket(zmq.SUB)
        self._render_socket.bind(f"tcp://{self._conf['Render']['ip']}:{self._conf['Render']['port']}")
        self._render_socket.subscribe("")

        try:
            iter = 0
            while iter < MAX_ITER:
                iter += 1
                logging.info(f'render getting results and image')
                self.render_msg = receive(self._render_socket)
                print(f'{self.render_msg}')
                results, image = self.render_msg.results, self.render_msg.image

                logging.info(f'render rendeting labeled image')
                self.render(results, image)

        except Exception as e:
            logging.warning(f'render exitting clean, exception {e}')
            traceback.print_exception(type(e), e, e.__traceback__)
            self.exit_clean()

        finally:
            logging.warning(f'render exitting clean')
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
            cv2.putText(image, self._classes[int(labels[i])], (x1, y1), label_font, 0.9, bgr,
                        2)  # Put a label over box.
            return image


def main():
    conf = ConfigReader().get_params()
    classes = ['person',
               'bicycle',
               'car',
               'motorcycle',
               'airplane',
               'bus',
               'train',
               'truck',
               'boat',
               'traffic light',
               'fire hydrant',
               'stop sign',
               'parking meter',
               'bench',
               'bird',
               'cat',
               'dog',
               'horse',
               'sheep',
               'cow',
               'elephant',
               'bear',
               'zebra',
               'giraffe',
               'backpack',
               'umbrella',
               'handbag',
               'tie',
               'suitcase',
               'frisbee',
               'skis',
               'snowboard',
               'sports ball',
               'kite',
               'baseball bat',
               'baseball glove',
               'skateboard',
               'surfboard',
               'tennis racket',
               'bottle',
               'wine glass',
               'cup',
               'fork',
               'knife',
               'spoon',
               'bowl',
               'banana',
               'apple',
               'sandwich',
               'orange',
               'broccoli',
               'carrot',
               'hot dog',
               'pizza',
               'donut',
               'cake',
               'chair',
               'couch',
               'potted plant',
               'bed',
               'dining table',
               'toilet',
               'tv',
               'laptop',
               'mouse',
               'remote',
               'keyboard',
               'cell phone',
               'microwave',
               'oven',
               'toaster',
               'sink',
               'refrigerator',
               'book',
               'clock',
               'vase',
               'scissors',
               'teddy bear',
               'hair drier',
               'toothbrush']
    render = Render(conf=conf, classes=classes)
    render.run()
