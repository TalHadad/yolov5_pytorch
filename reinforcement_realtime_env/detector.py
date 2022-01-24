# server_pc.py
from enum import Enum
import cv2
import socket
import time
import pickle
import torch
import numpy as np

from server_handler import Server, Analyzer

class Server_PC(Server):
    def __init__(self, ip, port: int, target):
        super().__init__(ip, port, analyzer=Analyzer_PC())

class Analyzer_PC(Analyzer):
    def get_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        return model

    def get_labeled_image(self, frame):

        self.results = self.model(frame)
        labels, cord = self.score_frame(self.results)
        labeled_img = self.plot_boxes(labels, cord, frame)
        return labeled_img


    def get_target_coords(self):
        x, y = 0, 0

        results_df = self.results.pandas().xyxy[0]
        print(f'results_df:\n{results_df}')  # img1 predictions (pandas)

        target_df = results_df[results_df['name'].isin([target])] # filter on spesific values
        print(f'target_df.size is {target_df.shape[0]}')
        print(f'target_df:\n{target_df}')
        if target_df.shape[0] > 1:
            print(f'Error: found multipule ({target_df.size}) targets')
        elif target_df.shape[0] == 1:
            x = target_df["xmin"].values[0]
            y = target_df["ymin"].values[0]
        return x, y

    """
    The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
    """
    def score_frame(self, results):
        #frame = [torch.tensor(frame)]
        #results = model(frame)
        labels = results.xyxyn[0][:, -1].numpy()
        cord = results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    """
    The function below takes the results and the frame as input and plots boxes over all the objects which have a score higer than our threshold.
    """
    def plot_boxes(self, labels, cord, frame):
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            # If score is less than 0.2 we avoid making a prediction.
            if row[4] < 0.2:
                continue
            x1 = int(row[0]*x_shape)
            y1 = int(row[1]*y_shape)
            x2 = int(row[2]*x_shape)
            y2 = int(row[3]*y_shape)
            bgr = (0, 255, 0) # color of the box
            classes = self.model.names # Get the name of label index
            label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
            cv2.rectangle(frame, \
                          (x1, y1), (x2, y2), \
                          bgr, 2) #Plot the boxes
            cv2.putText(frame,\
                        classes[int(labels[i])], \
                        (x1, y1), \
                        label_font, 0.9, bgr, 2) #Put a label over box.
            return frame

    """
    The Function below oracestrates the entire operation and performs the real-time parsing for video stream.
    """
    def __call__(self):
        player = self.get_video_stream() #Get your video stream.
        assert player.isOpened() # Make sure that their is a stream.
        #Below code creates a new video writer object to write our
        #output stream.
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG") #Using MJPEG codex
        out = cv2.VideoWriter(out_file, four_cc, 20, (x_shape, y_shape))
        ret, frame = player.read() # Read the first frame.
        while ret: # Run until stream is out of frames
            start_time = time() # We would like to measure the FPS.
            results = self.score_frame(frame) # Score the Frame
            frame = self.plot_boxes(results, frame) # Plot the boxes.
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3) #Measure the FPS.
            print(f"Frames Per Second : {fps}")
            out.write(frame) # Write the frame onto the output.
            ret, frame = player.read() # Read next frame.

if __name__ == '__main__':
    #ip = socket.gethostname()
    ip = '192.168.1.106'
    port = 8005
    target = 'person'
    server = Server_PC(ip, port, target)
    server.start()
