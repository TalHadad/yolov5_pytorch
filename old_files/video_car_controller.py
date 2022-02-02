# video_car_controller.py

import time
import numpy as np
import torch
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)

class Controller:

    def __init__(self, target):
        self.backward_pin = 11 # white_pin
        self.forward_pin = 12 # purple_pin
        self.right_pin = 13 # green_pin
        self.left_pin = 15 # blue_pin
        self.action_and_color_pwm = {self.backward_pin: "backward, white", self.forward_pin: "forward, purple", self.right_pin: "right, green", self.left_pin: "left, blue"}

        self.wait_seconds = 2
        self.pwm_frequency = 100
        self.pwm_duty_cycle = 19

        self.pwms = {}

        self.target = target

    def pwm_setup(self):
        print('setting GPIOs.')
        for num in self.action_and_color_pwm:
            GPIO.setup(num, GPIO.OUT)
            pwm = GPIO.PWM(num, self.pwm_frequency)
            self.pwms[num] = pwm

    def move_forward(self):
        self.pwms[self.forward_pin].start(self.pwm_duty_cycle)
    def stop_forward(self):
        self.pwms[self.forward_pin].stop()

    def move_left(self):
        self.pwms[self.left_pin].start(self.pwm_duty_cycle)
    def stop_left(self):
        self.pwms[self.left_pin].stop()

    def move_right(self):
        self.pwms[self.right_pin].start(self.pwm_duty_cycle)
    def stop_right(self):
        self.pwms[self.right_pin].stop()

    def move_backward(self):
        self.pwms[self.backward_pin].start(self.pwm_duty_cycle)
    def stop_backward(self):
        self.pwms[self.backward_pin].stop()

    def wait(self):
        time.sleep(self.wait_seconds)

    def move_car(self, previous_x, previous_y, current_x, current_y):


        # stop (do nothing)
        if previous_x==current_x and previous_y==current_y:
            print("target isn't moving: doing nothing.")
            pass

        # Forward
        elif previous_x==current_x and previous_y>current_y:
            print("target moved backward: moving forward.")
            self.move_forward()
            self.wait()
            self.stop_forward()

            # forward_left
        elif previous_x>current_x and previous_y>current_y:
            print('target moved left and backward: moving left and forward.')
            self.move_left()
            self.move_forward()
            self.wait()
            self.stop_forward()
            self.stop_left()

            # forward_right
        elif previous_x<current_x and previous_y>current_y:
            print('target moved right and backward: moving right and forward.')
            self.move_right()
            self.move_forward()
            self.wait()
            self.stop_forward()
            self.stop_right()

            # backward
        elif previous_x==current_x and previous_y<current_y:
            print('target moved forward: moving backward.')
            self.move_backward()
            self.wait()
            self.stop_backward()

            # backward_left
        elif previous_x<current_x and previous_y<current_y:
            print('target moved right and forward: moving left and backward.')
            self.move_left()
            self.move_backward()
            self.wait()
            self.stop_left()
            self.stop_backward()

            # backward_right
        elif previous_x>current_x and previous_y<current_y:
            print('target moved left and forward: moving right and backward.')
            self.move_right()
            self.move_backward()
            self.wait()
            self.stop_right()
            self.stop_backward()

    def print_time(self, start):
        stop = time.time()
        seconds = stop - start
        print(f'Time taken : {seconds} seconds')
        # Calcutate frames per seconds
        fps = 1 / seconds
        print(f'Estimated frames per second : {fps}')

    def detect_objects(self):
        try:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            i = 0
            previous_x = 0
            previous_y = 0
            print(f'target is : {self.target}')

            while True:
                start = time.time()

                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                if not ret:
                    print('Could not read frame.')
                    break
                cap.release()

                results = model(frame)
                # cv2.imshow(win_name, np.asarray(results.imgs[0], dtype=np.uint8))
                results_df = results.pandas().xyxy[0]
                print(f'results_df:\n{results_df}')  # img1 predictions (pandas)

                target_df = results_df[results_df['name'].isin([target])] # filter on spesific values
                print(f'target_df.size is {target_df.size}')
                if target_df.size > 1:
                    print(f'Error: found multipule ({target_df.size}) targets')
                elif target_df.size == 1:
                    current_x = target_df["xmin"].values[0]
                    current_y = target_df["ymin"].values[0]
                    print(f'found {target} in x={current_x} and y={current_y}')
                    if previous_x!=0 and previous_y!=0:
                        self.move_car(previous_x, previous_y, current_x, current_y)
                    previous_x = current_x
                    previous_y = current_y

                self.print_time(start)


        finally:
            print("clean exit.")
            cap.release()
            for pwm_num in self.pwms:
                self.pwms[pwm_num].stop()
            GPIO.cleanup()
            print('Detections have been performed successfully.')

    def start(self):
        self.pwm_setup()
        self.detect_objects()
if __name__ == '__main__':
    target = 'person'
    ctrl = Controller(target)
    ctrl.start()
