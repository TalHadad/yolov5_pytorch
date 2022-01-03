# cam_video.py
import cv2
import time
import numpy as np
import torch
import threading

torch.device('cpu')

# global variable to exit the run by pressing some user specific keystroke
exit = 0

def analyze_cv_single_picture():
    global exit

    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        cap = cv2.VideoCapture(0)
        i = 0
        target_x = 0
        target_y = 0
        print(f'target is : {target}')
        while not exit and i<4:
            i += 1

            start = time.time()

            ret, frame = cap.read()
            if not ret:
                print('Could not read frame.')
                break

            # imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

            # Inference
            results = model(frame)

            # Results
            #results.print()
            #results.save()  # or .show()
            #results.show()
            #results.xyxy[0]
            # cv2.imshow(win_name, np.asarray(results.imgs[0], dtype=np.uint8))
            print(f'results.pandas().xyxy[0]:\n{results.pandas().xyxy[0]}')  # img1 predictions (pandas)
            #      xmin    ymin    xmax   ymax  confidence  class    name
            # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
            # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
            # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
            # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

            results_df = results.pandas().xyxy[0]
            target_df = results_df[results_df['name'].isin([target])] # filter on spesific values
            if target_df.size > 0:
                print(f'target_df.size is {target_df.size}')
                x = target_df["xmin"].values[0]
                y = target_df["ymin"].values[0]
                print(f'found {target} in x={x} and y={y}')
                if target_x!=0 and target_y!=0:
                    if target_x>x:
                        print('target moved left')
                    else:
                        print('target moved right')
                target_x = x
                target_y = y

            stop = time.time()
            seconds = stop - start
            print(f'Time taken : {seconds} seconds')

            # Calcutate frames per seconds
            fps = 1 / seconds
            print(f'Estimated frames per second : {fps}')

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print('Quitting.')
                break

    finally:
        cap.release()
        #cv2.destroyAllWindows()
        print('Detections have been performed successfully.')

def get_user_input():
    global exit
    keystrk = input('Press a key to exit.\n')
    # thread dosen't continue until key is pressed
    print('done')
    exit = 1

if __name__ == '__main__':
    target = 'person'
    analyze_cv_single_picture()
    #analyze = threading.Thread(target=analyze_cv_single_picture)
    #user_input = threading.Thread(target=get_user_input)
    #analyze.start()
    #user_input.start()
    #analyze.join()
    #user_input.join()
