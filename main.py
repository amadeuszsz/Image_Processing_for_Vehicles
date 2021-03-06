import numpy as np
import cv2
import time
import pyopencl as cl
import pyopencl.cltypes
import time
import sys
import imutils

from traffic_sign_recognition import TrafficSignRecognition
from GPUSetup import GPUSetup

#Video variables
cap = cv2.VideoCapture('videos/video1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

def sync_fps(time_start):
    '''
    Sleeping function. Wait until reach desired frame rate.
    :param time_start: last frame timestamp
    :return:
    '''
    timeDiff = time.time() - time_start
    if (timeDiff < 1.0 / (fps)):
        time.sleep(1.0 / (fps) - timeDiff)

def recognize_sign(path=None):
    signRecognizer = TrafficSignRecognition()
    name = "Traffic Sign Recognition"
    if path is not None:
        cap = cv2.VideoCapture(path)
    else:
        cap = cv2.VideoCapture('videos/video1.mp4')

    while (cap.isOpened()):
        time_start = time.time()
        ret, frame = cap.read()
        signRecognizer.load_new_frame(frame)
        original_frame, marked_frame = signRecognizer.connected_components()
        final_frame = np.concatenate((original_frame, marked_frame), axis=0)
        final_frame = imutils.resize(final_frame, height=800)  
        cv2.imshow(name, final_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('w'):
            cv2.waitKey(-1)
        # print("Total loop time: ", time.time() - time_start)
        # sync_fps(time_start=time_start)
    cap.release()
    cv2.destroyAllWindows()

def main():
    recognize_sign(sys.argv[1])

if __name__ == "__main__":
    main()