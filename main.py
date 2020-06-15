import numpy as np
import cv2
import time
import pyopencl as cl
import pyopencl.cltypes
import time

from lane_detection import LaneDetection
from traffic_sign_recognition import TrafficSignRecognition
from GPUSetup import GPUSetup

#Video variables
cap = cv2.VideoCapture('videos/video5.mp4')
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


def main():
    signRecognizer = TrafficSignRecognition()
    name = "Traffic Sign Recognition"
    window = cv2.namedWindow(name)
    cv2.moveWindow(name, 150, 150)

    while (cap.isOpened()):
        time_start = time.time()
        ret, frame = cap.read()

        signRecognizer.load_new_frame(frame)
        # final_frame = signRecognizer.frame_preprocessing()
        final_frame = signRecognizer.connected_components()

        # *-----------------------------DEBUG---------------------------------
        # img_concate_Verti = np.concatenate((signRecognizer.frame, signRecognizer.hsv, signRecognizer.after_mask), axis=0)
        # img_concate_Verti = cv2.resize(img_concate_Verti, (960, 640))
        # cv2.imshow('concatenated_Verti', img_concate_Verti)
        # *-------------------------------------------------------------------
        final_frame = cv2.resize(final_frame, (960, 540))  
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


if __name__ == "__main__":
    main()