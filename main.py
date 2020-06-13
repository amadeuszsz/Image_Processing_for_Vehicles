import numpy as np
import cv2
import time
from lane_detection import LaneDetection
from traffic_sign_recognition import TrafficSignRecognition

cap = cv2.VideoCapture('videos/video2.mp4')
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
    while (cap.isOpened()):
        time_start = time.time()
        ret, frame = cap.read()
        #######################################
        #Do things here
        #x = LaneDetection(frame)
        #name, final_frame = x.lane_detection()
        x = TrafficSignRecognition(frame)
        name = "Traffic Sign Recognition"
        final_frame = x.frame_preprocessing()
        #######################################
        cv2.imshow(name, final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('w'):
            while True:
                if cv2.waitKey(1) & 0xFF == ord('w'):
                    break
        sync_fps(time_start=time_start)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()