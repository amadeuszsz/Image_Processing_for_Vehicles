import numpy as np
import cv2

import sign

class TrafficSignRecognition():
    signs = [] # of type Sign

    def __init__(self, frame):
        self.frame = frame

    def frame_preprocessing(self):
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        #define range of blue color in HSV
        lower_blue = np.array([165, 120, 70])
        upper_blue = np.array([195, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(self.hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        self.after_mask = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        return self.after_mask

    def connected_components(self):
        pass