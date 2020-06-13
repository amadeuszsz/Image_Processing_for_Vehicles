import numpy as np
import cv2

class TrafficSignRecognition():
    def __init__(self, frame):
        self.frame = frame

    def frame_preprocessing(self):
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        self.lower_blue = np.array([150, 0, 0])
        self.upper_blue = np.array([255, 255, 255])
        # Threshold the HSV image to get only blue colors
        self.mask = cv2.inRange(self.hsv, self.lower_blue, self.upper_blue)
        # Bitwise-AND mask and original image
        self.res = cv2.bitwise_and(self.frame, self.frame, mask=self.mask)

        return self.hsv