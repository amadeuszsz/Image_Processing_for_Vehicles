import numpy as np
import cv2

import sign

class TrafficSignRecognition():
    def __init__(self, frame):
        self.frame = frame
        self.height, self.width, self.channels = frame.shape
        self.objects_coords = []

    def frame_preprocessing(self):
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        #define range of blue color in HSV
        lower_blue = np.array([165, 120, 70])
        upper_blue = np.array([195, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(self.hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        self.after_mask = cv2.bitwise_and(self.frame, self.frame, mask=mask)

    def connected_components(self, offset=5, min_object_size=100):
        self.frame_preprocessing()
        label = 1
        #Coordinates (indices [x, y]) of pixels with R channel (BGR code) greater than 40
        coords = np.argwhere(self.after_mask[:, :, 2] > 40)
        labels = np.zeros(shape=(self.height, self.width), dtype=int)

        #Assigning initial labels for pixels with specific coordinates
        for coord in coords:
            labels[coord[0], coord[1]] = label
            label+=1

        #Connecting pixels (8 connectivity)
        while True:
            try:
                break_flag = 1
                for coord in coords:
                    #(3+offset)x(3+offset) matrix with center of coord variable
                    neighbouring_pixels = labels[coord[0] - 1 - offset:coord[0] + 2 + offset, coord[1] - 1 - offset:coord[1] + 2 + offset]
                    #Minimum value of label excluding 0
                    min_label = np.min(neighbouring_pixels[np.nonzero(neighbouring_pixels)])
                    #If any connection then keep loop
                    if (labels[coord[0], coord[1]] > min_label):
                        labels[coord[0], coord[1]] = min_label
                        break_flag = 0
                if break_flag:
                    break
            except Exception as ex:
                print(ex)

        #List of objects (unique label on frame)
        objects = np.unique(labels)
        objects = np.delete(objects, np.where(objects == 0))

        #Rejecting small objects
        for object in objects:
            if np.count_nonzero(labels == object) < min_object_size:
                labels[labels == object] = 0
                objects = np.delete(objects, np.where(objects == object))

        #Getting coords of objects
        for object in objects:
            most_left = self.width
            most_right = 0
            most_top = self.height
            most_bottom = 0
            for coord in coords:
                if labels[coord[0], coord[1]] == object:
                    if(coord[1] < most_left): most_left = coord[1]
                    if(coord[1] > most_right): most_right = coord[1]
                    if(coord[0] < most_top): most_top = coord[0]
                    if(coord[0] > most_bottom): most_bottom = coord[0]
            self.objects_coords.append([(most_left, most_top), (most_right, most_bottom)])

        cv2.rectangle(self.frame, self.objects_coords[0][0], self.objects_coords[0][1], (0, 255, 0), 2)

        # for coord in coords:
        #     if labels[coord[0], coord[1]] > 0:
        #         self.after_mask[coord[0], coord[1]] = [255, 0, 0]

        return self.frame