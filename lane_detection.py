import numpy as np
import cv2


class LaneDetection():
    def __init__(self, frame):
        self.frame = frame
        self.height, self.width, self.channels = frame.shape

    def frame_preprocessing(self, frame):
        '''
        Frame preprocessing. Selecting ROI (Region of Interest), grayscaling and smoothing
        :param frame: original video frame
        :return: preprocessed frame, ROI's shift
        '''
        self.height_shift = int(self.height / 2)
        self.roi_frame = frame[self.height_shift:self.height, 0:self.width]
        self.gray = cv2.cvtColor(self.roi_frame, cv2.COLOR_BGR2GRAY)
        self.blur = cv2.GaussianBlur(self.gray, (5, 5), 2)
        return self.blur, self.height_shift

    def frame_postprocessing(self, frame, processed_frame, scale=1.0):
        '''
        Merging original and processed frame for better comparison
        :param frame: original video frame
        :param processed_frame: processed frame
        :param scale: frame scale factor
        :return: postprocessed frame
        '''
        self.height_m = int(self.height * scale)
        self.width_m = int(self.width * scale * 2)
        self.blank_frame = np.zeros((int(self.height / 2), self.width, 3), np.uint8)
        self.processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
        self.merged_frames = np.concatenate((self.blank_frame, self.processed_frame_rgb), axis=0)
        self.merged_frames = np.concatenate((frame, self.merged_frames), axis=1)
        self.resized_frame = cv2.resize(self.merged_frames, (self.width_m, self.height_m))
        return self.resized_frame

    def canny_edge_detection(self, frame, sigma=0.33):
        '''
        Canny edge detection with automatic threshold values
        :param frame: preprocessed frame
        :param sigma: threshold width range
        :return: frame's edges
        '''
        self.median = np.median(frame)
        self.thresh_low = int(max(0, (1.0 - sigma) * self.median))
        self.thresh_up = int(min(255, (1.0 + sigma) * self.median))
        self.edges = cv2.Canny(frame, self.thresh_low, self.thresh_up)
        print(self.median, self.thresh_low, self.thresh_up)
        return self.edges

    # TODO
    def lane_detection(self):
        '''
        :param frame:
        :return:
        '''
        self.preprocessed_frame, self.height_shift = self.frame_preprocessing(self.frame)
        self.roi_edges = self.canny_edge_detection(self.preprocessed_frame)
        self.lines = cv2.HoughLinesP(self.roi_edges, 1, np.pi / 180, 127, minLineLength=10, maxLineGap=250)
        try:
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.frame, (x1, y1 + self.height_shift), (x2, y2 + self.height_shift), (0, 255, 0), 2)
        except Exception as ex:
            print(ex)

        self.final_frame = self.frame_postprocessing(frame=self.frame, processed_frame=self.roi_edges, scale=0.7)
        return "Lane Detection", self.final_frame