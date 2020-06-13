import numpy as np
import cv2
import pyopencl as cl
import pyopencl.cltypes

from sign import Sign
from GPUSetup import GPUSetup
from templates import Templates


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

    def templateSumSquare(self):
        templates = Templates().templates
        full_frame_img = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        results = []

        # #!Temporary
        # template_img2 = cv2.imread('templates/0.png')
        # self.signs.append(Sign(x=10,y=10,height=10,width=10))
        # #!--------

        for sign in self.signs:
            # Load sign + Otsu's thresholding and Gaussian filtering
            frame_img = full_frame_img[sign.y:sign.y+sign.height, sign.x:sign.y+sign.width]
            blur = cv2.GaussianBlur(frame_img,(5,5),0)
            _, frame_arr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
            frame_arr = np.array(frame_img).astype(np.float32)

            single_sign_results = []
            for template_img in templates:
                # Load template + Otsu's thresholding and Gaussian filtering for Template
                template_img = cv2.resize(template_img, (sign.width, sign.height), interpolation = cv2.INTER_AREA)
                template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(template_img,(5,5),0)
                _, template_arr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
                template_arr = np.array(template_img).astype(np.float32)

                mem_flags = cl.mem_flags
                # build input images buffer
                template_buf = cl.Buffer(GPUSetup.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=template_arr)
                frame_buf = cl.Buffer(GPUSetup.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=frame_arr)

                # build destination OpenCL Image
                mse_buf = cl.Buffer(GPUSetup.context, mem_flags.WRITE_ONLY, template_arr.nbytes)

                # execute OpenCL function
                GPUSetup.program.square_sum(GPUSetup.queue, template_arr.shape, None, template_buf, frame_buf, mse_buf)

                # copy result back to host
                mse = np.empty_like(template_arr)
                cl.enqueue_copy(GPUSetup.queue, mse, mse_buf)
                single_sign_results.append(np.sum(mse))
            results.append(np.argmax(single_sign_results))
        return results