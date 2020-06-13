import numpy as np
import cv2
import pyopencl as cl
import pyopencl.cltypes

from sign import Sign
from GPUSetup import GPUSetup
from templates import Templates


class TrafficSignRecognition():
    signs = [] # of type Sign

    def __init__(self, frame):
        self.frame = frame

    def frame_preprocessing(self):
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = np.array([165, 0, 0])
        upper_blue = np.array([195, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(self.hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        self.after_mask = cv2.bitwise_and(self.frame, self.frame, mask=mask)

        return self.after_mask

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
