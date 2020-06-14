import numpy as np
import cv2
import pyopencl as cl
import pyopencl.cltypes
import glob
import os, sys

from sign import Sign
from GPUSetup import GPUSetup


class TrafficSignRecognition():
    def __init__(self, frame):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        self.height, self.width, self.channels = frame.shape
        self.objects_coords = []
        self.signs = []
        self.templates = []
        self.templates_hsv = []
        self.template_preprocesing()


    def template_preprocesing(self):
        for filename in glob.iglob(os.getcwd()+ '/templates/*.png', recursive=True):
            template = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_RGB2RGBA)
            self.templates.append(template)

            h = template.shape[0]
            w = template.shape[1]       
            #*Buffors
            template_buf = cl.image_from_array(GPUSetup.context, template, 4)
            fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
            dest_buf = cl.Image(GPUSetup.context, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))
            
            #*RGB to HSV
            GPUSetup.program.rgb2hsv(GPUSetup.queue, (w, h), None, template_buf, dest_buf)
            template_hsv = np.empty_like(template)
            cl.enqueue_copy(GPUSetup.queue, template_hsv, dest_buf, origin=(0, 0), region=(w, h))
            self.templates_hsv.append(template_hsv)


    def frame_preprocessing(self):
        #*Load and convert source image
        frame = np.array(self.frame)
        
        #*Set properties
        h = frame.shape[0]
        w = frame.shape[1]
        mask = np.zeros((1, 2), cl.cltypes.float4)
        mask[0, 0] = (165, 120, 70, 0)      #Lower bound 
        mask[0, 1] = (195, 255, 255, 0)     #Upper bound
        
        #*Buffors
        frame_buf = cl.image_from_array(GPUSetup.context, frame, 4)
        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        dest_buf = cl.Image(GPUSetup.context, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))
        mask_buf = cl.Buffer(GPUSetup.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mask)
        
        #*RGB to HSV
        GPUSetup.program.rgb2hsv(GPUSetup.queue, (w, h), None, frame_buf, dest_buf)
        self.hsv = np.empty_like(frame)
        cl.enqueue_copy(GPUSetup.queue, self.hsv, dest_buf, origin=(0, 0), region=(w, h))
        
        #*Apply mask
        frame_buf = cl.image_from_array(GPUSetup.context, self.hsv, 4)
        GPUSetup.program.hsvMask(GPUSetup.queue, (w, h), None, frame_buf,mask_buf, dest_buf)
        self.after_mask = np.empty_like(frame)
        cl.enqueue_copy(GPUSetup.queue, self.after_mask, dest_buf, origin=(0, 0), region=(w, h))

        return self.after_mask

        # self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        # # define range of blue color in HSV
        # lower_blue = np.array([165, 120, 70])
        # upper_blue = np.array([195, 255, 255])
        # # Threshold the HSV image to get only blue colors
        # mask = cv2.inRange(self.hsv, lower_blue, upper_blue)
        # # Bitwise-AND mask and original image
        # self.after_mask = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        # return self.after_mask


    def connected_components(self, offset=5, min_object_size=100):
        self.frame_preprocessing()
        label = 1
        # Coordinates (indices [x, y]) of pixels with R channel (BGR code) greater than 40
        coords = np.argwhere(self.after_mask[:, :, 2] > 40)
        labels = np.zeros(shape=(self.height, self.width), dtype=int)

        # Assigning initial labels for pixels with specific coordinates
        for coord in coords:
            labels[coord[0], coord[1]] = label
            label += 1

        # Connecting pixels (8 connectivity)
        while True:
            try:
                break_flag = 1
                for coord in coords:
                    # (3+offset)x(3+offset) matrix with center of coord variable
                    neighbouring_pixels = labels[coord[0] - 1 - offset:coord[0] + 2 + offset,
                                          coord[1] - 1 - offset:coord[1] + 2 + offset]
                    # Minimum value of label excluding 0
                    min_label = np.min(neighbouring_pixels[np.nonzero(neighbouring_pixels)])
                    # If any connection then keep loop
                    if (labels[coord[0], coord[1]] > min_label):
                        labels[coord[0], coord[1]] = min_label
                        break_flag = 0
                if break_flag:
                    break
            except Exception as ex:
                print(ex)

        # List of objects (unique label on frame)
        objects = np.unique(labels)
        objects = np.delete(objects, np.where(objects == 0))

        # Rejecting small objects
        for object in objects:
            if np.count_nonzero(labels == object) < min_object_size:
                labels[labels == object] = 0
                objects = np.delete(objects, np.where(objects == object))

        # Getting coords of objects
        for object in objects:
            most_left = self.width
            most_right = 0
            most_top = self.height
            most_bottom = 0
            for coord in coords:
                if labels[coord[0], coord[1]] == object:
                    if (coord[1] < most_left): most_left = coord[1]
                    if (coord[1] > most_right): most_right = coord[1]
                    if (coord[0] < most_top): most_top = coord[0]
                    if (coord[0] > most_bottom): most_bottom = coord[0]
            self.objects_coords.append([(most_left, most_top), (most_right, most_bottom)])
            sign = Sign(x=most_left, y=most_top, width=most_right - most_left, height=most_bottom - most_top)
            self.signs.append(sign)
            cv2.rectangle(self.frame, self.objects_coords[0][0], self.objects_coords[0][1], (0, 255, 0), 2)

        # for coord in coords:
        #     if labels[coord[0], coord[1]] > 0:
        #         self.after_mask[coord[0], coord[1]] = [255, 0, 0]

        # return self.frame
        for coord in coords:
            if labels[coord[0], coord[1]] > 0:
                self.after_mask[coord[0], coord[1]] = [255, 0, 0,0]
        self.templateSumSquare()
        return self.frame


    def templateSumSquare(self):
        print("***********************\nTemplates num: ", len(self.templates))
        print("Signs num: ", len(self.signs))
        # full_frame_img = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2GRAY)
        full_frame_img = self.hsv[:,:,2]
        results = []

        for sign in self.signs:
            # Load sign + Otsu's thresholding and Gaussian filtering
            frame_img = full_frame_img[sign.y:sign.y+sign.height, sign.x:sign.x+sign.width]
            blur = cv2.GaussianBlur(frame_img,(5,5),0)
            _, frame_arr = cv2.threshold(frame_img,0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)

            single_sign_results = []
            for template in self.templates_hsv:
                # Load template + Otsu's thresholding and Gaussian filtering for Template
                template_img = cv2.resize(template, (sign.width, sign.height), interpolation = cv2.INTER_AREA)     
                # template_img = cv2.cvtColor(template_img, cv2.COLOR_HSV2GRAY)
                template_img = template_img[:,:,2]

                blur = cv2.GaussianBlur(template_img,(5,5),0)
                _, template_arr = cv2.threshold(template_img,0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)


            # *-----------------------------DEBUG---------------------------------
                img_concate_Verti=np.concatenate((frame_arr, template_arr),axis=0)
                cv2.imshow('concatenated_Verti',img_concate_Verti)
                key = cv2.waitKey(10)
                while(key!=ord('w')):
                    key = cv2.waitKey(19)
                    pass
            # *-------------------------------------------------------------------

                mem_flags = cl.mem_flags
                # build input images buffer
                template_buf = cl.Buffer(GPUSetup.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=template_arr)
                frame_buf = cl.Buffer(GPUSetup.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=frame_arr)

                # build destination OpenCL Image
                ssd_buf = cl.Buffer(GPUSetup.context, mem_flags.WRITE_ONLY, template_arr.nbytes)

                # execute OpenCL function
                GPUSetup.program.square_sum(GPUSetup.queue, template_arr.shape, None, template_buf, frame_buf, ssd_buf)

                # copy result back to host
                ssd = np.empty_like(template_arr)
                cl.enqueue_copy(GPUSetup.queue, ssd, ssd_buf)
                single_sign_results.append(np.sum(ssd))

            results.append(np.argmin(single_sign_results))
            if min(single_sign_results) < 8000:
                sign.type = np.argmin(single_sign_results)

            # *----------------------------DEBUG----------------------------------
            if sign.type is not None:
                print("\n++++++++++++++\nType: ", sign.type, "\nX: ",sign.x, "\nT: ", sign.y, "\nWidth: ", sign.width,"\nHeight: ", sign.height)
                print("Sum squared errors: ", single_sign_results)
            else:
                print("\n--------------\nType: ", sign.type, "\nX: ",sign.x, "\nT: ", sign.y, "\nWidth: ", sign.width,"\nHeight: ", sign.height)
                print("Sum squared errors: ", single_sign_results)
            # *-------------------------------------------------------------------
        return results
