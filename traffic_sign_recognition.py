import numpy as np
import cv2
import pyopencl as cl
import pyopencl.cltypes
import glob
import os, sys
import time
from sign import Sign
from GPUSetup import GPUSetup
from detected_objects import DetectedObjects

class TrafficSignRecognition():
    def __init__(self):
        self.templates = []
        self.templates_hsv = []
        self.templates_mask = []
        self.template_preprocesing()
        self.old_signs = []

    def sort_filenames(self, filename):     #extract just file name (which is number and later index)
        filename = filename[::-1]
        return (int)((filename[4:filename.find('/')])[::-1])

    def template_preprocesing(self):
        names = []
        for filename in glob.iglob(os.getcwd() + '/templates/*.png', recursive=True):
            names.append(filename)

        for name in sorted(names, key=self.sort_filenames):
            template = cv2.cvtColor(cv2.imread(name), cv2.COLOR_RGB2RGBA)
            self.templates.append(template)

            h = template.shape[0]
            w = template.shape[1]

            # *Buffors
            template_buf = cl.image_from_array(GPUSetup.context, template, 4)
            fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
            dest_buf = cl.Image(GPUSetup.context, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

            # *RGB to HSV
            GPUSetup.program.rgb2hsv(GPUSetup.queue, (w, h), None, template_buf, dest_buf)
            template_hsv = np.empty_like(template)
            cl.enqueue_copy(GPUSetup.queue, template_hsv, dest_buf, origin=(0, 0), region=(w, h))
            self.templates_hsv.append(template_hsv)

            # *Apply masks
            template_mask = self.clear_sign(template_hsv, template)
            self.templates_mask.append(template_mask)

    def clear_sign(self,img_hsv, img=None):
        h = img_hsv.shape[0]
        w = img_hsv.shape[1]
        cont_img_hsv = np.ascontiguousarray(img_hsv)

        red_mask = np.zeros((1, 2), cl.cltypes.float4)
        red_mask[0, 0] = (150, 70, 40, 0)  # Lower bound red
        red_mask[0, 1] = (210, 255, 255, 0)  # Upper bound red
        
        black_mask = np.zeros((1, 2), cl.cltypes.float4)
        black_mask[0, 0] = (0, 0, 0, 0)  # Lower bound black
        black_mask[0, 1] = (255, 255, 100, 0)  # Upper bound black

        #*Buffors
        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        dest_buf = cl.Image(GPUSetup.context, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

        #*Red mask
        img_buf = cl.image_from_array(GPUSetup.context, cont_img_hsv, 4)
        mask_buf = cl.Buffer(GPUSetup.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=red_mask)
        GPUSetup.program.hsv_bin_mask(GPUSetup.queue, (w, h), None, img_buf, mask_buf, dest_buf)
        img_mask_red = np.empty_like(cont_img_hsv)
        cl.enqueue_copy(GPUSetup.queue, img_mask_red, dest_buf, origin=(0, 0), region=(w, h))

        #*Black mask
        img_buf = cl.image_from_array(GPUSetup.context, cont_img_hsv, 4)
        mask_buf = cl.Buffer(GPUSetup.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=black_mask)
        GPUSetup.program.hsv_bin_mask_center(GPUSetup.queue, (w, h), None, img_buf, mask_buf, np.int32(w), np.int32(h), dest_buf)
        img_mask_black = np.empty_like(cont_img_hsv)
        cl.enqueue_copy(GPUSetup.queue, img_mask_black, dest_buf, origin=(0, 0), region=(w, h))

        #*Merge both
        img_buff_red = cl.image_from_array(GPUSetup.context, img_mask_red, 4)
        img_buff_black = cl.image_from_array(GPUSetup.context, img_mask_black, 4)
        GPUSetup.program.merge_bin(GPUSetup.queue, (w, h), None, img_buff_red, img_buff_black, dest_buf)
        img_merge = np.empty_like(cont_img_hsv)
        cl.enqueue_copy(GPUSetup.queue, img_merge, dest_buf, origin=(0, 0), region=(w, h))

        # # *-----------------------------DEBUG---------------------------------
        # img_concate_Verti = np.concatenate((img, img_hsv), axis=0)
        # img_gray = img_hsv[:,:,2]
        # blur = cv2.GaussianBlur(img_gray,(5,5),0)
        # _, img_otsu = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)

        # cv2.imshow('Original and HSV', img_concate_Verti)
        # cv2.imshow('Gray', img_gray)
        # cv2.imshow("Otsu by cv", img_otsu)
        # cv2.imshow("Hsv", img_hsv)
        # cv2.imshow("After Red Mask",img_mask_red)
        # cv2.imshow("After Black Mask",img_mask_black)
        # cv2.imshow("Merge of masks", img_merge)

        # key = cv2.waitKey(10)
        # while (key != ord('w')):
        #     key = cv2.waitKey(19)
        #     pass
        # # *-------------------------------------------------------------------

        return img_merge

    def load_new_frame(self, frame):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        self.height, self.width, self.channels = frame.shape
        self.objects_coords = []
        self.signs = []
        self.marked_fame = frame

    def frame_preprocessing(self):
        # *Load and convert source image
        frame = np.array(self.frame)

        # *Set properties
        h = frame.shape[0]
        w = frame.shape[1]
        mask = np.zeros((1, 2), cl.cltypes.float4)
        mask[0, 0] = (165, 90, 70, 0)  # Lower bound
        mask[0, 1] = (195, 255, 255, 0)  # Upper bound

        # *Buffors
        frame_buf = cl.image_from_array(GPUSetup.context, frame, 4)
        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        dest_buf = cl.Image(GPUSetup.context, cl.mem_flags.WRITE_ONLY, fmt, shape=(w, h))

        # *RGB to HSV
        GPUSetup.program.rgb2hsv(GPUSetup.queue, (w, h), None, frame_buf, dest_buf)
        self.hsv = np.empty_like(frame)
        cl.enqueue_copy(GPUSetup.queue, self.hsv, dest_buf, origin=(0, 0), region=(w, h))

        # *Apply mask
        frame_buf = cl.image_from_array(GPUSetup.context, self.hsv, 4)
        mask_buf = cl.Buffer(GPUSetup.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mask)
        GPUSetup.program.hsv_mask(GPUSetup.queue, (w, h), None, frame_buf, mask_buf, dest_buf)
        self.after_mask = np.empty_like(frame)
        cl.enqueue_copy(GPUSetup.queue, self.after_mask, dest_buf, origin=(0, 0), region=(w, h))
        return self.after_mask

    def connected_components(self, offset=5, min_object_size=500):
        self.frame_preprocessing()
        start_time_all = time.time()
        label = 1
        # Coordinates (indices [x, y]) of pixels with R channel (BGR code) greater than 40
        coords = np.argwhere(self.after_mask[:, :, 2] > 40)
        labels = np.zeros(shape=(self.height, self.width), dtype=int)

        # Assigning initial labels for pixels with specific coordinates
        for coord in coords:
            labels[coord[0], coord[1]] = label
            label += 1

        # Connecting pixels using kernels (8 connectivity)
        transitions = 0
        timer = 0;
        timer_kernel = 0;
        while True:
            try:
                start_time_kernel = time.time()
                mem_flags = cl.mem_flags
                # build input & destination labels array
                labels_cc_buf = cl.Buffer(GPUSetup.context,  mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, size=labels.nbytes, hostbuf=labels)
                # execute OpenCL function
                GPUSetup.program.connected_components(GPUSetup.queue, labels.shape, None, np.int32(self.width), np.int32(self.height), labels_cc_buf)
                # copy result back to host
                labels_cc = np.empty_like(labels)
                cl.enqueue_copy(GPUSetup.queue, labels_cc, labels_cc_buf)
                elapsed_time_kernel = time.time() - start_time_kernel
                timer_kernel+=elapsed_time_kernel
                start_time = time.time()

                if((labels_cc==labels).all()):
                    elapsed_time = time.time() - start_time
                    timer+=elapsed_time
                    break
                else:
                    elapsed_time = time.time() - start_time
                    timer += elapsed_time
                    labels=labels_cc
                    transitions += 1
            except Exception as ex:
                print(ex)
        print("\n______NEW_FRAME_______")
        print("Elapsed time in kernels: ", timer_kernel)
        print("Elapsed time python: ", timer)
        print("Labels connected. Transitions: ", transitions)

        detected_objects = DetectedObjects(labels, coords, self.width, self.height)
        self.objects_coords = detected_objects.objects_coords
        self.signs = detected_objects.signs

        #Drawing detected objects
        for sign in self.signs:
            cv2.rectangle(self.frame, (sign.x, sign.y), (sign.x+sign.width, sign.y+sign.height), (0, 255, 0), 2)

        elapsed_time_all = time.time() - start_time_all
        print("Whole Sign Detection: ", elapsed_time_all)
        self.templateSumSquare()
        return self.frame[:,:,:3], self.marked_fame

    def templateSumSquare(self, error_limit = 10000, old_sign_err_mult = 0.9, old_sign_pix_diff=20):
        start_time = time.time()
        for sign in self.signs:
            #*Load sign + masking and binaryzation
            frame_img = self.hsv[sign.y:sign.y+sign.height, sign.x:sign.x+sign.width]
            frame_masked =  np.array(self.clear_sign(frame_img)[:,:,2]).astype(np.float32)
            frame_masked_flat = frame_masked.flatten()

            single_sign_results = []
            for template in self.templates_mask:
                #*Load template + masking and binaryzation
                template_arr = cv2.resize(template, (sign.width, sign.height), interpolation=cv2.INTER_AREA)
                template_masked = np.array(template_arr[:,:,2]).astype(np.float32)
                template_masked_flat = template_masked.flatten()

                #*Calculate error/difference
                template_buf = cl.Buffer(GPUSetup.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=template_masked_flat)
                frame_buf = cl.Buffer(GPUSetup.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=frame_masked_flat)
                ssd_buf = cl.Buffer(GPUSetup.context, cl.mem_flags.WRITE_ONLY, template_masked_flat.nbytes)
                GPUSetup.program.square_sum(GPUSetup.queue, template_masked_flat.shape, None, template_buf, frame_buf, ssd_buf)

                #*copy result back to host
                ssd = np.empty_like(template_masked_flat)
                cl.enqueue_copy(GPUSetup.queue, ssd, ssd_buf)
                single_sign_results.append(np.sum(ssd)/len(ssd))

            #*Take into account previous frame classification
            for old_sign in self.old_signs: 
                if(old_sign.type is not None):
                    if (np.abs(old_sign.x-sign.x) < old_sign_pix_diff) and (np.abs(old_sign.y-sign.y) < old_sign_pix_diff):
                        single_sign_results[old_sign.type] *= old_sign_err_mult
            if min(single_sign_results) < error_limit:
                sign.type = np.argmin(single_sign_results)

            # *--------------------------DEBUG/INFO-------------------------------
            if sign.type is not None:
                print("\n+++Sign recognized+++\nType: ", sign.type, " | X: ", sign.x, " | Y: ", sign.y, " | Width: ", sign.width,
                    " | Height: ", sign.height)
                print("Min Sum squared errors: ", min(single_sign_results))
            else:
                print("\n-----Not a Sign------\nType: ", sign.type, " | X: ", sign.x, " | Y: ", sign.y, " | Width: ", sign.width,
                    " | Height: ", sign.height)
                print("Min Sum squared errors: ", min(single_sign_results))
            # *-------------------------------------------------------------------

        #*Update history
        self.old_signs = self.signs

        #* Visualy overlap result on self.marked_frame
        for sign in self.signs:
            if sign.type is not None:
                template_arr = cv2.resize(self.templates_mask[sign.type], (sign.width, sign.height), interpolation=cv2.INTER_AREA)
                template_masked = np.array(template_arr[:,:,2]).astype(np.float32)
                template_masked = np.dstack((template_masked,template_masked,template_masked))
                self.marked_fame[sign.y:sign.y+sign.height, sign.x:sign.x+sign.width] = template_masked

        elapsed_time = time.time() - start_time
        print("Sign recognition time: ", elapsed_time)
