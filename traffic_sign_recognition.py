import numpy as np
import cv2

import sign
from GPUSetup import GPUSetup

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
        self.after_mask = cv2.bitwise_and(self.frame, self.frame, mask=self.mask)

        return self.after_mask

    def templateSumSquare(self):
        for sign in signs:
            vector = np.zeros((1, 1), cl.cltypes.float4)
            matrix = np.zeros((1, 4), cl.cltypes.float4)
            matrix[0, 0] = (1, 2, 4, 8)
            matrix[0, 1] = (16, 32, 64, 128)
            matrix[0, 2] = (3, 6, 9, 12)
            matrix[0, 3] = (5, 10, 15, 25)
            vector[0, 0] = (1, 2, 4, 8)

            ## Step #8. Allocate device memory and move input data from the host to the device memory.
            mem_flags = cl.mem_flags
            matrix_buf = cl.Buffer(GPUSetup.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=matrix)
            vector_buf = cl.Buffer(GPUSetup.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=vector)
            matrix_dot_vector = np.zeros(4, np.float32)
            destination_buf = cl.Buffer(GPUSetup.context, mem_flags.WRITE_ONLY, matrix_dot_vector.nbytes)
            
            ## Step #9. Associate the arguments to the kernel with kernel object.
            ## Step #10. Deploy the kernel for device execution.
            GPUSetup.program.matrix_dot_vector(GPUSetup.queue, matrix_dot_vector.shape, None, matrix_buf, vector_buf, destination_buf)
            
            ## Step #11. Move the kernel’s output data to host memory.
            cl.enqueue_copy(GPUSetup.queue, matrix_dot_vector, destination_buf)
            
            ## Step #12. Release context, program, kernels and memory.
            ## PyOpenCL performs this step for you, and therefore,
            ## you don't need to worry about cleanup code
            
            print(matrix_dot_vector)

