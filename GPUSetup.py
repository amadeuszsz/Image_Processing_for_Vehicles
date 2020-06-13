import numpy as np
import cv2
import pyopencl as cl
import pyopencl.cltypes


class GPUSetup():
    def __init__(self):
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.program = cl.Program(self.context, open('kernels/kernels.cl').read()).build()
        self.queue= cl.CommandQueue(self.context)