import numpy as np
import cv2
import pyopencl as cl
import pyopencl.cltypes


class GPUSetup():
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    program = cl.Program(context, open('kernels/kernels.cl').read()).build()
    queue= cl.CommandQueue(context)