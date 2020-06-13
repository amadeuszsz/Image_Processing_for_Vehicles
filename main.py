import numpy as np
import cv2
import time
import pyopencl as cl
import pyopencl.cltypes

from lane_detection import LaneDetection
from traffic_sign_recognition import TrafficSignRecognition
from GPUSetup import GPUSetup

#Video variables
cap = cv2.VideoCapture('videos/video5.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

#OpenCL variables
gpuSetup = GPUSetup()

def sync_fps(time_start):
    '''
    Sleeping function. Wait until reach desired frame rate.
    :param time_start: last frame timestamp
    :return:
    '''
    timeDiff = time.time() - time_start
    if (timeDiff < 1.0 / (fps)):
        time.sleep(1.0 / (fps) - timeDiff)


def gpuTest():
    vector = np.zeros((1, 1), cl.cltypes.float4)
    matrix = np.zeros((1, 4), cl.cltypes.float4)
    matrix[0, 0] = (1, 2, 4, 8)
    matrix[0, 1] = (16, 32, 64, 128)
    matrix[0, 2] = (3, 6, 9, 12)
    matrix[0, 3] = (5, 10, 15, 25)
    vector[0, 0] = (1, 2, 4, 8)

    ## Step #8. Allocate device memory and move input data from the host to the device memory.
    mem_flags = cl.mem_flags
    matrix_buf = cl.Buffer(gpuSetup.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=matrix)
    vector_buf = cl.Buffer(gpuSetup.context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=vector)
    matrix_dot_vector = np.zeros(4, np.float32)
    destination_buf = cl.Buffer(gpuSetup.context, mem_flags.WRITE_ONLY, matrix_dot_vector.nbytes)
     
    ## Step #9. Associate the arguments to the kernel with kernel object.
    ## Step #10. Deploy the kernel for device execution.
    gpuSetup.program.matrix_dot_vector(gpuSetup.queue, matrix_dot_vector.shape, None, matrix_buf, vector_buf, destination_buf)
     
    ## Step #11. Move the kernel’s output data to host memory.
    cl.enqueue_copy(gpuSetup.queue, matrix_dot_vector, destination_buf)
     
    ## Step #12. Release context, program, kernels and memory.
    ## PyOpenCL performs this step for you, and therefore,
    ## you don't need to worry about cleanup code
     
    print(matrix_dot_vector)

def main():
    while (cap.isOpened()):
        time_start = time.time()
        ret, frame = cap.read()
        #######################################
        #Do things here
        #x = LaneDetection(frame)
        #name, final_frame = x.lane_detection()
        name = "Traffic Sign Recognition"
        final_frame = TrafficSignRecognition(frame).connected_components()
        #######################################
        cv2.imshow(name, final_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('w'):
            cv2.waitKey(-1)
        sync_fps(time_start=time_start)
    cap.release()
    cv2.destroyAllWindows()
    #gpuTest()


if __name__ == "__main__":
    main()