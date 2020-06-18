# Traffic Sign Recognition

The application allows for detection and recognition of traffic signs from a video signal. The project was
created using the Python programming language and the OpenCL framework, which provides the ability
to use the graphics card during image processing. For character detection, HSV was used to extract
interesting image properties and then the Connected Components algorithm detects single objects. Af-
ter that, using the predefined templates database, the application compares the detected objects with
templates. The type of traffic sign is determined when relevant similarity is achieved.

To run type "python3 main.py path_to_video".
