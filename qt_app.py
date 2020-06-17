'''
Simple Qt gui for traffic signs recognition.
Not working yet.
'''

import cv2
import sys
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QDialog, QPushButton, QInputDialog, QFileDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QDir, QUrl
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import time
import pyopencl as cl
import pyopencl.cltypes
import time
import imutils
import os

from lane_detection import LaneDetection
from traffic_sign_recognition import TrafficSignRecognition
from GPUSetup import GPUSetup



'''
To display an OpenCV image, we have to convert the image 
into a QImage then into a QPixmap where we can display 
the image with a QLabel
'''

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        signRecognizer = TrafficSignRecognition()
        name = "Traffic Sign Recognition"
        self.filepath = ""
        cap = cv2.VideoCapture(self.filepath)
        if self.filepath:
            while True:
                ret, frame = cap.read()
                if ret:
                    ret, frame = cap.read()
                    signRecognizer.load_new_frame(frame)
                    original_frame, marked_frame = signRecognizer.connected_components()
                    final_frame = np.concatenate((original_frame, marked_frame), axis=0)
                    rgbImage = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(768, 864, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    if key == ord('w'):
                        cv2.waitKey(-1)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 768 + 2*20 + 500
        self.height = 864 + 2*20
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def import_video(self):
        #TODO
        self.filepath = QFileDialog.getOpenFileUrl(self, "Open file", directory=QUrl(os.path.abspath(os.path.dirname(__file__))),
                                                   filter="Video files (*.mp4)")

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Create a button
        self.import_video_button = QPushButton("Import video", self)
        self.import_video_button.move(808, 20)
        self.import_video_button.clicked.connect(self.import_video)
        # Create a video label
        self.label = QLabel(self)
        self.label.move(20, 20)
        self.label.resize(768, 864)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())