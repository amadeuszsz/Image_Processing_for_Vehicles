import cv2
import glob
import os, sys

class Templates:
    templates = []
    def __init__(self):
        for filename in glob.iglob(os.getcwd()+ '/templates/*.png', recursive=True):
            self.templates.append(cv2.imread(filename))