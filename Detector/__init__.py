import cv2
import numpy as np



class Detector:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        _, frame = self.cap.read()
        return frame

    def show_frame(self, frame, window_name="Frame"):
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

        






    





