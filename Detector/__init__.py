import cv2
import numpy as np


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Detector(metaclass=SingletonMeta):
    def __init__(self, source=0):
        if not hasattr(self, "cap"):
            self.cap = cv2.VideoCapture(source)
        self.preprocessing_steps = []

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame.")
            return None
        return frame

    def preprocess_frame(self, frame):
        if "gaussian_blur" in self.preprocessing_steps:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
        if "median_blur" in self.preprocessing_steps:
            frame = cv2.medianBlur(frame, 5)

        if "bilateral_filter" in self.preprocessing_steps:
            frame = cv2.bilateralFilter(frame, 9, 75, 75)

        if "morph_close" in self.preprocessing_steps:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

        if "morph_open" in self.preprocessing_steps:
            kernel = np.ones((5, 5), np.uint8)
            frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

        return frame

    def show_frame(self, frame, window_name="Frame"):
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def main_loop(self, detectors):
        while True:
            frame = self.get_frame()
            if frame is None:
                continue
            for detector in detectors:
                preprocessed_frame = self.preprocess_frame(frame)
                processed_frame = detector.process_frame(preprocessed_frame)
                detector_name = type(detector).__name__
                self.show_frame(processed_frame, window_name=detector_name)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.release()
