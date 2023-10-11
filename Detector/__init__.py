import cv2

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Detector(metaclass=SingletonMeta):
    def __init__(self, source=0):
        if not hasattr(self, "cap"):  # Check if cap already exists
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

    def main_loop(self, detectors):
        while True:
            frame = self.get_frame()
            for detector in detectors:
                processed_frame = detector.process_frame(frame)
                detector_name = type(detector).__name__
                self.show_frame(processed_frame, window_name=detector_name)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release()
