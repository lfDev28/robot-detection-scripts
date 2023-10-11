import cv2
from Detector import Detector


class EdgeDetector(Detector):
    def process_frame(self, frame):
        return self.detect_edges(frame)

    def detect_edges(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        return edges
