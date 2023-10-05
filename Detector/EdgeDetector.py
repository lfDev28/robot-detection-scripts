from Detector import Detector
import cv2

class EdgeDetector(Detector):
    def __init__(self, source=0):
        super().__init__(source)
    
    def detect_edges(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        return edges

    def process(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Detect edges
            edges = self.detect_edges(frame)

            # Display the result
            cv2.imshow('Edges', edges)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

