from Detector.Feature import Feature
from Detector.SignDetector import SignDetector
import cv2
import numpy as np



def feature():
    feature_detector = Feature(source=0)

    try:
        while True:
            frame = feature_detector.get_frame()
            detected_frame = feature_detector.detect_feature(frame, "yellow", "circle")
            feature_detector.show_frame(detected_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    except KeyboardInterrupt:
        pass
    finally:
        feature_detector.release()


def main():
    sign_detector = SignDetector()
    while True:
        frame = sign_detector.get_frame()
        detected_frame = sign_detector.detect_sign(frame)
        sign_detector.show_frame(detected_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    sign_detector.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

        



