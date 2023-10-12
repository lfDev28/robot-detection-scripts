from Detector.Feature import Feature
from Detector.EdgeDetector import EdgeDetector
from Detector.SignDetector import SignDetector
from Detector import Detector


if __name__ == '__main__':
    # Initialising all the detectors
    edge_detector = EdgeDetector()
    feature_detector = Feature()
    sign_detector = SignDetector()

    # Adding the detectors to the main loop

    # Initialising the main detector and running the main loop
    main_detector = Detector()
    main_detector.main_loop([edge_detector])
