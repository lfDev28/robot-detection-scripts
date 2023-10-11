from Detector.Feature import Feature
from Detector.EdgeDetector import EdgeDetector
from Detector.SignDetector import SignDetector
from Detector import Detector

if __name__ == '__main__':
    edge_detector = EdgeDetector()
    feature_detector = Feature()
    sign_detector = SignDetector()

    main_detector = Detector()
    main_detector.main_loop([edge_detector, feature_detector, sign_detector])
