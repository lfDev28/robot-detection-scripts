import cv2
import os
from Detector import Detector
from helpers import close_screen_capture


class SignDetector(Detector):
    def __init__(self, source=0):
        super().__init__(source)

        current_wd = os.getcwd()
        classFile = os.path.join(
            current_wd, "Object_Detection_Files", "coco.names")
        with open(classFile, "rt") as f:
            self.classNames = f.read().rstrip("\n").split("\n")

        configPath = os.path.join(
            current_wd, "Object_Detection_Files", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
        weightsPath = os.path.join(
            current_wd, "Object_Detection_Files", "frozen_inference_graph.pb")

        self.net = cv2.dnn_DetectionModel(weightsPath, configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def process_frame(self, frame):
        return self.detect_sign(frame)

    def detect_sign(self, frame):
        _, objectInfo = self.getObjects(
            frame, 0.45, 0.2, objects=["stop sign"])

        for _, className in objectInfo:
            if className == "stop":
                print("STOP")

        return frame

    def getObjects(self, img, thres, nms, draw=True, objects=[]):
        classIds, confs, bbox = self.net.detect(
            img, confThreshold=thres, nmsThreshold=nms)
        objectInfo = []
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = self.classNames[classId - 1]
                if className in objects:
                    objectInfo.append([box, className])
                    if draw:
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, className.upper(), (box[0]+10, box[1]+30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        return img, objectInfo
