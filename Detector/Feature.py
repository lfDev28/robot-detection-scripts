import cv2
from Detector import Detector
from helpers import SHAPES, COLOURS, close_screen_capture


class Feature(Detector):
    def __init__(self, source=0):
        super().__init__(source)
        self.preprocessing_steps = [
            "gaussian_blur", "median_blur"]
        self.targets = [("red", "square"), ("blue", "square"),
                        ("yellow", "square"), ("green", "square")]

    def process_frame(self, frame):
        return self.detect_features(frame)

    def detect_features(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for target_color, target_shape in self.targets:
            if target_color in COLOURS:

                masks = [cv2.inRange(hsv, pair[0], pair[1])
                         for pair in COLOURS[target_color]]
                mask = cv2.bitwise_or(masks[0], masks[1]) if len(
                    masks) > 1 else masks[0]
            else:
                raise ValueError(f"Color {target_color} not supported!")
            mask = cv2.medianBlur(mask, 5)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 1000:
                    continue

                shape_fn = SHAPES.get(target_shape)
                if not shape_fn:
                    print(
                        f"Warning: Shape function for {target_shape} not found.")
                    continue
                shape_result = shape_fn(contour)
                if not shape_result:
                    continue
                if target_shape == "circle":
                    center, radius = shape_result
                    cv2.circle(frame, (int(center[0]), int(
                        center[1])), int(radius), (0, 255, 0), 2)
                    cv2.putText(frame, f"{target_color} {target_shape}", (int(center[0] - radius), int(center[1] - radius) - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                elif target_shape in ["square", "rectangle"]:
                    shape_type, approx = shape_result
                    if shape_type == target_shape:
                        x, y, w, h = cv2.boundingRect(approx)
                        cv2.rectangle(frame, (x, y), (x+w, y+h),
                                      (0, 255, 0), 2)
                        cv2.putText(frame, f"{target_color} {shape_type}", (x, y - 10),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                elif target_shape == "triangle":
                    x, y, w, h = cv2.boundingRect(shape_result)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{target_color} {target_shape}", (x, y - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        return frame
