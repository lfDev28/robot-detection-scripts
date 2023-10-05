
import cv2
import numpy as np
from Detector import Detector
from helpers import SHAPES, COLOURS

class Feature(Detector):
    def __init__(self, source=0):
        super().__init__(source)

    def detect_feature(self, frame, target_color, target_shape):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Prepare the color mask based on the target color
        if target_color in COLOURS:
            masks = [cv2.inRange(hsv, pair[0], pair[1]) for pair in COLOURS[target_color]]

            mask = cv2.bitwise_or(masks[0], masks[1]) if len(masks) > 1 else masks[0]
        else:
            raise ValueError(f"Color {target_color} not supported!")
            
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if target_shape in SHAPES:
                shape_fn = SHAPES[target_shape]
                shape_result = shape_fn(contour)

                if target_shape == "circle":
                    if shape_result:  # Check if shape_result is not None
                        center, radius = shape_result
                        cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
                        
                elif target_shape == "square" or target_shape == "rectangle":
                    shape_type, approx = shape_result
                    if shape_type == target_shape:
                        x, y, w, h = cv2.boundingRect(approx)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                elif shape_result:  # For triangles
                    x, y, w, h = cv2.boundingRect(shape_result)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                
        return frame