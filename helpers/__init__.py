
import cv2 
import numpy as np

def is_square_or_rectangle(contour):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:  # Modify these values as necessary for better accuracy.
            return "square", approx
        else:
            return "rectangle", approx
    return None, None

def is_circle(contour):
    center, radius = cv2.minEnclosingCircle(contour)
    true_area = cv2.contourArea(contour)
    expected_area = np.pi * (radius ** 2)
    
    if 0.8 <= (true_area / expected_area) <= 1.2:  # Adjust as necessary
        return center, radius
    return None

SHAPES = {
    "square": is_square_or_rectangle,
    "rectangle": is_square_or_rectangle,
    "triangle": lambda contour: cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True) if len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)) == 3 else None,
    "circle": is_circle
}
COLOURS = {
    "red": [
        (np.array([0, 50, 50], dtype="uint8"), np.array([12, 255, 255], dtype="uint8")),
        (np.array([168, 50, 50], dtype="uint8"), np.array([180, 255, 255], dtype="uint8"))
    ],

    "green": [
        (np.array([40, 50, 50], dtype="uint8"), np.array([80, 255, 255], dtype="uint8"))
    ],
    "yellow": [
        (np.array([20, 50, 50], dtype="uint8"), np.array([40, 255, 255], dtype="uint8"))
    ],
    "blue": [
        (np.array([100, 50, 50], dtype="uint8"), np.array([140, 255, 255], dtype="uint8"))
    ]
}


def close_screen_capture():
    # listen for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return True