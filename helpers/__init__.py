
import cv2
import numpy as np


def is_circle(contour):
    center, radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    if area == 0:
        return None
    circle_area = 3.1415 * (radius ** 2)
    # This checks if the shape area is roughly a circle
    if 0.85 <= (circle_area / area) <= 1.15:
        return center, radius
    return None


def is_square_or_rectangle(contour):
    approx = cv2.approxPolyDP(
        contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        if 0.95 <= aspectRatio <= 1.05:  # square will have aspect ratio close to one
            return "square", approx
        else:  # else rectangle
            return "rectangle", approx
    return None


def is_triangle(contour):
    approx = cv2.approxPolyDP(
        contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 3:
        return approx  # Only return approx, since it's surely a triangle
    return None


SHAPES = {
    'circle': is_circle,
    'square': is_square_or_rectangle,
    'rectangle': is_square_or_rectangle,
    'triangle': is_triangle,
}

COLOURS = {
    # Red wraps around in HSV
    "red": [
        (np.array([0, 50, 50]), np.array([10, 255, 255])),  # low range
        (np.array([170, 50, 50]), np.array([180, 255, 255]))  # high range
    ],
    'green': [(np.array([35, 50, 20]), np.array([85, 255, 255]))],
    'blue': [(np.array([110, 50, 20]), np.array([130, 255, 255]))],
    'yellow': [(np.array([20, 50, 20]), np.array([30, 255, 255]))],
    # Add more colors as needed
}


def close_screen_capture():
    # listen for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return True
