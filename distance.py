from pupil_apriltags import Detector
import cv2
import numpy as np
import math

detector = Detector(
    families="tag16h5",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
FOCAL_LENGTH = 702.098333 # result will be in inches
PAPER_WIDTH = 6 # in inches
INCHES_TO_METERS = 1/39.37

def find_distance(pixel_width):
    """
    :return: distance in meters
    """
    # (actual_width * focal_length) / pixels = distance
    return ((PAPER_WIDTH * FOCAL_LENGTH) / pixel_width) * INCHES_TO_METERS


while True:
    result, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(gray)
    if not detections:
        print("Nothing")
    else:
        for detect in detections:
            if detect.hamming == 0:
                pt1, pt2, _, _ = detect.corners
                width = math.dist(pt1, pt2)
                distance = find_distance(width)
                img = cv2.putText(img, str(distance), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('Result', img)
    if cv2.waitKey(1) == 13:
        break