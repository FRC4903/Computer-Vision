from pupil_apriltags import Detector
import cv2
import json
from scipy.spatial.transform import Rotation

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))
    image = cv2.line(image,
                     (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]),
                     color,
                     3)
    image = cv2.line(image,
                     (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH),
                     color,
                     3)
    return image

def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

detector = Detector(
    families="tag16h5",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)
cam = cv2.VideoCapture(0)
with open("images/calibration/matrices.json") as f:
    data = json.load(f)
    camera_matrix = data["camera_matrix"]["data"]
camera_params = [camera_matrix[0], camera_matrix[4], camera_matrix[2], camera_matrix[5]]
print(camera_params)

while True:
    result, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(gray, estimate_tag_pose=True, tag_size=0.1524, camera_params=camera_params)
    if not detections:
        print("Nothing")
    else:
        for detect in detections:
            if detect.hamming == 0:
                rot_matrix = Rotation.from_matrix(detect.pose_R)
                euler = rot_matrix.as_euler('zxy', degrees=True)
                print(euler)
                img = plotPoint(img, detect.center, CENTER_COLOR)
                img = plotText(img, detect.center, CENTER_COLOR, detect.tag_id)
                for corner in detect.corners:
                    img = plotPoint(img, corner, CORNER_COLOR)

    cv2.imshow('Result', img)
    if cv2.waitKey(1) == 13:
        break