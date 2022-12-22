#experimental additional functionality to aid drivers with target finding and selection
from pupil_apriltags import Detector
import cv2
import json
from scipy.spatial.transform import Rotation
from PIL import Image
import numpy as np

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

while True:
    result, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(gray, estimate_tag_pose=True, tag_size=0.1524, camera_params=camera_params)
    if not detections:
        print("Nothing")
    else:
        for detect in detections:
            if detect.hamming == 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                smile = Image.open("images/target.jpg")
                back = Image.fromarray(img)
                corners = np.array(detect.corners)
                bottom_right = corners.max(axis=0)
                top_left = corners.min(axis=0)
                width = int(bottom_right[0] - top_left[0])
                height = int(bottom_right[1] - top_left[1])
                
                smile = smile.resize((width, height))
                rot_matrix = Rotation.from_matrix(detect.pose_R)
                euler = rot_matrix.as_euler('zxy', degrees=True)
                smile = smile.rotate(360 - euler[0])
                back.paste(smile, (int(top_left[0]), int(top_left[1])), smile)
                img = np.asarray(back)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('Result', img)
    if cv2.waitKey(1) == 13:
        break