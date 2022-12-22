from pupil_apriltags import Detector
import cv2
import json
from scipy.spatial.transform import Rotation
import math

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

known_distance = 60

image = cv2.imread("images/Ref_image_apriltag.jpg")
grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detections = detector.detect(grayimg)
if not detections:
    print("Nothing")
else:
    for detect in detections:
        if detect.hamming == 0:
            side_lengths=[]
            for i in range(len(detect.corners)):
                side_lengths.append((abs(detect.corners[i][0]-detect.corners[(i+1)%len(detect.corners)][0])**2+abs(detect.corners[i][1]-detect.corners[(i+1)%len(detect.corners)][1])**2)**(1/2))
            side_lengths.sort(reverse=True)
            ref_image_width = side_lengths[0]

known_width = 0.8125

focal_length = (ref_image_width*known_distance)/known_width

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
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
                img = plotPoint(img, detect.center, CENTER_COLOR)
                img = plotText(img, detect.center, CENTER_COLOR, detect.tag_id)
                for corner in detect.corners:
                    img = plotPoint(img, corner, CORNER_COLOR)
                side_lengths=[]
                for i in range(len(detect.corners)):
                    side_lengths.append((abs(detect.corners[i][0]-detect.corners[(i+1)%len(detect.corners)][0])**2+abs(detect.corners[i][1]-detect.corners[(i+1)%len(detect.corners)][1])**2)**(1/2))
                side_lengths.sort(reverse=True)
                distance = (known_width*focal_length)/side_lengths[0]
                distance += math.cos((euler[1]*math.pi*2/360)*known_width/2) + math.cos((euler[2]*math.pi*2/360)*known_width/2)
                vert_px = detect.center[1] - img.shape[1]
                lat_px = detect.center[0] - img.shape[0]
                vert_inches = (ref_image_width*known_distance)/vert_px
                lat_inches = (ref_image_width*known_distance)/lat_px
                vert_offset = math.arcsin(vert_inches/distance)/math.pi/2*360
                lat_offset = math.arcsin(lat_inches/distance)/math.pi/2*360
                camera_angle = [lat_offset,vert_offset]
                print("[TAG_ORIENTATION: " + str(euler) + ", \nDISTANCE: " + str(distance) + ", \nCAMERA_ANGLE: "+ str(camera_angle) +"]\n")

    cv2.imshow('Result', img)
    if cv2.waitKey(1) == 13:
        break