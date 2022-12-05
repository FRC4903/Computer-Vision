#todo: provide more accurate readings by considering camera parameters found during calibration
#todo: figure out how to get orientation of the apriltag

import pupil_apriltags
import cv2
import numpy as np

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

def draw_pose(self,overlay, camera_params, tag_size, pose, z_sign=1):
    opoints = np.array([
        -2, -2, 0,
        2, -2, 0,
        2, 2, 0,
        2, -2, -4 * z_sign,
    ]).reshape(-1, 1, 3) * 0.5 * tag_size

    fx, fy, cx, cy = camera_params

    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)

    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]

    dcoeffs = np.zeros(5)

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)

    ipoints = np.round(ipoints).astype(int)

    ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

    cv2.line(overlay, ipoints[0], ipoints[1], (0,0,255), 2)
    cv2.line(overlay, ipoints[1], ipoints[2], (0,255,0), 2)
    cv2.line(overlay, ipoints[1], ipoints[3], (255,0,0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, 'X', ipoints[0], font, 0.5, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(overlay, 'Y', ipoints[2], font, 0.5, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(overlay, 'Z', ipoints[3], font, 0.5, (255,0,0), 2, cv2.LINE_AA)

def _draw_cube(overlay, camera_params, tag_size, pose,centroid, z_sign=1):

    opoints = np.array([
        -10, -8, 0,
        10, -8, 0,
        10, 8, 0,
        -10, 8, 0,
        -10, -8, 2 * z_sign,
        10, -8, 2 * z_sign,
        10, 8, 2 * z_sign,
        -10, 8, 2 * z_sign,
    ]).reshape(-1, 1, 3) * 0.5 * tag_size

    edges = np.array([
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        0, 4,
        1, 5,
        2, 6,
        3, 7,
        4, 5,
        5, 6,
        6, 7,
        7, 4
    ]).reshape(-1, 2)

    fx, fy, cx, cy = camera_params

    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)

    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]

    dcoeffs = np.zeros(5)

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)

    ipoints = np.round(ipoints).astype(int)

    ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

    for i, j in edges:
        cv2.line(overlay, ipoints[i], ipoints[j], (0, 255, 0), 1, 16)

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
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 3)

data = np.load("calibration_data/MultiMatrix.npz")

camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

fx = camMatrix[0][0]
fy = camMatrix[1][1]
cx = camMatrix[0][2]
cy = camMatrix[1][2]
camera_intrinsics_vector = [fx, fy, cx, cy]

detector = pupil_apriltags.Detector(
   families="tag36h11",
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
detections = detector.detect(grayimg, estimate_tag_pose=True,camera_params=camera_intrinsics_vector,tag_size=0.8125)
if not detections:
    print("Nothing")
else:
    for detect in detections:
        image = plotPoint(image, detect.center, CENTER_COLOR)
        image = plotText(image, detect.center, CENTER_COLOR, "id: "+str(detect.tag_id)+" dist: "+str(known_distance)+"ft")
        print(detect.corners)
        side_lengths=[]
        for corner in detect.corners:
            image = plotPoint(image, corner, CORNER_COLOR)
        for i in range(len(detect.corners)):
            side_lengths.append((abs(detect.corners[i][0]-detect.corners[(i+1)%len(detect.corners)][0])**2+abs(detect.corners[i][1]-detect.corners[(i+1)%len(detect.corners)][1])**2)**(1/2))
        print(side_lengths)
        side_lengths.sort(reverse=True)
        ref_image_width = side_lengths[0]


cv2.imshow('ref_image', image)

known_width = 0.8125

focal_length = (ref_image_width*known_distance)/known_width

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

looping = True

while looping:
    result, image = cam.read()
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(grayimg, estimate_tag_pose=True,camera_params=camera_intrinsics_vector,tag_size=0.8125)
    if not detections:
        print("Nothing")
    else:
        for detect in detections:
            #image = plotPoint(image, detect.center, CENTER_COLOR)
            print(detect.corners)
            side_lengths=[]
            _draw_cube(image, camera_intrinsics_vector, 0.8125, detect.pose_R, detect.center)

            '''for corner in detect.corners:
                image = plotPoint(image, corner, CORNER_COLOR)'''
            for i in range(len(detect.corners)):
                side_lengths.append((abs(detect.corners[i][0]-detect.corners[(i+1)%len(detect.corners)][0])**2+abs(detect.corners[i][1]-detect.corners[(i+1)%len(detect.corners)][1])**2)**(1/2))
            side_lengths.sort(reverse=True)
            print(side_lengths)
            distance = (known_width*focal_length)/side_lengths[0]
            #image = plotText(image, detect.center, CENTER_COLOR, "id: "+str(detect.tag_id)+" dist: "+str(round(distance,2))+"ft")



    cv2.imshow('Result', image)
    key = cv2.waitKey(100)
    #Terminate when Return Key is Hit
    if key == 13:
        looping = False

cv2.destroyAllWindows()
cv2.imwrite("final.png", image)