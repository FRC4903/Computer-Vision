import numpy as np
import cv2
from pupil_apriltags import Detector
from time import time

from cscore import CameraServer, VideoMode
from networktables import NetworkTablesInstance

ntinst = NetworkTablesInstance.getDefault()
ntinst.initialize(server='10.49.3.2')
ntinst.startClientTeam(4903)
ntinst.startDSClient()
nt = ntinst.getTable('SmartDashboard')

frame_height = 480
frame_width = 640

cs = CameraServer
camera = cs.startAutomaticCapture("apriltag camera", "/dev/video0")
camera.setResolution(frame_width, frame_height)
CameraServer.enableLogging()
#cs.setVideoMode(VideoMode.PixelFormat.kBGR, frame_width, frame_height, 15)

sink = cs.getVideo()
output = cs.putVideo("April Tags", frame_width, frame_height)

# Edit these variables for config.
camera_params = 'camera calibration/CameraCalibration.npz'

framerate = 30
output_overlay = True
undistort_frame = True
debug_mode = True
show_framerate = True

# Load camera parameters
with np.load(camera_params) as file:
    cameraMatrix, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix', 'dist', 'rvecs', 'tvecs')]

aprilCameraMatrix = [cameraMatrix[0][0], cameraMatrix[1][1], cameraMatrix[0][2], cameraMatrix[1][2]]

# options = DetectorOptions(families="tag36h11")
detector = Detector(
    families='tag16h5',
    nthreads=3,
    quad_decimate=2.0,
    quad_sigma=3.0,
    decode_sharpening=1.0,
    refine_edges=3,
)

# Read until video is completed
while True:
    # Capture frame-by-frame
    ret, frame = sink.grabFrameNoTimeout(np.zeros(shape=(frame_height, frame_width, 3), dtype=np.uint8))
    if ret != 0:
        start_time = time()

        inputImage = frame

        if undistort_frame:
            height, width = inputImage.shape[:2]
            newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
            inputImage = cv2.undistort(inputImage, cameraMatrix, dist, None, newCameraMatrix)

        image = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        if debug_mode:
            print("[INFO] detecting AprilTags...")
        results = detector.detect(image, estimate_tag_pose=True, camera_params=aprilCameraMatrix, tag_size=0.2032)

        # print(results)
        if debug_mode:
            print(f"[INFO] {len(results)} total AprilTags detected")
            print(f"[INFO] Looping over {len(results)} apriltags and getting data")

        nt.putNumber("x", -1)
        nt.putNumber("y", -1)

        # loop over the AprilTag detection results
        for r in results:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            if r.hamming == 0:
                (ptA, ptB, ptC, ptD) = r.corners
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))
                ptA = (int(ptA[0]), int(ptA[1]))
                # draw the bounding box of the AprilTag detection
                cv2.line(inputImage, ptA, ptB, (0, 255, 0), 2)
                cv2.line(inputImage, ptB, ptC, (0, 255, 0), 2)
                cv2.line(inputImage, ptC, ptD, (0, 255, 0), 2)
                cv2.line(inputImage, ptD, ptA, (0, 255, 0), 2)

                cv2.circle(inputImage, ptA, 4, (0, 0, 255), -1)
                cv2.circle(inputImage, ptB, 4, (0, 0, 255), -1)
                cv2.circle(inputImage, ptC, 4, (0, 0, 255), -1)
                cv2.circle(inputImage, ptD, 4, (0, 0, 255), -1)
                # draw the center (x, y)-coordinates of the AprilTag
                (cX, cY) = (int(r.center[0]), int(r.center[1]))
                cv2.circle(inputImage, (cX, cY), 5, (0, 0, 255), -1)
                # draw the tag family on the image
                tagFamily = r.tag_family.decode("utf-8")
                cv2.putText(inputImage, tagFamily, (ptD[0], ptD[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                x_centered = cX - frame_width / 2
                y_centered = -1 * (cY - frame_height / 2)

                nt.putNumber("x", x_centered)
                nt.putNumber("y", y_centered)

                cv2.putText(inputImage, f"Center X coord: {x_centered}", (ptB[0] + 10, ptB[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

                cv2.putText(inputImage, f"Center Y coord: {y_centered}", (ptB[0] + 10, ptB[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

                cv2.putText(inputImage, f"Tag ID: {r.tag_id}", (ptC[0] - 70, ptC[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

                cv2.circle(inputImage, (int((frame_width / 2)), int((frame_height / 2))), 5, (0, 0, 255), 2)

                poseRotation = r.pose_R
                poseTranslation = r.pose_t
                poseTranslation = [i*31.9541559133 for i in poseTranslation]

                if debug_mode:
                    print(f"[DATA] Detection rotation matrix:\n{poseRotation}")
                    print(f"[DATA] Detection translation matrix:\n{poseTranslation}")
                    # print(f"[DATA] Apriltag position:\n{}")

                # draw x y z axis
                # x axis
                cv2.line(inputImage, (cX, cY), (cX + int(poseRotation[0][0] * 100), cY + int(poseRotation[1][0] * 100)), (0, 0, 255), 2)
                # y axis
                cv2.line(inputImage, (cX, cY), (cX + int(poseRotation[0][1] * 100), cY + int(poseRotation[1][1] * 100)), (0, 255, 0), 2)
                # z axis
                cv2.line(inputImage, (cX, cY), (cX + int(poseRotation[0][2] * 100), cY + int(poseRotation[1][2] * 100)), (255, 0, 0), 2)

        if debug_mode:
            # show the output image after AprilTag detection
            print("[INFO] displaying image after overlay")

        if show_framerate:
            end_time = time()
            cv2.putText(inputImage, f"FPS: {1 / (end_time - start_time)}", (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        output.putFrame(inputImage)


    # Break the loop
    else:
        break
