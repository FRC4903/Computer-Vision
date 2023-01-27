# make it so that every time the space key is pressed, a new image is saved
# and the calibration is performed
import glob
import numpy as np
import cv2 as cv
import keyboard

from cscore import CameraServer

amt=0
frame = None
"""
def image():
	global amt, frame
	cv.imwrite(f'camera_calibration_{amt:0>3}.jpg', frame)
	amt+=1

def brake():
	pain=False
"""
cs = CameraServer
CameraServer.enableLogging()
camera = cs.startAutomaticCapture("apriltag camera", "/dev/video0")
camera.setResolution(640, 480)
sink = cs.getVideo()
output = cs.putVideo("calibration", 640, 480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# start a loop and show camera feed
pain=True
while pain:
    ret, frame = sink.grabFrame(np.zeros(shape=(480, 640, 3), dtype=np.uint8))
    output.putFrame(frame)
    # if space is pressed, save the image
    if keyboard.is_pressed('space'):
        cv.imwrite('camera_calibration_{amt}.jpg'.format(amt = amt), frame)
        amt+=1
    	#keyboard.hook_key('space', image, suppress=False)
    # if q is pressed, quit the program
    if keyboard.is_pressed('q'):
        break
	#keyboard.hook_key('q', brake, suppress=False)
