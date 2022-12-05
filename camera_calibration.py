#todo: find camera parameters (cx, cy, fx ,fy)

import numpy as np
import cv2 as cv
import glob



#Init Variables

chessboardSize = (23,15)
frameSize = (1280,720)



criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('images\calibration\*.jpg')
print(images)
for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('img', img)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)


cv.destroyAllWindows()

# Calibration

print(objpoints, imgpoints)

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)


print("calibrated")

print("writing data to calibration_data/MultiMatrix.npz")
np.savez(
    "calibration_data/MultiMatrix",
    camMatrix=cameraMatrix,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)

print("-------------------------------------------")

print("loading data stored using numpy savez function\n \n \n")

data = np.load("calibration_data/MultiMatrix.npz")

camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

print(camMatrix)

print("loaded calibration data successfully")
