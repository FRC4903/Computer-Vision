#AprilTag code I copied from the FIRST Workshop Awad showed in class
#https://www.youtube.com/watch?v=GglIQQm1RVA
#I'll probably use it later as a reference when we get the Pi up and running


'''import cv2
from imutils.video import VideoStream
import numpy as np
from numpy import savetxt
import imutils
import argparse
import datetime
import logging
import time
from networktables import NetworkTables


#Take arguments from the command line
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())


# initialize the video stream and allow the cammera sensor to warm up
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

#Initialize network tables - note that the IP address incorporates the team number in the second and third positions:
NetworkTables.initialize(server="10.61.62.2") #.61.62. : For team 6162
sd = NetworkTables.getTable("SmartDashboard")



def empty(a):
	pass


#This part, when uncommented, allows the user to use trackbars to adjust various parameters
# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("HUE_Min", "Trackbars", 28, 179, empty)
# cv2.createTrackbar("HUE_Max", "Trackbars", 87, 179, empty)
# cv2.createTrackbar("SAT_Min", "Trackbars", 3, 255, empty)
# cv2.createTrackbar("SAT_Max", "Trackbars",255, 255, empty)
# cv2.createTrackbar("VAL_Min", "Trackbars", 64, 255, empty)
# cv2.createTrackbar("VAL_Max", "Trackbars", 255, 255, empty)


def hexagonDetector(c):
	shape = ""
	perimeter = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)

	if len(approx) == 6:
		shape = "hexagon" 

	return shape



#Define the upper and lower colour bounds of the object being sought:
lower_green = np.array([89, 53, 55])
upper_green = np.array([129, 255, 255])

lower_yellow = np.array([24, 94, 183])
upper_yellow = np.array([83, 255, 255])


#Always repeat the following code:
while True:
    #Read a frame
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Extract colour information

    #Populate trackbars
	# h_min = cv2.getTrackbarPos("HUE_Min", "Trackbars")
	# h_max = cv2.getTrackbarPos("HUE_Max", "Trackbars")
	# s_min = cv2.getTrackbarPos("SAT_Min", "Trackbars")
	# s_max = cv2.getTrackbarPos("SAT_Max", "Trackbars")
	# v_min = cv2.getTrackbarPos("VAL_Min", "Trackbars")
	# v_max = cv2.getTrackbarPos("VAL_Max", "Trackbars")
    #Set lower and upper limits to green colour sought by the algorithm:
	# lower_green = np.array([h_min, s_min, v_min])
	# upper_green = np.array([h_max, s_max, v_max])

    #Process the image - green
	mask_green = cv2.inRange(hsv, lower_green, upper_green)
	mask_green = cv2.erode(mask_green, None, iterations=2)
	mask_green = cv2.dilate(mask_green, None, iterations=2)

    #Process the image - yellow
	mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
	mask_yellow = cv2.erode(mask_yellow, None, iterations=2)
	mask_yellow = cv2.dilate(mask_yellow, None, iterations=2)

    #Define contours for green and yellow
	contours_green = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, 
								cv2.CHAIN_APPROX_SIMPLE)
	contours_green = imutils.grab_contours(contours_green)

    #Process the contours to find particular shapes
	for c in contours_green:
		M = cv2.moments(c)

		area = cv2.contourArea(c)
		approx = cv2.approxPolyDP(c, 0.025*cv2.arcLength(c, True), True)
		x = approx.ravel()[0] #to put text on contour of object
		y = approx.ravel()[1]

		if area > 200: # only detect if object is large enough

			hX = int(M["m10"]/M["m00"]) # Coordinates of centroid
			hY = int(M["m01"]/M["m00"]) 
			coordinates = [hX, hY] #Y-axis is inverted (above-to-below)

			cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
            #Did we find a hexagon?
			if len(approx) == 6:
                ###############################
                # Trouble shooting code
				#logging.info("Target found!")
				#cv2.putText(frame, "Hexagon", (x, y), font, 1, (0,0,0))
				#file = open("coordinates.txt", "w+")
				#for n in coordinates:
				#	n = str(n)
				#	file.write("%s\n" % n)
				#file.close()
				#logging.info("Coordinates recorded")
                #################################

                #If yes, send coordinates to networktables:
				sd.putNumber("hX", hX)
				sd.putNumber("hY", hY)
				#logging.info("Coordinates recorded and sent.")

	contours_yellow = cv2.findContours(mask_yellow.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours_yellow = imutils.grab_contours(contours)
	#Did we find the ball?
	for c in contours_yellow:
		M = cv2.moments(c)

		area = cv2.contourArea(c)
		approx = cv2.approxPolyDP(c, 0.008*cv2.arcLength(c, True), True)
		x = approx.ravel()[0]
		y = approx.ravel()[1]
        #If yes (ie area is over "100" units and the number of "sides" is greater than 100 (a round object)
		if area > 100:
			cX = int(M["m10"]/M["m00"])
			cY = int(M["m01"]/M["m00"])

            #Then send the coordinates over network tables
			if len(approx) > 10:
				sd.putNumber("cX", cX)
				sd.putNumber("cY", cY)




#Finished?  Record log information and stop program
#This code is unreachable in the current state but may be adapted in other configurations of the program
logging.info("[{}] cleaning up".format(
	datetime.datetime.now()))

cv2.destroyAllWindows()	
vs.stop()'''