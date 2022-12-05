import cv2 as cv
import argparse
import pupil_apriltags
from pupil_apriltags import Detector
import copy
import numpy as np


def main():


    at_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )






    vid = cv.VideoCapture(0)
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        debug_image = copy.deepcopy(frame)



        fx = 921.03031133
        fy = 917.5014192
        cx = 662.23719425
        cy = 378.27939133

        camera_intrinsics_vector = [fx, fy, cx, cy]





        #---Converts to grayscale and configs detector---#
        frame = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
                frame,
                estimate_tag_pose=True,
                camera_params=camera_intrinsics_vector,
                tag_size=0.13,
            )


        debug_image = draw_tags(debug_image, tags)
        
        



        # Display the resulting frame
        cv.imshow('frame', debug_image)


        
        # the 'q' button is set as the quitting button
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()





def draw_tags(
    image,
    tags,
):


    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners


        pose_R = tag.pose_R
        pose_t = tag.pose_t
        homography = tag.homography


        print(("Pose_R:") , pose_R)
        print("sin_x:" , pose_R[0][0])
        print("cos_x:" , pose_R[0][1])
        print("tan_x:" , pose_R[0][2])

        print("sin_y:" , pose_R[1][0])
        print("cos_y:" , pose_R[1][1])
        print("tan_y:" , pose_R[1][2])

        print("sin_z:" , pose_R[2][0])
        print("cos_z:" , pose_R[2][1])
        print("tan_z:" , pose_R[2][2])




        print("\n")



        print(("Pose_t:") , pose_t)
        print("x:" , pose_t[0][0])
        print("y:" , pose_t[1][0])
        print("z:" , pose_t[2][0])
        print("\n")


        print (("Homography:") , homography)


        print("\n")
        print("------------------")
        print("\n")
    







        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        cv.putText(image, str(tag_family) + ':' + str(tag_id), (corner_01[0], corner_01[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
        
        cv.rectangle(image, (0, 0), (400, 400), (0, 0, 0), -1)
        
        PCX = (int)(200.0 + (pose_t[0] * 100.0))
        PCY = (int)(200.0 + (pose_t[1] * 100.0))
        PCZ = (int)(pose_t[2] * 100.0)

        print(PCX)
        print(PCY)
        print(PCZ)
        
        cv.circle(image, (PCX, PCZ), 5, (255, 255, 255), -1)
        

    return image




if __name__ == '__main__':
    main()