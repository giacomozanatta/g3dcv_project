import os
import numpy as np
import cv2
from lib.videocapture import *
import configs
from util import *

# termination criteria: convergent to 0.001, max iter 30
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
# Needed by calibration function
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

def process_calibration(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find the position of the internal corners of the chessboard.
    # (9,6) -> dimension of the chessboard  
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # cornersubpix: find an accurated location for corner.
        # winsize (11,11) half of search size ('padding' from corner)
        # (just for debug purpose)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, (9, 6), corners2, ret)
        if configs.DEBUG:
            cv2.imshow('Frame', frame)
            cv2.waitKey(25)
        output_video.write(frame)


########## LOAD VIDEO ##########
video_capture = VideoCapture(configs.calibration["input_video"])

if not os.path.exists("output"):
    os.makedirs("output")

output_video = cv2.VideoWriter('output/calibration_debug.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 2, (video_capture.video_cap_info.video_width, video_capture.video_cap_info.video_height))


########## PRINT INFOS ##########
print("[INFO] Frames to process: {}".format(configs.calibration["frames_to_process"]))
print("[INFO] Frame count: {}".format(video_capture.video_cap_info.total_frames))
frame_offset = int(video_capture.video_cap_info.total_frames/configs.calibration["frames_to_process"])
print("[INFO] Frame offset: {}".format(frame_offset))

#################### CALIBRATE CAMERA ####################
video_capture.process_video(process_calibration, configs.calibration["frames_to_process"])
first_frame = video_capture.get_first_frame()
print("[INFO] Calibrating camera...")
# calibrate camera in order to retrieve intrinsic and extrinsic parameters of camera.
# we are interested in: distortion coefficient, K matrix (camera matrix)
# dist coeff permits to remove radial distortion (straight lines view as a curve), tangent distortion (when lense is not aligned perfectly parallel to the image plane.)
# rvecs, tvecs -> extrinsic parameters -> rotation and translation vectors -> translates 3D point to a 2D point.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).shape[::-1], None, None)
print("[INFO] dist array: {}".format(dist))
print("[INFO K matrix: {}".format(mtx))
print("[INFO] Save data")

#################### SAVE DATA FOR NEXT STEPS ####################
np.save("output/dist.npy", dist)
np.save("output/K.npy", mtx)
video_capture.cap.release()
output_video.release()
cv2.destroyAllWindows()