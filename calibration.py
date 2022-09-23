import os
import numpy as np
import cv2
from lib.videocapture import *

DEBUG = True

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

def process_calibration(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, (9, 6), corners2, ret)
        if DEBUG:
            cv2.imshow('Frame', frame)
            cv2.waitKey(25)
        output_video.write(frame)
    else:
        raise Exception("[ERROR] Fail to get frame from video.")

frames_to_process = 100

video_capture = VideoCapture('data/calibration.mp4')

total_frames = video_capture.video_cap_info.total_frames
video_width = video_capture.video_cap_info.video_width
video_height = video_capture.video_cap_info.video_height
fps = video_capture.video_cap_info.fps

if not os.path.exists("output"):
    os.makedirs("output")

output_video = cv2.VideoWriter('output/calibration_debug.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 2, (video_width, video_height))



print("[INFO] Frames to process: {}".format(frames_to_process))
print("[INFO] Frame count: {}".format(total_frames))
frame_offset = int(total_frames/total_frames)
print("[INFO] Frame offset: {}".format(frame_offset))

video_capture.process_video(process_calibration, frames_to_process)
first_frame = video_capture.get_first_frame()

# perform calibration
print("[INFO] Calibrating camera...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).shape[::-1], None, None)
print("[INFO] dist array: {}".format(dist))
print("[INFO K matrix: {}".format(mtx))
print("[INFO] Save data")

np.save("output/dist.npy", dist)
np.save("output/K.npy", mtx)
video_capture.cap.release()
output_video.release()
cv2.destroyAllWindows()

