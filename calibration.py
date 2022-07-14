import os
import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

frames_to_read = 100

cap = cv2.VideoCapture('data/calibration.mp4')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))



if not os.path.exists("output"):
    os.makedirs("output")

output_video = cv2.VideoWriter('output/calibration_debug.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 2, (video_width, video_height))

print("[INFO] Frames to read: {}".format(frames_to_read))
print("[INFO] Frame count: {}".format(total_frames))
frame_offset = int(total_frames/frames_to_read)
print("[INFO] Frame offset: {}".format(frame_offset))

for i in range(frames_to_read):
    frame_to_process = frame_offset*(i)
    print("[INFO] Processing frame {}/{}".format(frame_to_process+1, total_frames))
    # set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_process)
    # read the frame
    ret, frame = cap.read()
    if ret:
        # print the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(frame, (9, 6), corners2, ret)
            output_video.write(frame)
            #cv2.imshow('img', frame)
        #cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        print("[ERROR] Fail to get frame from video.")
cap.release()
output_video.release()
cv2.destroyAllWindows()
# perform calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) # mtx is k
print("[INFO] dist array: {}".format(dist))
print("[INFO K matrix: {}".format(mtx))
print("[INFO] Save data")

np.save("output/dist.npy", dist)
#dist_p = np.load("data/processed/dist.npy")
#print(dist_p)
np.save("output/K.npy", mtx)
