import numpy as np
import cv2
from lib.videocapture import *

dist = np.load("output/dist.npy")
K = np.load("output/K.npy")


def undistort_frame(frame):
    h,  w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(frame, K, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('aftercalibration.png', dst)
    return dst

def thresholding(frame):
    print(frame)

def background_removal(frame):
    print(frame)



def carving_function(frame):
    # undistort camera
    cv2.imwrite('beforecalibration.png', frame)
    undistort_frame(frame)


video_capture = VideoCapture('data/obj01.mp4')
video_capture.process_video(carving_function)
# STEP 1: open the video (take the filename as parameter)

#STEP 2: Background removal

#STEP 3: Thresholding and find contours
# Otsu's thresholding after Gaussian filtering
#blur = cv2.GaussianBlur(img,(5,5),0)
#ret3,th3 = cv2.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#STEP 4: Solve Pnp

#STEP 5: Create CUBE with set of voxels

#STEP 6: Space Carving

#STEP 7: colouring
