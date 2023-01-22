from ctypes.wintypes import MAX_PATH
import numpy as np
import cv2
from lib.videocapture import *
from backgroundremoval import *
from test import *

import configs as conf

from lib.point import *

dist = np.load("output/dist.npy")
K = np.load("output/K.npy")

central_point = Point(660,540)

object_id = "obj01"

padding = 320

object_region = Region(250,980,190,890)

def carving_process(frame):
    ## UNDISTORT FRAME ##
    # use dist and K generated from calibration to undistort the current frame.
    cv2.imshow('FRAME', frame)
    frame = undistort_frame(frame)
    cv2.imshow('FRAME_UNDISTORT', frame)
    ## BACKGROUND REMOVAL
    background_removal(frame, conf, object_id)
    # cv2.imshow('frame_bgrem', frame)
    cv2.imshow('FRAME_BGREM', frame)
    project_voxels(frame, conf, object_id)
    cv2.imshow('FRAME_PROJ_VOXEL', frame)
    
    cv2.waitKey(25)

def test_region_object(frame):
    #frame[object_region.min_h:object_region.max_h, object_region.min_w:object_region.max_w] = [255,255,255]
    #cv2.rectangle(frame,(object_region.min_h,object_region.min_w),(object_region.max_h,object_region.max_w),(0,255,0),3)

    cv2.rectangle(frame,(central_point.x-padding,central_point.y-padding),(central_point.x+padding,central_point.y+padding),(0,0,255),3)
    cv2.circle(frame,(central_point.x,central_point.y), 10, (0,0,255), -1)

def undistort_frame(frame):
    h,  w = frame.shape[:2]
    cv2.imwrite('beforecalibration.png', frame)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(frame, K, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('aftercalibration.png', dst)
    return dst


def carving_function(frame):
    ## UNDISTORT FRAME ##
    undistort_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,105, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = 255 - thresh

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    cv2.imshow('thresh', thresh)
    cv2.imshow('result', result)


video_capture = VideoCapture('data/' + object_id + '.mp4')
video_capture.process_video(carving_process)
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
