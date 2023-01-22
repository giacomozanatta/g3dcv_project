from lib.videocapture import *
import configs as conf
from lib.region import *
import cv2
import numpy as np

obj = "obj02"

# Parameters

central_point = conf.objects[obj]["center"]

padding = conf.objects[obj]["padding"]

video_capture = VideoCapture('data/obj02.mp4')
background_region = Region(0,1080,0,1000)

def p(frame):
    b_min = np.array([50,  20, 10],np.uint8)
    b_max = np.array([160,  110, 90],np.uint8)
    ## CREATE MASK ##
    # Set 
    mask = cv2.inRange(frame, b_min, b_max)[background_region.min_h:background_region.max_h, background_region.min_w:background_region.max_w]
    mask[conf.bgrem_yellow_rem.min_w:conf.bgrem_yellow_rem.max_w, conf.bgrem_yellow_rem.min_h:conf.bgrem_yellow_rem.max_h] = 255
    
    ## CLOSING: dilation + erosion ##

    # NOTE: not working well

    #dilate_kernel = np.ones((2, 2), np.uint8)
    #erode_kernel = np.ones((6, 6), np.uint8)
    #dilated_mask = cv2.dilate(mask, dilate_kernel, 1)
    #img_closing = cv2.erode(dilated_mask, erode_kernel, 1)

    #cv2.imshow('CLOSING', img_closing)

    
    ## BLUR and CANNY for Find Contours ##

    #blurred = cv2.GaussianBlur(mask, (5, 5), 21)
    #edged = cv2.Canny(blurred, 10, 50)

    #cv2.imshow("MASK", edged)

    

    # Make all pixels in mask black
    # NOTE: only for obj02 -> background between legs
    frame[background_region.min_h:background_region.max_h, background_region.min_w:background_region.max_w][mask>0] = [0,0,0]
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area  = cv2.contourArea(contour)
        if area > 30000:
            approx = cv2.approxPolyDP(contour,1,True)
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), -1)
    cv2.imshow("CONTOURS", frame)
    
    cv2.waitKey(25)

video_capture.process_video(p)

