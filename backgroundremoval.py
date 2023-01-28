import cv2
import numpy as np
from lib.videocapture import *
from lib.region import *


def background_removal(frame, conf):
    b_min = np.array([50,  20, 10],np.uint8) # BGR notation
    b_max = np.array([180,  160, 90],np.uint8) # BGR notation
    ## CREATE MASK ##
    # Create a mask in bg_region (set intensity to 255 (WHITE) if pixel in image is blue (between b_min and b_max))
    mask = cv2.inRange(frame, b_min, b_max)[conf.bg_region.min_h:conf.bg_region.max_h, conf.bg_region.min_w:conf.bg_region.max_w]
    
    #mask[conf.bgrem_yellow_rem.min_w:conf.bgrem_yellow_rem.max_w, conf.bgrem_yellow_rem.min_h:conf.bgrem_yellow_rem.max_h] = 255 -- unused

    #cv2.imshow("BGREM_MASK", mask)
   
    ## CLOSING: dilation + erosion ##
    ## DILATION: bridging gaps
    dilated_mask = cv2.dilate(mask, conf.dilate_kernel_closing, 1)
    cv2.imshow('dilated_mask', dilated_mask)
    ## EROSION: reducing the size of obj (remove some border)
    mask_close = cv2.erode(dilated_mask, conf.erode_kernel_closing, 10)
    cv2.imshow('eroded_mask', dilated_mask)

    ## BLUR and CANNY for Find Contours ##
    ## NOTE: not working well
    #blurred = cv2.GaussianBlur(mask, (5, 5), 21)
    #edged = cv2.Canny(blurred, 10, 50)

    #cv2.imshow("MASK", edged)

    # NOTE: only for obj02 -> remove background between "legs" of dyno
    if conf.objects[conf.working_object]['enhanced_bg_removal']:
        # Make all pixels in frame (in the bg_region) that has mask > 0 to 0 (BLACK)
        frame[conf.bg_region.min_h:conf.bg_region.max_h, conf.bg_region.min_w:conf.bg_region.max_w][mask_close>0] = [0,0,0]
        cv2.imshow("BGREM_ENHANCED", frame)
    contours, _ = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area  = cv2.contourArea(contour)
        # the polygon extracted is ALL - OBJECT
        if area > 60000:
            approx = cv2.approxPolyDP(contour,1,True) # get polygon
            frame_cp = frame.copy()
            cv2.drawContours(frame_cp, [approx], 0, (0, 0, 255), 4)
            # Fill the contours with BLACK
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), -1)


# OLD VERSION OF BACKGROUND_REMOVAL
def background_removal_old_v1(frame, conf):
    
    b_min = np.array([50,  20, 10],np.uint8)
    b_max = np.array([160,  102, 90],np.uint8)

    mask = cv2.inRange(frame, b_min, b_max)[conf.bg_region.min_h:conf.bg_region.max_h, conf.bg_region.min_w:conf.bg_region.max_w]
    cv2.imshow(mask)
    # Make all pixels in mask black
    frame[conf.bg_region.min_h:conf.bg_region.max_h, conf.bg_region.min_w:conf.bg_region.max_w][mask>0] = [0,0,0]

    # cv2.waitKey(0)
    cv2.waitKey(25)