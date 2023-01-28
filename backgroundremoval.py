import cv2
import numpy as np
from lib.videocapture import *
from lib.region import *


b_min_label = np.array([100,  62, 40],np.uint8) # BGR notation
b_max_label = np.array([120,  80, 100],np.uint8) # BGR notation

frame_counter = 0
def background_removal(frame, conf):
    global frame_counter
    frame_counter+=1
    ## CREATE MASK ##
    # Create a mask in bg_region (set intensity to 255 (WHITE) if pixel in image is blue (between b_min and b_max))
    mask = cv2.inRange(frame, conf.b_min, conf.b_max)[conf.bg_region.min_h:conf.bg_region.max_h, conf.bg_region.min_w:conf.bg_region.max_w]
    if conf.working_object == 'obj01' and (frame_counter in range(295, 345) or frame_counter in range(680, 713)):
    #cv2.rectangle(frame,(750,600),(850,700),(0,255,0),3)
    
        mask_label = cv2.inRange(frame, b_min_label, b_max_label)[600:700, 750:850]
        mask[600:700, 750:850][mask_label==0] = 0
    #mask[conf.bgrem_yellow_rem.min_w:conf.bgrem_yellow_rem.max_w, conf.bgrem_yellow_rem.min_h:conf.bgrem_yellow_rem.max_h] = 255 # unused 

    cv2.imshow("BGREM_MASK", mask)
   
    ## CLOSING: dilation + erosion ##
    ## DILATION: bridging gaps
    mask = cv2.dilate(mask, conf.dilate_kernel_closing, 1)
    cv2.imshow('dilated_mask', mask)
    ## EROSION: reducing the size of obj (remove some border)
    mask = cv2.erode(mask, conf.erode_kernel_closing, 10)
    cv2.imshow('eroded_mask', mask)

    ## BLUR and CANNY for Find Contours ##
    ## NOTE: not working well
    #blurred = cv2.GaussianBlur(mask, (5, 5), 21)
    #edged = cv2.Canny(blurred, 10, 50)

    #cv2.imshow("MASK", edged)
    label_region = Region(1300,1329,1500,1588) # region to consider for background removal
    # NOTE: only for obj02 -> remove background between "legs" of dyno
    if conf.objects[conf.working_object]['enhanced_bg_removal']:
        # Make all pixels in frame (in the bg_region) that has mask > 0 to 0 (BLACK)
        frame[conf.bg_region.min_h:conf.bg_region.max_h, conf.bg_region.min_w:conf.bg_region.max_w][mask>0] = [0,0,0]
        cv2.imshow("BGREM_ENHANCED", frame)
    # finCountours -> try to find the contours of a shape from mask.
    # It uses Suzuki algorithm: border following algorithm. This algorithm try to define hierarchical relationships among the borders, differentiating also between outer boundary and hole boundary.
    # since we want to consider only the object, we use EXTERNAL MODE: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    # note: it wants a single channel image. Non zero pixels are threted as 1. 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area  = cv2.contourArea(contour)
        # the polygon extracted is ALL - OBJECT
        if area > 60000:
            approx = cv2.approxPolyDP(contour,1,True) # get polygon
            # HOW APPROXPOLYDP WORKS  (RAMER DOUGLAS PEUCKER)
            # it takes a curve and a costant E (distance dimension > 0). Output an approximated curve.
            # big epsilon -> more approximated
            # recursively divides the line.
            # first step: takes all point and marks first and last point as 'kept'. Define a line from the first point (S) and the last one (E).
            # the examine each of the intermediary point, calculate distance from the points to the line.
            # If the farthest point (Dmax) from the line is < E -> all other points are < E so we throw away all the intermediary points.
            # Otherwise: consider 2 line: L1, FROM S to Dmax and L2, from Dmax to E.
            # cal the algorithm on each of the two lines.
            frame_cp = frame.copy()
            cv2.drawContours(frame_cp, [approx], 0, (0, 0, 255), 4)
            cv2.imshow('contours', frame_cp)
            # Fill the contours with BLACK
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), -1)
    # KEEP THE LABEL FOR OBJ01



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