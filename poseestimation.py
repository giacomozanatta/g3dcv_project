import cv2
import numpy as np
from marker import *


def is_point_in_array(point, array):
    for p in array:
        if np.array_equal(p[0], point[0]):
            return True
    return False

def draw_corner(frame, imgPts):
    cv2.circle(frame, np.array(imgPts[0][0][0], dtype=np.int32), 4, (0,255,0), 4)
    cv2.circle(frame, np.array(imgPts[0][1][0], dtype=np.int32), 4, (0,255,0), 4)
    cv2.circle(frame, np.array(imgPts[0][2][0], dtype=np.int32), 4, (0,255,0), 4)
    cv2.circle(frame, np.array(imgPts[0][3][0], dtype=np.int32), 4, (0,255,0), 4)


def pose_estimation(frame, conf, K):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Make all pixels in mask white
    gray[conf.bg_region.min_h:conf.bg_region.max_h, conf.bg_region.min_w:conf.bg_region.max_w] = 0
    otsu,th = cv2.threshold(gray, 160,255,cv2.THRESH_BINARY)
    cv2.imshow('region_obj', th)
    contours, hierarchies = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("[INFO] Len contours -> {}".format(len(contours)))
    print("[INFO] Len hierarchies -> {}".format(len(hierarchies[0])))
    # list for storing names of shapes
    IMG_PTS = []
    OBJ_PTS = []
    for i, contour in enumerate(contours):

        approx = cv2.approxPolyDP(contour, conf.approx_poly_epsilon, True)
        area = cv2.contourArea(contour)

        # MARKER has 5 corners and it is not convex. Skip retrieved polygons that does not match these conditions.
        # skip also small polygons
        if approx.shape[0] != 5 or cv2.isContourConvex(approx) or area < 4200: 
            continue
        #draw contours for debugging
        winSize = (5, 5)
        zeroZone = (1, 1)
        corners = cv2.cornerSubPix(th, np.float32(approx), winSize, zeroZone, conf.criteria)

        # order the corners (find the first corner)
        hull = cv2.convexHull(approx)
        for j, point in enumerate(approx):
            if not is_point_in_array(point, hull):
                corners = np.roll(corners, approx.shape[0]-j, axis=0)
                break

        cv2.drawContours(frame, [approx], 0, (100, 100, 0), -1)
        cv2.drawContours(frame, [hull], 0, (100, 0, 100), 3)
        rtval, rvec, tvec = cv2.solvePnP(Marker_0, corners, K, np.array([]), flags = cv2.SOLVEPNP_IPPE);
        imgPts = cv2.projectPoints(Marker_circles_0, rvec, tvec, K, np.array([]))



        # FIND MARKER NUMBER
        mk_number = get_marker_number(th, imgPts)
        
        if not len(IMG_PTS):
            IMG_PTS = np.array(corners, copy=True)
        else:
            IMG_PTS = np.append(IMG_PTS, corners, axis=0)
        if not len(OBJ_PTS):
            OBJ_PTS = np.array(get_marker_position(mk_number), copy=True)
        else:
            OBJ_PTS = np.append(OBJ_PTS, get_marker_position(mk_number), axis=0)
        
        cv2.circle(frame, np.int32(corners[0][0]), 5, (0,0,255), 4)
        cv2.circle(frame, np.int32(corners[1][0]), 5, (0,255,255), 4)
        cv2.circle(frame, np.int32(corners[2][0]), 5, (255,0,255), 4)
        cv2.circle(frame, np.int32(corners[3][0]), 5, (255,0,0), 4)
        cv2.circle(frame, np.int32(corners[4][0]), 5, (0,255,0), 4)
        # draw the corners:
        # A -> RED
        # B -> YELLOW
        # C -> MAGENTA
        # D -> BLUE
        # E -> GREEN
        draw_corner(frame, imgPts)

    rtval, rvec, tvec = cv2.solvePnP(OBJ_PTS, IMG_PTS, K, np.array([]), flags = cv2.SOLVEPNP_IPPE);

    #imgPts = cv2.projectPoints(get_marker_position(3), rvec, tvec, K, dist)
    imgpts = cv2.projectPoints(conf.axis, rvec, tvec, K, np.array([]))
    cv2.line(frame, np.array(imgpts[0][0][0], dtype=np.int32), np.array(imgpts[0][1][0], dtype=np.int32), (255,0,0), 5)
    cv2.line(frame, np.array(imgpts[0][0][0], dtype=np.int32), np.array(imgpts[0][2][0], dtype=np.int32), (0,255,0), 5)
    cv2.line(frame, np.array(imgpts[0][0][0], dtype=np.int32), np.array(imgpts[0][3][0], dtype=np.int32), (0,0,255), 5)
    for j in range(24):
        imgPts = cv2.projectPoints(get_marker_position(j), rvec, tvec, K, np.array([]))
        cv2.drawContours(frame, np.array([imgPts[0]], dtype=np.int32), 0, (0, 255, 0, 100), 2)
        cv2.putText(frame, str(j), np.int32(imgPts[0][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    cv2.imshow('pose est', frame)
    return rvec, tvec

