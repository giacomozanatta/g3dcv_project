from lib.videocapture import *
import configs as conf
from lib.region import *
import cv2
import numpy as np
import math
from marker import *
import time
obj = "obj01"

central_point = conf.objects[obj]["center"]

padding = conf.objects[obj]["padding"]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
axis = np.float32([[0,0,0], [100,0,0],[0,100,0], [0,0,100]]).reshape(-1,3)

dist = np.load("data/processed/dist.npy")
K = np.load("data/processed/K.npy")

def draw_corner(frame, imgPts):
    cv2.circle(frame, np.array(imgPts[0][0][0], dtype=np.int32), 4, (0,255,0), 4)
    cv2.circle(frame, np.array(imgPts[0][1][0], dtype=np.int32), 4, (0,255,0), 4)
    cv2.circle(frame, np.array(imgPts[0][2][0], dtype=np.int32), 4, (0,255,0), 4)
    cv2.circle(frame, np.array(imgPts[0][3][0], dtype=np.int32), 4, (0,255,0), 4)
    cv2.circle(frame, np.array(imgPts[0][4][0], dtype=np.int32), 4, (0,255,0), 4)

def get_marker_number(th_img, points):
    out = 0
    for i in range(5):
            # white (255) is 0, black (0) is 1
            if th_img[int(points[0][i][0][1]), int(points[0][i][0][0])] == 0:
                out += int(math.pow(2, i))
    return out
def is_point_in_array(point, array):
    for p in array:
        if np.array_equal(p[0], point[0]):
            return True
    return False

def draw(img, corners, imgpts):
    corner = corners[0].ravel().astype(int)
    img = cv2.line(img, corner, imgpts[0].ravel().astype(int), (255,0,0), 5)
    img = cv2.line(img, corner, imgpts[1].ravel().astype(int), (0,255,0), 5)
    img = cv2.line(img, corner, imgpts[2].ravel().astype(int), (0,0,255), 5)
    return img

def test_region_object(frame, conf, obj):
    central_point = conf.objects[obj]["center"]

    padding = conf.objects[obj]["padding"]
    #frame[object_region.min_h:object_region.max_h, object_region.min_w:object_region.max_w] = [255,255,255]
    #cv2.rectangle(frame,(object_region.min_h,object_region.min_w),(object_region.max_h,object_region.max_w),(0,255,0),3)

    cv2.rectangle(frame,(central_point.x-padding,central_point.y-padding),(central_point.x+padding,central_point.y+padding),(0,0,255),3)
    cv2.circle(frame,(central_point.x,central_point.y), 10, (0,0,255), -1)

video_capture = VideoCapture(conf.objects[obj]["input_video"])
frame = video_capture.get_frame(205)

background_region = Region(0,1080,0,1000)



def project_voxels(frame, conf, obj):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(frame,(5,5),0)
    b_min = np.array([50,  20, 10],np.uint8)
    b_max = np.array([160,  102, 90],np.uint8)
    print(gray[background_region.min_h:background_region.max_h, background_region.min_w:background_region.max_w])
    # Make all pixels in mask white
    gray[background_region.min_h:background_region.max_h, background_region.min_w:background_region.max_w] = 0
    otsu,th3 = cv2.threshold(gray, 160,255,cv2.THRESH_BINARY)
    cv2.imshow('region_obj', th3)
    contours, hierarchies = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        #print(area)
        #draw contours for debugging

        #cv2.cornerSubPix()
        winSize = (5, 5)
        zeroZone = (1, 1)
        corners = cv2.cornerSubPix(th3, np.float32(approx), winSize, zeroZone, criteria)

        # order the corners (find the first corner)
        hull = cv2.convexHull(approx)
        for j, point in enumerate(approx):
            if not is_point_in_array(point, hull):
                corners = np.roll(corners, approx.shape[0]-j, axis=0)
                break

        cv2.drawContours(frame, [approx], 0, (100, 100, 0), -1)
        cv2.drawContours(frame, [hull], 0, (100, 0, 100), 3)
        rtval, rvec, tvec = cv2.solvePnP(Marker_0, corners, K, np.array([]), flags = cv2.SOLVEPNP_IPPE);
        #M = cv2.getPerspectiveTransform(np.array(np.squeeze(hull),dtype=np.float32), np.array([
                        #[(70, 0)],     # A
                        #(65, 5),     # B
                        #(98, 5),     # C
                        #(98, -5),    # D
                        #(65, -5),    # E
                    #], dtype=np.float32))
        imgPts = cv2.projectPoints(Marker_circles_0, rvec, tvec, K, np.array([]))



        # FIND MARKER NUMBER
        mk_number = get_marker_number(th3, imgPts)
        
        if not len(IMG_PTS):
            IMG_PTS = np.array(corners, copy=True)
        else:
            IMG_PTS = np.append(IMG_PTS, corners, axis=0)
        if not len(OBJ_PTS):
            OBJ_PTS = np.array(get_marker_position(mk_number), copy=True)
        else:
            OBJ_PTS = np.append(OBJ_PTS, get_marker_position(mk_number), axis=0)
        #imgPts = cv2.projectPoints(get_marker_position(3), rvec, tvec, K, dist)
        #cv2.drawContours(frame, np.array([imgPts[0]], dtype=np.int32), 0, (255, 100, 100), 6)
        
        #for i in range(24):
        #    imgPts = cv2.projectPoints(get_marker_position(i), rvec, tvec, K, np.array([]))
        #    cv2.drawContours(frame, np.array([imgPts[0]], dtype=np.int32), 0, (100, 0, 255), 4)
        print(mk_number)
        #cv2.line(frame, np.array(imgPtd[0][0][0], dtype=np.int32), np.array(imgPtd[0][1][0], dtype=np.int32), (255,0,0), 5)
        #cv2.line(frame, np.array(imgPtd[0][0][0], dtype=np.int32), np.array(imgPtd[0][2][0], dtype=np.int32), (0,255,0), 5)
        #cv2.line(frame, np.array(imgPtd[0][0][0], dtype=np.int32), np.array(imgPtd[0][3][0], dtype=np.int32), (0,0,255), 5)

        #cv2.line(frame, imgPtd[0][1][0], imgPtd[0][2][0], (0,255,0), 5)
        #cv2.line(frame, imgPtd[0][0][0], imgPtd[0][2][0], (0,0,255), 5)
        #imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K, dist)
        #img = draw(frame,imagePoints,imgpts)
        #cv2.imshow('img',img)
        #cv2.imshow('CIAO', cv2.warpPerspective(frame, M, dsize=(1000,1000)))
        #H = np.matrix( cv2.findHomography( corners, Marker_0, 0 )[0] )
        #4 if th3[int(corners[0][0][1]), int(corners[0][0][0])] == 0 else -1

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
        
        #aux = np.ascontiguousarray( np.expand_dims( np.transpose([contour]),axis=1), dtype=np.float32  )
        #aux = cv2.cornerSubPix( img, aux, (self.refineCornerHWinsize,self.refineCornerHWinsize), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.001) )

        #self.corners = np.matrix( np.transpose( np.squeeze(aux) ) )
        
        # Using cv2.putText() method
        #cv2.putText(frame, 'A', np.int32(corners[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2, cv2.LINE_AA)
        #cv2.putText(frame, 'B', np.int32(corners[1][0]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 2, cv2.LINE_AA)
        #cv2.putText(frame, 'C', np.int32(corners[2][0]), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 2, cv2.LINE_AA)
        #cv2.putText(frame, 'D', np.int32(corners[3][0]), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2, cv2.LINE_AA)
        #cv2.putText(frame, 'E', np.int32(corners[4][0]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2, cv2.LINE_AA)
        #order_corners(corners)

        
        #hierarchy = hierarchies[0][i]
        #child = hierarchy[2]
        #children_counter = 0


        #contour_counter = 0
        '''if child == -1:
            continue
        while(child != -1):
            internal_contour = contours[child]
            internal_approx = cv2.approxPolyDP(internal_contour, 0.01 * cv2.arcLength(internal_contour, True), True)
    
            internal_area  = cv2.contourArea(internal_contour)
            if len(internal_approx) > 8 and (internal_area > 40 and internal_area < 400):
                print("[INFO] Area of contour -> {}".format(internal_area))
                children_counter+=1
                cv2.drawContours(frame, [internal_contour], 0, (0, 200, 200), 10)
            hierarchy = hierarchies[0][child]
            child = hierarchy[0]
        if children_counter == 5:
            print(contour)
            print(contour.shape)
            print("[INFO] Found the 0!")
            #corners2 = cv2.cornerSubPix(gray, contours, (11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            contour2  = np.squeeze(contour)
            n,m = contour2.shape
            #cx = np.zeros((n,1))
            #cNew = np.hstack((contour2,cx))
            objp = np.zeros((n,3), np.float32)
            ret,rvecs, tvecs = cv2.solvePnP(objp, imagePoints, K, dist)
            # project 3D points to an image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K, dist)
            img = draw(frame,imagePoints,imgpts)
            cv2.imshow('img',img)

            break
            #M = cv2.moments(contour)
            #cX = int(M["m10"] / M["m00"])
            #cY = int(M["m01"] / M["m00"])
            #cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)'''
    #time.sleep(2)
    rtval, rvec, tvec = cv2.solvePnP(OBJ_PTS, IMG_PTS, K, np.array([]), flags = cv2.SOLVEPNP_IPPE);

    #imgPts = cv2.projectPoints(get_marker_position(3), rvec, tvec, K, dist)
    imgpts = cv2.projectPoints(axis, rvec, tvec, K, np.array([]))
    cv2.line(frame, np.array(imgpts[0][0][0], dtype=np.int32), np.array(imgpts[0][1][0], dtype=np.int32), (255,0,0), 5)
    cv2.line(frame, np.array(imgpts[0][0][0], dtype=np.int32), np.array(imgpts[0][2][0], dtype=np.int32), (0,255,0), 5)
    cv2.line(frame, np.array(imgpts[0][0][0], dtype=np.int32), np.array(imgpts[0][3][0], dtype=np.int32), (0,0,255), 5)

    for i in range(24):
        imgPts = cv2.projectPoints(get_marker_position(i), rvec, tvec, K, np.array([]))
        cv2.drawContours(frame, np.array([imgPts[0]], dtype=np.int32), 0, (0, 255, 0, 100), 2)
        cv2.putText(frame, str(i), np.int32(imgPts[0][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)




keyPoints = np.array([[ 279.03286469,  139.80463604],
                     [ 465.40665724,  136.70519839],
                     [ 465.40665724,  325.1505936],
                     [ 279.03286469,  230.927896]])

#print(objp.shape)

a = np.array([[1.,2.,3.], [4.,5.,6.], [1.,2.,3.], [4.,5.,6.],[1.,2.,3.], [4.,5.,6.],[1.,2.,3.], [4.,5.,6.]])
n,m = a.shape # for generality
a0 = np.zeros((n,1))
aNew = np.hstack((a,a0))
#print(a.shape) 
# shape of a = (8,3)
#c = a[0]
#c = np.array([1,2])
#print(c.shape)
#c.add(0)

#print(c.shape)
#b = np.reshape(a, (8, 3, 1)) 
# changing the shape, -1 means any number which is suitable

#print(b.shape) 
#print(a)
#print(b)
# size of b = (8,3,1)

xxx = np.array([[[1221,137]],

 [[1221,136]],

 [[1221,136]],

 [[1231,139]],

 [[1231,140]],

 [[1221,141]],

 [[1221,146]],

 [[1221,145]],

 [[1231,144]],

 [[1231,145]],

 [[1231,147]],

 [[1231,148]],

 [[1231,150]],

 [[1231,151]],

 [[1231,158]],

 [[1231,159]],

 [[1231,161]],

 [[1231,161]],

 [[1231,162]],

 [[1231,166]],

 [[1231,167]],

 [[1231,168]],

 [[1231,169]],

 [[1231,174]],

 [[1231,176]],

 [[1231,179]],

 [[1231,180]],

 [[1231,181]],

 [[1231,182]],

 [[1231,186]],

 [[1231,187]],

 [[1231,193]],

 [[1231,194]],

 [[1231,195]],

 [[1241,195]],

 [[1241,196]],

 [[1241,197]],

 [[1241,199]],

 [[1231,100]],

 [[1241,101]],

 [[1241,104]],

 [[1241,105]],

 [[1241,108]],

 [[1241,109]],

 [[1241,111]],

 [[1241,112]],

 [[1241,115]],

 [[1241,116]],

 [[1241,126]],

 [[1241,127]],

 [[1241,131]],

 [[1241,131]],

 [[1241,132]],

 [[1241,133]],

 [[1241,134]],

 [[1241,137]],

 [[1241,139]],

 [[1241,140]],

 [[1241,142]],

 [[1241,143]],

 [[1241,149]],

 [[1241,150]],

 [[1241,155]],

 [[1251,156]],

 [[1251,161]],

 [[1241,162]],

 [[1241,162]],

 [[1241,161]],

 [[1241,160]],

 [[1241,160]],

 [[1241,159]],

 [[1241,159]],

 [[1241,158]],

 [[1241,158]],

 [[1241,157]],

 [[1231,157]],

 [[1231,160]],

 [[1231,161]],

 [[1231,167]],

 [[1231,168]],

 [[1231,179]],

 [[1231,180]],

 [[1231,183]],

 [[1231,184]],

 [[1231,187]],

 [[1231,188]],

 [[1231,186]],

 [[1231,183]],

 [[1231,182]],

 [[1231,173]],

 [[1231,172]],

 [[1231,170]],

 [[1221,169]],

 [[1221,162]],

 [[1221,161]],

 [[1221,158]],

 [[1221,157]],

 [[1221,152]],

 [[1221,151]],

 [[1221,146]],

 [[1221,145]],

 [[1221,141]],

 [[1221,140]],

 [[1221,139]],

 [[1221,138]],

 [[1221,136]],

 [[1221,135]],

 [[1221,125]],

 [[1221,124]],

 [[1221,119]],

 [[1221,118]],

 [[1221,116]],

 [[1221,115]],

 [[1221,111]],

 [[1211,110]],

 [[1211,105]],

 [[1211,104]],

 [[1211,103]],

 [[1211,102]],

 [[1211,195]],

 [[1211,194]],

 [[1211,190]],

 [[1211,189]],

 [[1211,185]],

 [[1211,184]],

 [[1211,181]],

 [[1211,180]],

 [[1211,177]],

 [[1211,176]],

 [[1211,170]],

 [[1211,169]],

 [[1211,167]],

 [[1211,166]],

 [[1211,165]],

 [[1211,164]],

 [[1211,163]],

 [[1211,162]],

 [[1211,160]],

 [[1211,157]],

 [[1211,155]],

 [[1211,154]],

 [[1211,153]],

 [[1211,153]],

 [[1211,151]],

 [[1211,150]],

 [[1211,148]],

 [[1221,145]],

 [[1221,144]],

 [[1221,143]],

 [[1221,141]],

 [[1221,140]],

 [[1221,139]],

 [[1221,138]]])
#(153, 1, 2)
print(xxx.shape)
xxx = np.squeeze(xxx)

n,m = xxx.shape # for generality
a0 = np.zeros((n,1))
aNew = np.hstack((xxx,a0))
print(aNew)
print(aNew.shape)