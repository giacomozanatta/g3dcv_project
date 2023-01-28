from lib.videocapture import *
import configs as conf
from lib.region import *
import cv2
import numpy as np
import math
from marker import *
import time
from lib.voxel import *
from lib.point import *
obj = "obj01"
 
central_point = conf.objects[obj]["center"]

padding = conf.objects[obj]["padding"]
voxel_set = VoxelSet(Point3D(0,0, 140), 60, 100)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
axis = np.float32([[0,0,0], [50,0,0],[0,50,0], [0,0,50]]).reshape(-1,3)

dist = np.load("data/processed/dist.npy")
K = np.load("data/processed/K.npy")


def save_ply(name, data):
    offset = data.offset
    with open('output/' + name + '.ply', 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment made by Giacomo Zanatta\n')
        f.write('element vertex ' + str(len(data.set) * 8) + '\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('element face ' + str(len(data.set) * 6) +'\n')
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        for point in data.set:
            f.write(str(point[0]-(offset/2)) + ' ' + str(point[1]-(offset/2)) + ' ' + str(point[2]-(offset/2)) + ' 100 100 100\n')
            f.write(str(point[0]-(offset/2)) + ' ' + str(point[1]-(offset/2)) + ' ' + str(point[2]+(offset/2)) + ' 100 100 100\n')
            f.write(str(point[0]-(offset/2)) + ' ' + str(point[1]+(offset/2)) + ' ' + str(point[2]+(offset/2)) + ' 100 100 100\n')
            f.write(str(point[0]-(offset/2)) + ' ' + str(point[1]+(offset/2)) + ' ' + str(point[2]-(offset/2)) + ' 100 100 100\n')
            f.write(str(point[0]+(offset/2)) + ' ' + str(point[1]-(offset/2)) + ' ' + str(point[2]-(offset/2)) + ' 100 100 100\n')
            f.write(str(point[0]+(offset/2)) + ' ' + str(point[1]-(offset/2)) + ' ' + str(point[2]+(offset/2)) + ' 100 100 100\n')
            f.write(str(point[0]+(offset/2)) + ' ' + str(point[1]+(offset/2)) + ' ' + str(point[2]+(offset/2)) + ' 100 100 100\n')
            f.write(str(point[0]+(offset/2)) + ' ' + str(point[1]+(offset/2)) + ' ' + str(point[2]-(offset/2)) + ' 100 100 100\n')
        # SE OGNI 'CUBO' HA 8 NODI E 6 FACCE -> 
        #   numero di edge = len(data.set) * 8
        #   numero di facce = len(data.set) * 6
        for i, point in enumerate(data.set):
            f.write('4 ' + str(0+(8*i)) + ' ' + str(1+(8*i)) + ' ' + str(2+(8*i)) + ' ' + str((3+(8*i))) + '\n')
            f.write('4 ' + str(7+(8*i)) + ' ' + str(6+(8*i)) + ' ' + str(5+(8*i)) + ' ' + str((4+(8*i))) + '\n')
            f.write('4 ' + str(0+(8*i)) + ' ' + str(4+(8*i)) + ' ' + str(5+(8*i)) + ' ' + str((1+(8*i))) + '\n')
            f.write('4 ' + str(1+(8*i)) + ' ' + str(5+(8*i)) + ' ' + str(6+(8*i)) + ' ' + str((2+(8*i))) + '\n')
            f.write('4 ' + str(2+(8*i)) + ' ' + str(6+(8*i)) + ' ' + str(7+(8*i)) + ' ' + str((3+(8*i))) + '\n')
            f.write('4 ' + str(3+(8*i)) + ' ' + str(7+(8*i)) + ' ' + str(4+(8*i)) + ' ' + str((0+(8*i))) + '\n')
        f.close()

def draw_corner(frame, imgPts):
    cv2.circle(frame, np.array(imgPts[0][0][0], dtype=np.int32), 4, (0,255,0), 4)
    cv2.circle(frame, np.array(imgPts[0][1][0], dtype=np.int32), 4, (0,255,0), 4)
    cv2.circle(frame, np.array(imgPts[0][2][0], dtype=np.int32), 4, (0,255,0), 4)
    cv2.circle(frame, np.array(imgPts[0][3][0], dtype=np.int32), 4, (0,255,0), 4)

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
    gray_2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    for j in range(24):
        imgPts = cv2.projectPoints(get_marker_position(j), rvec, tvec, K, np.array([]))
        cv2.drawContours(frame, np.array([imgPts[0]], dtype=np.int32), 0, (0, 255, 0, 100), 2)
        cv2.putText(frame, str(j), np.int32(imgPts[0][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

    imgpts = cv2.projectPoints(np.array(voxel_set.set, dtype = np.double), rvec, tvec, K, np.array([]))
    #print(imgpts)
    l = 0
    for j in range(len(imgpts[0])):
        #print(gray_2[np.int32(imgpts[0][j][0][1]),np.int32(imgpts[0][j][0][0])])
        #print(imgpts[0][i][0][1])
        if gray_2[np.int32(imgpts[0][j][0][1]),np.int32(imgpts[0][j][0][0])] == 0:
            voxel_set.set.pop(l)
            l -= 1
        #cv2.line(frame, np.array(imgpts[0][i][0][0], dtype=np.int32), np.array(imgpts[0][1][0], dtype=np.int32), (255,0,0), 5)
        cv2.circle(frame, np.int32(imgpts[0][j][0]), 1, (0,0,255), -1)
        l += 1
    #cv2.imshow('GRAY', gray_2)
    save_ply(obj, voxel_set)