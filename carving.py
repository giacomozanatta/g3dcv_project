import numpy as np
import cv2
from lib.videocapture import *
from backgroundremoval import *
from poseestimation import *
from undistort import *
from test import *

import configs as conf

from lib.point import *

import pickle

dist = np.load("output/dist.npy")
K = np.load("output/K.npy")

central_point = Point(660,540)


voxel_set = VoxelSet(configs.voxel_set_center, configs.objects[configs.working_object]['voxel_set_padding'], configs.objects[configs.working_object]['voxel_set_N'])

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

def carving_process(frame):
    global voxel_set
    ## UNDISTORT FRAME ##
    # use dist and K generated from calibration to undistort the current frame.
    cv2.imshow('FRAME', frame)
    frame = undistort_frame(frame, K, dist)
    cv2.imshow('FRAME_UNDISTORT', frame)
    ## BACKGROUND REMOVAL
    background_removal(frame, conf)
    # cv2.imshow('frame_bgrem', frame)
    cv2.imshow('FRAME_BGREM', frame)
    # POSE ESTIMATION
    old_frame = frame.copy()
    rvec, tvec = pose_estimation(frame, conf, K)
    cv2.imshow('FRAME_POSEEST', frame)
    #voxel_set = carving_function(frame, conf, K, rvec, tvec, voxel_set)
    #project_voxels(frame, conf, object_id)
    cv2.imshow('FRAME_VOXELS', frame)
    cv2.waitKey(25)

def carving_function(frame, conf, K, rvec, tvec, voxel_set):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgpts = cv2.projectPoints(np.array(voxel_set.set, dtype = np.double), rvec, tvec, K, np.array([]))
    l = 0
    for j in range(len(imgpts[0])):
        if gray[np.int32(imgpts[0][j][0][1]),np.int32(imgpts[0][j][0][0])] == 0:
            voxel_set.set.pop(l)
            l -= 1
        cv2.circle(frame, np.int32(imgpts[0][j][0]), 1, (0,0,255), -1)
        l += 1
    return voxel_set

video_capture = VideoCapture('data/' + configs.working_object + '.mp4')
video_capture.process_video(carving_process)
save_ply(conf.working_object, voxel_set)
# save voxel_set
with open('output/voxels_' + conf.working_object + '.pkl', 'wb') as f:
    pickle.dump(voxel_set, f, pickle.HIGHEST_PROTOCOL)
