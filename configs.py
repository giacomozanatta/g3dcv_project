from lib.point import *
from lib.region import *
import numpy as np
import cv2
from lib.voxel import *
### CONFIG FILE ###
# Settings for objects

working_object = "obj01" # object to process (object id)

DEBUG = True # debug flag (if true, print info on console)

calibration = {
    "frames_to_process": 100, # no of frames to consider for calibration
    "input_video": "data/calibration.mp4"
}



# bgrem_yellow_rem =  Region(1000,1020,200,800) # not used

bg_region = Region(0,1080,0,1100) # region to consider for background removal

b_min = np.array([50,  25, 10],np.uint8) # BGR notation
b_max = np.array([180,  160, 90],np.uint8) # BGR notation

approx_poly_epsilon = 5 

dilate_kernel_opening = np.ones((2, 2), np.uint8)
erode_kernel_opening = np.ones((16, 16), np.uint8)

dilate_kernel_closing = np.ones((1, 1), np.uint8)
erode_kernel_closing = np.ones((2,2), np.uint8)

corner_subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)

objects = {
    "obj01": {
        "center": Point(660,540),
        "padding": 310,
        "input_video": "data/obj01.mp4",
        "enhanced_bg_removal": False,
        "voxel_set_padding": 60, # VOXEL SET: 
        "voxel_set_N": 100,
        "voxel_set_center": Point3D(0,0,140), # VOXEL SET CENTER, in world coordinates
    },
    "obj02": {
        "center": Point(660,540),
        "padding": 310,
        "input_video": "data/obj02.mp4",
        "enhanced_bg_removal": True,
        "voxel_set_padding": 60,
        "voxel_set_N": 100,
        "voxel_set_center": Point3D(0,0,140),
    },
    "obj03": {
        "center": Point(660,540),
        "padding": 310,
        "input_video": "data/obj03.mp4",
        "enhanced_bg_removal": False,
        "voxel_set_padding": 65,
        "voxel_set_N": 100,
        "voxel_set_center": Point3D(0,0,140),
    },
    "obj04": {
        "center": Point(660,540),
        "padding": 310,
        "input_video": "data/obj04.mp4",
        "enhanced_bg_removal": False,
        "voxel_set_padding": 60,
        "voxel_set_N": 100,
        "voxel_set_center": Point3D(0,0,150),
    },
}

axis = np.float32([[0,0,0], [50,0,0],[0,50,0], [0,0,50]]).reshape(-1,3)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
