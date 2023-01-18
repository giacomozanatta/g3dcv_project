from lib.point import *
from lib.region import *
import numpy as np
DEBUG = True

calibration = {
    "frames_to_process": 100,
    "input_video": "data/calibration.mp4"
}

bgrem_yellow_rem =  Region(1000,1020,200,800)

bg_region = Region(0,1080,0,1000)

approx_poly_epsilon = 10 

dilate_kernel = np.ones((2, 2), np.uint8)
erode_kernel = np.ones((1, 1), np.uint8)
objects = {
    "obj01": {
        "center": Point(660,540),
        "padding": 310,
        "input_video": "data/obj01.mp4",
        "enhanced_bg_removal": False
    },
    "obj02": {
        "center": Point(660,540),
        "padding": 310,
        "input_video": "data/obj02.mp4",
        "enhanced_bg_removal": True
    },
    "obj03": {
        "center": Point(660,540),
        "padding": 310,
        "input_video": "data/obj03.mp4",
        "enhanced_bg_removal": False
    },
    "obj04": {
        "center": Point(660,540),
        "padding": 310,
        "input_video": "data/obj04.mp4",
        "enhanced_bg_removal": False
    },
}

