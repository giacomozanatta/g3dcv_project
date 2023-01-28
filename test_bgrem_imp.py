import numpy as np
import cv2
from lib.videocapture import *
from backgroundremoval import *
from poseestimation import *
from undistort import *
from test import *
import configs

def intensity_correction(frame, max):
    threshold = 255 / max
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            if frame[y,x] > max:
                frame[y,x] = 255
            else:
                frame[y,x] *= threshold

def correct_gamma(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def f(frame):
    # save the frame

    _frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_3 = correct_gamma(_frame, 5)
    cv2.imshow('GAMMA', frame_3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('GAMMA-GRAY', gray)
    intensity_correction(gray, 150)
    cv2.imshow('GAMMA-GRAY-INTENSITY', gray)

    
    #dst = cv2.equalizeHist(gray)
    #cv2.imshow('GAMMA-GRAY-EQUALIZED', dst)
    #_frame = 10 * gray
    #cv2.imshow('aaa', _frame)
    # S = c*r^y (c, y positive constant)
    #background_removal(frame, configs)
    #cv2.imshow('RESULT', frame)
    cv2.waitKey(25)

video_capture = VideoCapture('data/' + configs.working_object + '.mp4')
video_capture.process_video(f)

frame = video_capture.get_frame(310)
frame_2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
frame_3 = correct_gamma(frame, 3)
cv2.imshow('ciao', frame_3)
frame_3 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('test.png', frame)
#background_removal(frame, configs)


#video_capture.process_video(f)


cv2.imshow('bgrem', frame_2)
cv2.imshow('bgrem2', frame_3)
cv2.waitKey(0)