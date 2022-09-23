import cv2
from .videocapinfo import VideoCapInfo


class VideoCapture:
    """This Class is a wrapper for cv2.VideoCapInfo, enhanced with additional functions and variables useful for the scope of the project.
        The constructor takes in input the path of the video file and create a cv2.VideoCapture and a VideoCapInfo object.
    """
    def __init__(self, file_name: str):
        self.cap = cv2.VideoCapture(file_name)
        self.video_cap_info = VideoCapInfo(self.cap)