import cv2
from .videocapinfo import VideoCapInfo
from util import *
class VideoCapture:
    """ This Class is a wrapper for cv2.VideoCapInfo, enhanced with additional functions and variables useful for the scope of the project.
        The constructor takes in input the path of the video file and create a cv2.VideoCapture and a VideoCapInfo object.
    """
    def __init__(self, file_name: str):
        self.cap = cv2.VideoCapture(file_name)
        self.video_cap_info = VideoCapInfo(self.cap)

    def process_video(self, process_func, frames_to_process=None):
        if frames_to_process == None:
            frames_to_process = self.video_cap_info.total_frames;
        for i in range(frames_to_process):
            frame_offset = int(self.video_cap_info.total_frames/frames_to_process)
            frame_to_process = frame_offset*(i)
            print("[INFO] Processing frame {}/{}".format(frame_to_process+1, self.video_cap_info.total_frames))
            # set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_process)
            # read the frame
            ret, frame = self.cap.read()
            if ret == True:
                process_func(frame)
        # reset the pos frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def get_first_frame(self):
        return self.get_frame(0)

    def get_frame(self, frame_no):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        _, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame