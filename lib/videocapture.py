import cv2
from .videocapinfo import VideoCapInfo


class VideoCapture:
    """ This Class is a wrapper for cv2.VideoCapInfo, enhanced with additional functions and variables useful for the scope of the project.
        The constructor takes in input the path of the video file and create a cv2.VideoCapture and a VideoCapInfo object.
    """
    def __init__(self, file_name: str):
        self.cap = cv2.VideoCapture(file_name)
        self.video_cap_info = VideoCapInfo(self.cap)

    def process_video(self, process_func, frames_to_process):
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
    def get_first_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        return frame