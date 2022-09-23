import cv2

class VideoCapInfo:
    """ This class stores useful information about a cv2.VideoCapture object, such as: total frames, video width, video height, and fps.
        Can be constructed passing a cv2.VideoCapture object
    """
    def __init__(self, cap: cv2.VideoCapture):
        """constructor for VideoCapInfo

        Args:
            cap (cv2.VideoCapture) - the VideoCapture
        """
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
