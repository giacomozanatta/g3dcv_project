import cv2

def undistort_frame(frame, K, dist):
    h,  w = frame.shape[:2]
    #cv2.imwrite('beforecalibration.png', frame)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    # undistort: compensate lens ditortion
    undist = cv2.undistort(frame, K, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    undist = undist[y:y+h, x:x+w]
    #cv2.imwrite('aftercalibration.png', undist)
    return undist