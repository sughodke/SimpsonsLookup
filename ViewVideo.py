import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")
import cv2
import requests
from io import BytesIO
import numpy as np

cap = cv2.VideoCapture('/Users/SidGhodke/Desktop/vlc-output.ts')

# Initiate detectors
dt = [
    cv2.ORB(200, 1.02, 100),
    cv2.ORB(),
    cv2.FeatureDetector_create("STAR"),
    # brief = cv2.DescriptorExtractor_create("BRIEF")
]

dt_idx=0

K=10
attempts=10/2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags=cv2.KMEANS_RANDOM_CENTERS

while(cap.isOpened()):
    ts = cap.get(0)
    cap.set(0, ts + 200)

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find the keypoints
    kp = dt[dt_idx].detect(gray, None)

    # compute the descriptors
    # kp, des = dt[dt_idx].compute(gray, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(
        gray,
        kp,
        color=(0,255,0), 
        flags=0
    )

    try:
        Z = np.float32(np.array([k.pt for k in kp]))
        ret,label,center=cv2.kmeans(
            Z, #data
            K,
            criteria,
            attempts,
            flags
        )

        for i in range(K):
            x, y = center[i]
            cv2.circle(img2, (x, y), radius=15, color=(255,0,0))
    except:
        pass

    cv2.imshow('frame', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print gray.shape, gray.dtype
        break

    if cv2.waitKey(1) & 0xFF == ord('t'):
        dt_idx = (dt_idx + 1) % len(dt)
        print 'Switched to %s' % dt[dt_idx]

cap.release()
cv2.destroyAllWindows()
