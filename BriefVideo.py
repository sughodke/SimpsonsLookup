import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")
import cv2
import requests
from io import BytesIO
# from collections import namedtuple
# Features = namedtuple('Features', ['kp', 'des'])

cap = cv2.VideoCapture('/Users/SidGhodke/Desktop/vlc-output.ts')

# Initiate STAR detector
star = cv2.FeatureDetector_create("STAR")

# Initiate BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")

# Initiate STAR detector
orb = cv2.ORB(200)



while(cap.isOpened()):
    ts = cap.get(0)
    cap.set(0, ts + 200)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find the keypoints with ORB
    orb_kp = orb.detect(gray, None)

    # compute the descriptors with ORB
    orb_kp, orb_des = orb.compute(gray, orb_kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(
        gray,
        orb_kp,
        color=(0,255,0), 
        flags=0
    )
    # cv2.imshow('orb', img2)

    # find the keypoints with STAR
    brief_kp = star.detect(gray, None)

    # compute the descriptors with BRIEF
    brief_kp, brief_des = brief.compute(gray, brief_kp)

    try:
        print orb_des.shape, brief_des.shape
    except:
        pass

    img3 = cv2.drawKeypoints(
        # gray,
        img2,
        brief_kp,
        color=(255,0,0), 
        flags=0
    )
    cv2.imshow('brief', img3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        from pprint import pprint
        pprint(brief_des)
        break

cap.release()
cv2.destroyAllWindows()
