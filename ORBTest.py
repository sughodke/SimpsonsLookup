import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")
import cv2
# from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('workfile.jpg', 0)

# Initiate STAR detector
orb = cv2.ORB()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

points = []
for k in kp:
    points.append(k.pt)
    
Z = np.float32(np.array(points))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(
    data=Z,
    K=5,
    criteria=criteria,
    attempts=10,
    flags=cv2.KMEANS_RANDOM_CENTERS
)

for i in range(5):
    x, y = center[i]
    cv2.circle(img, (x, y), radius=15, color=(255,0,0))

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
# cv2.imwrite('workfile2.jpg', img2)

cv2.imshow('frame', img2)
cv2.waitKey(0)

