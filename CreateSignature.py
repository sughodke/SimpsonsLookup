#!/usr/bin/env python

"""
Usage:
    CreateSignature.py [options] <video> <output>
    CreateSignature.py ( -h | --help )

Options:
    --no-video                  Do not show video in output.
    --frame-velocity=VELOCITY   Jump timestamps with VELOCITY increments
                                [default: 200]
    --max-frames=COUNT          If specified, only process at most COUNT frames.
    --show-kps                  Show keypoints per frame.
    --trail-length=FRAMES       Show track trails with length FRAMES.
                                [default: 0]
    --no-cluster                Do not attempt clustering.
    --no-show-states            Do not show tracked states.
    --max-position-sigma=SIGMA  Do not show positions with >SIGMA error in location.

Input video can be anything OpenCV can read.
"""

import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")

import cv2
import docopt
import h5py
import numpy as np

def main():
    options = docopt.docopt(__doc__)

    # Open file for capture
    #'/Users/SidGhodke/Desktop/vlc-output.ts'
    cap = cv2.VideoCapture(options['<video>'])

    # Initiate STAR detector
    orb = cv2.ORB()

    # mapping from keypoint to timestamp
    f = h5py.File(options['<output>'], 'w')
    grp = f.create_group("subgroup")

    frames = 0

    while(cap.isOpened()):
        ts = cap.get(0)
        cap.set(0, ts + float(options['--frame-velocity']))

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find the keypoints with ORB
        kp = orb.detect(gray, None)

        # compute the descriptors with ORB
        kp, des = orb.compute(gray, kp)

        try:

            if options['--show-kps']:
                print('Frame index: {0} => {1} keypoints'.format(ts, des.shape))

            dset = grp.create_dataset('t'+str(round(ts)), data=des, dtype=des.dtype)

            frames += 1

        except Exception, e:
            print('Frame skipped', e)

        if not options['--no-video']:
            # draw only keypoints location,not size and orientation
            img2 = cv2.drawKeypoints(
                gray,
                kp,
                color=(0,255,0), 
                flags=0
            )
            
            cv2.imshow('frame', img2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ret, buf = cv2.imencode('.jpg', gray)
        # f = open('workfile.jpg', 'wb')
        # f.write(buf)

        if options['--max-frames'] is not None and \
            ts > float(options['--max-frames']) * 1000:
            break

    f.close()
    cap.release()

    if not options['--no-video']:
        cv2.destroyAllWindows()


    print('Written {0} frames'.format(frames))

  
if __name__ == '__main__':
    main()
