#!/usr/bin/env python

"""
Usage:
    FindFrame.py [options] <needle> <datafile>
    FindFrame.py ( -h | --help )

Options:
    --no-cluster                Do not attempt clustering.
    --no-show-states            Do not show tracked states.
    --max-position-sigma=SIGMA  Do not show positions with >SIGMA error in location.

Input file can be anything OpenCV can read.
"""

import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")

import cv2
import docopt
import h5py
import numpy as np

def main():
    options = docopt.docopt(__doc__)

    needle = cv2.imread(options['<needle>'], 0)

    f = h5py.File(options['<datafile>'], 'r')
    grp = f["subgroup"]

    # Initiate STAR detector
    orb = cv2.ORB()

    kp = orb.detect(needle, None)
    kp, des = orb.compute(needle, kp)

    print 'Opened {0} => {1} {2} {3}'.format(
            options['<needle>'], 
            needle.shape, 
            needle.dtype,
            des.shape
        )

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for ts in grp:
        frame_des = grp[ts][:]
        idx = ts

        try:

            matches = bf.match(des, frame_des)

            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            good_match = sum([x.distance for x in matches[:10]])
            
            if good_match < 100.0:
                print 'Frame %s matched with %f%% accuracy' % \
                        (idx, 100 - good_match)

        except Exception, e:
            print('Frame skipped', e)

  
if __name__ == '__main__':
    main()
