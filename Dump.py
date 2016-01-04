#!/usr/bin/env python

"""
Usage:
    Dump.py <input>

Input should be a valid HDF5 file.
"""

import docopt
import h5py

options = docopt.docopt(__doc__)

f = h5py.File(options['<input>'], 'r')

def printname(name):
    print name

f.visit(printname)

# f.close()

