#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: npy2binary.py
# date: Mon January 12 18:53 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""npy2binary: convert numpy arrays to spro binary files

"""

from __future__ import division

import struct


def arr2bin(arr, framerate, header_vals=None):
    """
    Convert a numpy array to spro style binary string.

    Parameters
    ----------
    arr : ndarray (nsamples, nfeatures)
    framerate : float
        framerate in Hertz
    header_vals : dict
        optional key, value pairs to put in the header

    Returns
    -------
    p : string
        encoded input

    """
    p = ''
    if header_vals:
        # add in optional text header
        p += '<header>\n'
        for key, val in header_vals.iteritems():
            p += '{0} = {1}\n'.format(key, val)
        p += '</header>\n'

    # binary header
    p += struct.pack('=HQf', arr.shape[1], 0, framerate)

    # data
    p += ''.join(struct.pack('=f', x) for x in arr.flatten())

    return p
