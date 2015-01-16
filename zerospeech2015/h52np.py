#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: h52np.py
# date: Mon January 12 18:06 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""h52np: convert Thomas' h5 file to just a bunch of npy files.

"""

from __future__ import division

import os.path as path

import h5py
import numpy as np

def convert(h5file, outdir):
    fid = h5py.File(h5file, 'r')
    f = fid['features']
    start_ix = 0
    for ix, filename in enumerate(f['files']):
        end_ix = f['file_index'][ix] + 1
        feats = f['features'][start_ix:end_ix]
        filename = filename.split('/')[1]
        np.save(path.join(outdir, filename + '.npy'), feats)
        start_ix = end_ix
    fid.close()

if __name__ == '__main__':
    post_dir = '/home/mwv/data/zerospeech/htk_posteriors/'
    convert(path.join(post_dir, 'train_HTK_posterior_10_20.features'),
            path.join(post_dir, 'npy'))
