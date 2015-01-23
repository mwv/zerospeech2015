"""
Convert sup7_train, deep_cos_cos2 and htk_posteriors to .spro format
"""

import numpy as np
import struct
import os
import os.path as path
import glob

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
    if arr.shape[1] > 32767:
        raise ValueError('array too large ({0} > 32767)'.format(arr.shape[1]))
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
    p += arr.astype(np.float32).tostring()

    # # data
    # p += ''.join(struct.pack('=f', x) for x in arr.flatten())

    return p

if __name__ == '__main__':
    base_dir = path.join(os.environ['HOME'], 'scratch', 'zerospeech')

    # 1. sup7
    data_dir = path.join(base_dir, 'npz_sup7_train')
    np_dir = path.join(data_dir, 'npz')
    spro_dir = path.join(data_dir, 'spro')

    for npz_file in glob.iglob(path.join(np_dir, '*.npz')):
        bname = path.splitext(path.basename(npz_file))[0]
        x = np.load(npz_file)['features']
        b = arr2bin(x, 100)
        with open(path.join(spro_dir, bname + '.spro'), 'wb') as fid:
            fid.write(b)

    # 2. deep_cos_cos2
    data_dir = path.join(base_dir, 'deep_cos_cos2')
    np_dir = path.join(data_dir, 'npz')
    spro_dir = path.join(data_dir, 'spro')

    for npz_file in glob.iglob(path.join(np_dir, '*.npz')):
        bname = path.splitext(path.basename(npz_file))[0]
        x = np.load(npz_file)['features']
        b = arr2bin(x, 100)
        with open(path.join(spro_dir, bname + '.spro'), 'wb') as fid:
            fid.write(b)

    # 3. htk_posteriors
    data_dir = path.join(base_dir, 'htk_posteriors')
    np_dir = path.join(data_dir, 'npy')
    spro_dir = path.join(data_dir, 'spro')

    for npy_file in glob.iglob(path.join(np_dir, '*.npy')):
        bname = path.splitext(path.basename(npy_file))[0]
        x = np.load(npy_file['features'])
        b = arr2bin(x, 100)
        with open(path.join(spro_dir, bname + '.spro'), 'wb') as fid:
            fid.write(b)
