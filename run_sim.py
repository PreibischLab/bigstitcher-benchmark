import os
import re

import numpy as np
import biobeam
import gputools
import pyopencl
from matplotlib import pyplot as plt
from skimage.io import imsave
from skimage.io import imread_collection


def split_str_digit(s):
    """
    split s into numeric (integer) and non-numeric parts
    return split as a tuple of ints and strings
    """
    res = []
    for m in re.finditer('(\d*)(\D*)', s):
        for g in m.groups():
            if g != '':
                try:
                    res.append(int(g))
                except ValueError:
                    res.append(g)
    return tuple(res)


def sim_lightsheet(base_path, out_dir, phantom_dir='sim-phantom', in_pattern='sim(\d+).tif', right_illum=False,
                   physical_dims=(800,800,100), ls_pos=400, lam=500, out_pattern='sim-biobeam{}.tif'):
    
    in_path = os.path.join(base_path, phantom_dir)
    files = os.listdir(in_path)
    files.sort(key=lambda f: split_str_digit(f))

    # read raw phantom
    ic = imread_collection([os.path.join(in_path, f) for f in files])
    img = ic.concatenate()

    # zero-pad image for conv
    img = np.pad(img, ((16,16),(0,0),(0,0)), "constant")
    
    # simulate right illumination with flip of x-axis
    if right_illum:
        img = np.flip(img, 2)
    
    # setup signal and refractive index
    signal = img
    dn = (np.sqrt(signal / np.max(signal))) * 0.01

    #create a microscope simulator
    m = biobeam.SimLSM_Cylindrical(dn = dn, signal = signal, zfoc_illum=physical_dims[0]/2 - ls_pos,
                       NA_illum= .1, NA_detect=.45,
                       n_volumes=2, lam_illum =lam/1000, lam_detect =lam/1000,
                       size = physical_dims, n0 = 1.33)
    
    
    # make outdir if necessary
    out_path = os.path.join(base_path, out_dir)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out = np.zeros((img.shape[0] - 16*2,) + img.shape[1:], dtype=img.dtype)
    for i in range(out.shape[0]):

        cz = i - (img.shape[0] - 16*2) // 2
        cz = cz * m._bpm_detect.units[-1]

        image = m.simulate_image_z(cz=cz, psf_grid_dim=(16,16), conv_sub_blocks=(4,4))
        out[i] = image[16] if not right_illum else np.flip(image[16], 1)

        # save immediately
        imsave(os.path.join(out_path, out_pattern.format(i)), out[i])
        print("{} of {} done".format(i+1, out.shape[0]))
    
    
if __name__ == '__main__':
    
    base_dir = '/scratch/hoerl/sim_tissue/'
    
    params = [
        (False, 250, 400),
        (False, 400, 400),
        (False, 550, 400),
        (True, 250, 400),
        (True, 400, 400),
        (True, 550, 400),
        (False, 250, 600),
        (False, 400, 600),
        (False, 550, 600),
        (True, 250, 600),
        (True, 400, 600),
        (True, 550, 600)
    ]
    
    for (right_illum, ls_pos, lam) in params:
        out_dir = 'sim-biobeam-illum_{}-ls_pos_{}-lam-{}'.format('right' if right_illum else 'left', ls_pos, lam)
        sim_lightsheet(base_dir, out_dir, right_illum=right_illum, ls_pos=ls_pos, lam=lam)
    print('ALL DONE.')
