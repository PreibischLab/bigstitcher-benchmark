import os

import numpy as np
import biobeam
from skimage.io import imsave
from skimage.io import imread_collection

from sim_util import split_str_digit
from sim_util import RI_DEFAULT, RI_DELTA_RANGE, dn_from_signal


def sim_lightsheet(base_path, out_dir, phantom_dir='sim-phantom', in_pattern='sim(\d+).tif', right_illum=False,
                   physical_dims=(400,400,50), ls_pos=200, lam=500, out_pattern='sim-biobeam{}.tif', planes_to_simulate=None,
                   ri_medium=RI_DEFAULT, ri_range=RI_DELTA_RANGE):
    
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
    dn = dn_from_signal(signal, ri_medium, ri_range)
    
    #create a microscope simulator
    m = biobeam.SimLSM_Cylindrical(dn = dn, signal = signal, zfoc_illum=(-1 if right_illum else 1) * physical_dims[0]/2 - ls_pos,
                       NA_illum= .1, NA_detect=.45,
                       n_volumes=4, lam_illum =lam/1000, lam_detect =lam/1000,
                       size = physical_dims, n0 = ri_medium)
    
    
    # make outdir if necessary
    out_path = os.path.join(base_path, out_dir)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out = np.zeros((img.shape[0] - 16*2,) + img.shape[1:], dtype=img.dtype)
    for i in range(out.shape[0]):
        if planes_to_simulate is not None and not i in planes_to_simulate:
            continue

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
        (False, 125, 400),
        (False, 200, 400),
        (False, 325, 400),
        (True, 125, 400),
        (True, 200, 400),
        (True, 325, 400),
        (False, 125, 600),
        (False, 200, 600),
        (False, 325, 600),
        (True, 125, 600),
        (True, 200, 600),
        (True, 325, 600)
    ]
    
    for (right_illum, ls_pos, lam) in params:
        out_dir = 'sim-biobeam-illum_{}-ls_pos_{}-lam-{}'.format('right' if right_illum else 'left', ls_pos, lam)
        sim_lightsheet(base_dir, out_dir, right_illum=right_illum, ls_pos=ls_pos, lam=lam)
    print('ALL DONE.')
