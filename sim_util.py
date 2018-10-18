import os
import re
import json
from itertools import product
import numpy as np
from skimage.io import imread_collection
from skimage.io import imsave
import biobeam
from scipy import ndimage as ndi


# defaults
CONV_PADDING = 16
RI_DEFAULT = 1.33
RI_DELTA_RANGE = (.03, .05)
DEFAULT_CONV_SUBBLOCKS = (4, 4)


def random_points_in_volume_min_distance(low, high, min_dist, n_points):
    """
    simple generation of random points in volume with minimal distance to each other
    
    Parameters
    ==========
    low: iterable
        lower bound of volume, inclusive
    high: iterable
        upper bound of volume, exclusive
    min_dist: int
        minimal distance of points to each other
    n_points: int
        how many points to generate
        
    Returns
    =======
    points: array
        n_points * max(len(low), len(high)) array of points
    """
    
    # crude lower bound on how many points we can definitely sample
    # raise exception if user asks for too many
    lb_points_possible = (np.prod(np.floor((np.array(high) - np.array(low)) / min_dist / 2)))
    if lb_points_possible < n_points:
        raise ValueError('this function may run into an endless loop generating ' + 
                         'this many points in the specified vlolume, please pick fewer.')

    # rejection sampling of random integer points in volume 
    # with defined minimal distance to each other
    points = []
    while len(points) < n_points:
        candidate = [np.random.randint(l, h) for l, h in zip(low, high)]
        rejected = False
        for p in points:
            if np.linalg.norm(np.array(p) - np.array(candidate)) < min_dist:
                rejected = True
                break
        if not rejected:
            points.append(candidate)

    return np.array(points)


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


def save_as_sequence(img, base_path, file_pattern='image{plane}.tif', pad_idx=True, axis=0):
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    for plane in range(img.shape[axis]):
        idx = tuple([slice(img.shape[ax]) if ax!=axis else plane for ax in range(len(img.shape))])
        img_i = img[idx].astype(np.float32)
        plane_padded = plane
        if pad_idx:
            pref = '0' * (len(str(img.shape[axis]-1)) - len(str(plane)))
            plane_padded = pref + str(plane_padded)
        imsave(os.path.join(base_path, file_pattern.format(plane=plane_padded)), img_i)
        
        
def load_tiff_sequence(raw_data_path, pattern=None):
    # load data
    files = os.listdir(raw_data_path)
    files.sort(key=lambda f: split_str_digit(f))

    # read raw phantom
    ic = imread_collection([os.path.join(raw_data_path, f) for f in files if (re.match(pattern, f) if pattern else True)])
    img = ic.concatenate()
    return img


def sim_lightsheet_img(img, desc, dn, right_illum,
                   na_illum, na_detect,
                   physical_dims=(400,400,50), ls_pos=200, lam=500,
                   ri_medium=RI_DEFAULT, ri_range=RI_DELTA_RANGE, padding=CONV_PADDING, conv_subblocks=DEFAULT_CONV_SUBBLOCKS):
    
    # zero-pad image for conv
    img = np.pad(img, ((padding,padding),(0,0),(0,0)), "constant")
    desc = np.pad(desc, ((padding,padding),(0,0),(0,0)), "constant")
    dn = np.pad(dn, ((padding,padding),(0,0),(0,0)), "constant")
    
    # simulate right illumination with flip of x-axis
    if right_illum:
        img = np.flip(img, 2)
        desc = np.flip(desc, 2)
        dn = np.flip(dn, 2)
    
    # create a microscope simulator for signal
    m = biobeam.SimLSM_Cylindrical(dn = dn, signal = img, zfoc_illum=(-1 if right_illum else 1) * physical_dims[0]/2 - ls_pos,
                       NA_illum=na_illum, NA_detect=na_detect,
                       n_volumes=4, lam_illum =lam/1000, lam_detect =lam/1000,
                       size = physical_dims, n0 = ri_medium)
    
    # create a microscope simulator for descriptors
    m_desc = biobeam.SimLSM_Cylindrical(dn = dn, signal = desc, zfoc_illum=(-1 if right_illum else 1) * physical_dims[0]/2 - ls_pos,
                       NA_illum=na_illum, NA_detect=na_detect,
                       n_volumes=4, lam_illum =lam/1000, lam_detect =lam/1000,
                       size = physical_dims, n0 = ri_medium)
    
    
    out = np.zeros((img.shape[0] - padding*2,) + img.shape[1:], dtype=img.dtype)
    out_desc = np.zeros((img.shape[0] - padding*2,) + img.shape[1:], dtype=img.dtype)
    for i in range(out.shape[0]):

        cz = i - (img.shape[0] - padding*2) // 2
        cz = cz * m._bpm_detect.units[-1]

        image = m.simulate_image_z(cz=cz, zslice=padding, psf_grid_dim=(16,16), conv_sub_blocks=tuple(conv_subblocks))
        out[i] = image[padding] if not right_illum else np.flip(image[padding], 1)
        
        image = m_desc.simulate_image_z(cz=cz, zslice=padding, psf_grid_dim=(16,16), conv_sub_blocks=tuple(conv_subblocks))
        out_desc[i] = image[padding] if not right_illum else np.flip(image[padding], 1)

    return out, out_desc
    

def sim_from_definition(def_path):
    
    # load settings
    with open(def_path, 'r') as fd:
        params = json.load(fd)
        
    save_fstring = params['save_fstring']
    seed = params['seed']
    raw_data_path = params['raw_data_path']
    raw_data_dims = params['raw_data_dims']
    downsampling = params['downsampling']
    phys_dims = params['phys_dims']
    na_illum = params['na_illum']
    na_detect = params['na_detect']
    ri_medium =  params['ri_medium']
    ri_delta_range = params['ri_delta_range']
    two_sided_illum = params['two_sided_illum']
    lambdas = params['lambdas']
    fov_size = params['fov_size']
    point_fov_size = params['point_fov_size']
    n_points_per_fov = params['n_points_per_fov']
    points_min_distance = params['points_min_distance']
    min_off = params['min_off']
    max_off = params['max_off']
    x_locs = params['x_locs']
    y_locs = params['y_locs']
    z_locs = params['z_locs']
    fields = params['fields']
    padding = params['padding']
    conv_subblocks = params['conv_subblocks']

    img = load_tiff_sequence(raw_data_path)
    
    # LOG: loaded raw data

    # downsample img
    if downsampling > 1:
        img = simple_downsample(img, downsampling)
        # LOG: downsampled

    dn = dn_from_signal(img, ri_medium, ri_delta_range)

    # make descriptor img
    desc_img = np.zeros_like(img)
    for field in params['fields'].values():
        for point in field['points']:
            point = list(np.array(point) // downsampling)
            desc_img[tuple(point)] = 1
    
    # TODO: blur slightly?
    desc_img = ndi.gaussian_filter(desc_img, 1.0)

    for lam, right_illum in product(lambdas, ([False] if not two_sided_illum else [True, False])):

        for xi in range(len(x_locs)):

            physical_dims_ = tuple(list(np.array(phys_dims)//downsampling))
            ls_pos_ = np.interp(x_locs[xi]//downsampling, (0, raw_data_dims[2]//downsampling), (0, phys_dims[2]//downsampling))

            # simulate signal and descriptors
            out_signal, out_desc = sim_lightsheet_img(img, desc_img, dn, right_illum, na_illum, na_detect, physical_dims_, ls_pos_,
                               lam, ri_medium, ri_delta_range, padding, conv_subblocks)

            # save the whole simulated volume
            out_dir_all = save_fstring.format(x=xi, y='all', z='all', lam=lam, illum=(1 if not right_illum else 2))
            save_as_sequence(out_signal, out_dir_all, file_pattern='bbeam_sim_c1_z{plane}.tif')
            save_as_sequence(out_desc, out_dir_all, file_pattern='bbeam_sim_c2_z{plane}.tif')

            yidx, zidx = np.meshgrid(range(len(y_locs)), range(len(z_locs)))    
            for yi, zi in zip(yidx.flat, zidx.flat):

                field_info = fields[','.join(map(str, (xi, yi, zi)))]
                off = field_info['off']
                loc = [z_locs[zi], y_locs[yi], x_locs[xi]]
                min_ = np.array(loc) + np.array(off) - np.ceil(np.array(fov_size) / 2)
                max_ = np.array(loc) + np.array(off) + np.floor(np.array(fov_size) / 2)

                # out dir
                out_dir = save_fstring.format(x=xi, y=yi, z=zi, lam=lam, illum=(1 if not right_illum else 2))

                # cut signal and descriptors
                # NB: downsampling
                min_ = min_//downsampling
                max_ = max_//downsampling

                idxs = tuple([slice(int(mi), int(ma)) for mi, ma in zip(min_, max_)])

                # save
                save_as_sequence(out_signal[idxs], out_dir, file_pattern='bbeam_sim_c1_z{plane}.tif')
                save_as_sequence(out_desc[idxs], out_dir, file_pattern='bbeam_sim_c2_z{plane}.tif')


def simple_downsample(a, factors):
    """
    very simple subsampling of array

    Parameters
    ----------
    a: array
    the array to downsample
    factors: int or sequence of ints
    downsampling factors

    Returns
    -------
    a2: array
    subsampled version of a
    """

    if np.isscalar(factors):
        factors = [factors] * len(a.shape)

    a2 = a
    for i, f in enumerate(factors):
        a2 = np.take(a2, np.arange(0, a2.shape[i], f), axis=i)

    return a2


def random_spots_in_radius(n_spots, n_dim, radius):
    """
    get random relative (integer) coordinates within radius
    """

    if np.isscalar(radius):
        radius = np.array([radius] * n_dim)
    else:
        radius = np.array(radius)

    res_spots = []
    while len(res_spots) < n_spots:
        # uniformly distributed on hypersquare
        candidate = [np.random.randint(-radius[i], radius[i] + 1) for i in range(len(radius))]
        # reject spots not in hypersphere
        if (np.sum(np.array(candidate) ** 2 / radius ** 2) <= 1):
            res_spots.append(candidate)

    return np.array(res_spots)


def dn_from_signal(signal, ri_medium=RI_DEFAULT, ri_range=RI_DELTA_RANGE):
    dn = signal / np.max(signal) * ri_medium * (ri_range[1] - ri_range[0]) + ri_medium * ri_range[0]
    # no ri change where we have no signal/tissue
    dn[signal==0] = 0
    return dn
