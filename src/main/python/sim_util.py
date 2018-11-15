import os
import re
import json
import warnings
import logging
from itertools import product
from argparse import Namespace
import numpy as np
from skimage.io import imread_collection, imread
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
    
    with warnings.catch_warnings():
        # ignore annoying "low constrast image" warnings printed by skimage
        warnings.filterwarnings("ignore", message=".*?low contrast image.*?")
        for plane in range(img.shape[axis]):
            idx = tuple([slice(img.shape[ax]) if ax!=axis else plane for ax in range(len(img.shape))])
            img_i = img[idx].astype(np.float32)
            plane_padded = plane
            if pad_idx:
                pref = '0' * (len(str(img.shape[axis]-1)) - len(str(plane)))
                plane_padded = pref + str(plane_padded)
            imsave(os.path.join(base_path, file_pattern.format(plane=plane_padded)), img_i)
        
        
def load_tiff_sequence(raw_data_path, pattern=None, downsampling=None, dtype=None):
    # load data
    files = os.listdir(raw_data_path)
    files.sort(key=lambda f: split_str_digit(f))

    files = [f for f in files if (re.match(pattern, f) if pattern else True)]

    # load downsampled:
    if downsampling is not None:
        files = [f for i,f in enumerate(files) if i%downsampling == 0]
        
    logging.debug('reading {} files from directory {}.'.format(len(files), raw_data_path))

    # read raw phantom
    if downsampling is None:
        ic = imread_collection([os.path.join(raw_data_path, f) for f in files])
        img = ic.concatenate()
        logging.debug('loading done.')
        if dtype is not None:
            img = img.astype(dtype)
        
        return img

    # this should work even if we do no downsampling
    # keep it in the else anyway for the time being
    else:
        out = None
        for (i,f) in enumerate(files):
            img_i = imread(os.path.join(raw_data_path, f))
            img_i = simple_downsample(img_i, downsampling)
            if out is None:
                out = np.zeros( (len(files),) + img_i.shape, dtype=img_i.dtype)
            out[i] = img_i
        logging.debug('loading done.')
        if dtype is not None:
            out = out.astype(dtype)
        return out


def sim_lightsheet_img(img, desc, dn, right_illum,
                   na_illum, na_detect,
                   physical_dims=(400,400,50), ls_pos=200, lam=500,
                   ri_medium=RI_DEFAULT, padding=CONV_PADDING,
                   is_padded=False, conv_subblocks=DEFAULT_CONV_SUBBLOCKS,
                   bprop_nvolumes=4, z_plane_subset=None):
    
    logging.debug('applying padding to images.')
    
    # zero-pad image for conv
    if not is_padded:
        img = np.pad(img, ((padding,padding),(0,0),(0,0)), "constant")
        dn = np.pad(dn, ((padding,padding),(0,0),(0,0)), "constant")
    logging.debug('applying padding to images done.')
    
    logging.debug('flipping images.')
    # simulate right illumination with flip of x-axis
    if right_illum:
        img = np.flip(img, 2)
        dn = np.flip(dn, 2)
    logging.debug('flipping images done.')
    
    if right_illum:
        ls_pos = physical_dims[0] - ls_pos
    
    logging.debug('setting up simulators.')
    # create a microscope simulator for signal
    m = biobeam.SimLSM_Cylindrical(dn = dn, signal = img,
                       zfoc_illum=ls_pos,
                       NA_illum=na_illum, NA_detect=na_detect,
                       n_volumes=bprop_nvolumes, lam_illum =lam/1000, lam_detect =lam/1000,
                       size = physical_dims, n0 = ri_medium)
    
    logging.debug('setting up simulators done.')
    
    # make sure z-plane subset is iterable
    if z_plane_subset is not None and np.isscalar(z_plane_subset):
        z_plane_subset = [z_plane_subset]    
    
    print('simulating images at z-position-subset: {}'.format(z_plane_subset))

    z_shape = (img.shape[0] - padding * 2,) if z_plane_subset is None else (len(z_plane_subset),)
    out = np.zeros(z_shape + img.shape[1:], dtype=img.dtype)
    out_desc = None if desc is None else np.zeros(z_shape + img.shape[1:], dtype=img.dtype)

    # get positions and number of planes to simulate
    z_positions = range(out.shape[0]) if z_plane_subset is None else z_plane_subset
    
    for i, pos in enumerate(z_positions):

        logging.debug('simulating signal plane {} of {}.'.format(i+1, len(z_positions)))
        cz = pos - (img.shape[0] - padding*2) // 2
        cz = cz * m._bpm_detect.units[-1]

        image = m.simulate_image_z(cz=cz, zslice=padding, psf_grid_dim=(padding,padding), conv_sub_blocks=tuple(conv_subblocks))
        out[i] = image[padding] if not right_illum else np.flip(image[padding], 1)

    if desc is not None:
        if not is_padded:
            desc = np.pad(desc, ((padding,padding),(0,0),(0,0)), "constant")
        if right_illum:
            desc = np.flip(desc, 2)

        m.signal = desc
        for i, pos in enumerate(z_positions):
            logging.debug('simulating descriptors plane {} of {}.'.format(i+1, len(z_positions)))
            cz = pos - (img.shape[0] - padding*2) // 2
            cz = cz * m._bpm_detect.units[-1]

            image = m.simulate_image_z(cz=cz, zslice=padding, psf_grid_dim=(16,16), conv_sub_blocks=tuple(conv_subblocks))
            out_desc[i] = image[padding] if not right_illum else np.flip(image[padding], 1)
    
    return out, out_desc
    
    
def load_params(def_path):
    
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
    if not 'clip_max_ri' in params:
        params['clip_max_ri'] = None
    clip_max_ri = params['clip_max_ri']
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
    bprop_nvolumes = params['bprop_nvolumes']
    if not 'only_necessary_planes' in params:
        params['only_necessary_planes'] = False
    only_necessary_planes = params['only_necessary_planes']
    
    # if supplied, use a grid of point sources as descriptor channel
    if not 'grid_descs' in params:
        params['grid_descs'] = None
    grid_descs = params['grid_descs']
    
    res = Namespace()
    res.__dict__.update(params)
    return res


def preview_from_definition(def_path, z=0):
    
    params = load_params(def_path)
    img = load_tiff_sequence(params.raw_data_path, downsampling=None if params.downsampling <= 1 else params.downsampling, dtype=np.float16)
    
    # downsample img
    #if params.downsampling > 1:
    #    img = simple_downsample(img, params.downsampling)

    # make ri-delta from signal
    logging.debug('generating r.i. delta image.')
    dn = dn_from_signal(img, params.ri_medium, params.ri_delta_range, params.clip_max_ri)
    logging.debug('generating r.i. delta image done.')

    # make descriptor img
    logging.debug('generating descriptor image.')
    desc_img = np.zeros_like(img)
    for field in params.fields.values():
        for point in field['points']:
            point = list(np.array(point) // params.downsampling)
            desc_img[tuple(point)] = 1
    logging.debug('generating descriptor image done.')
    
    # blur slightly
    logging.debug('applying blur to images.')
    #desc_img = ndi.gaussian_filter(desc_img, 0.5)
    #img = ndi.gaussian_filter(img, 0.5)
    logging.debug('applying blur to images done.')
    
    # pad right away, otherwise images are copied later on
    img = np.pad(img, ((params.padding,params.padding),(0,0),(0,0)), "constant")
    dn = np.pad(dn, ((params.padding,params.padding),(0,0),(0,0)), "constant")
    desc_img = np.pad(desc_img, ((params.padding,params.padding),(0,0),(0,0)), "constant")

    res = {}
    for lam, right_illum in product(params.lambdas, ([False] if not params.two_sided_illum else [True, False])):
        for xi in range(len(params.x_locs)):
            
            physical_dims_ = tuple(list(np.array(params.phys_dims)))
            ls_pos_ = np.interp(params.x_locs[xi], (0, params.raw_data_dims[2]), (0, params.phys_dims[0]))

            # simulate signal and descriptors
            out_signal, out_desc = sim_lightsheet_img(img, desc_img, dn, right_illum, params.na_illum,
                                                      params.na_detect, physical_dims_, ls_pos_,
                                                      lam, params.ri_medium, params.padding, True,
                                                      params.conv_subblocks, params.bprop_nvolumes, z_plane_subset=z )
            
            k = (lam, right_illum, xi)
            res[k] = (out_signal, out_desc)

    return res
            
    
def sim_from_definition(def_path):
    
    # load settings
    params = load_params(def_path)

    img = load_tiff_sequence(params.raw_data_path, downsampling=None if params.downsampling <= 1 else params.downsampling)
    
    # LOG: loaded raw data

    # downsample img
    #if params.downsampling > 1:
    #    img = simple_downsample(img, params.downsampling)
        # LOG: downsampled

    dn = dn_from_signal(img, params.ri_medium, params.ri_delta_range, params.clip_max_ri)

    # make descriptor img
    if params.grid_descs is not None:
        Xs = np.meshgrid(*[np.arange(0,n) for n in img.shape], indexing='ij')
        grid_img = np.prod([_X%params.grid_descs==0 for _X in Xs], axis = 0).astype(np.float32)
   
    desc_img = np.zeros_like(img)
    for field in params.fields.values():
        for point in field['points']:
            point = list(np.array(point) // params.downsampling)
            desc_img[tuple(point)] = 1

    # blur slightly
    desc_img = ndi.gaussian_filter(desc_img, 0.5)
    img = ndi.gaussian_filter(img, 0.5)
    if params.grid_descs is not None:
        grid_img = ndi.gaussian_filter(grid_img, 0.5)
                                
    # pad right away, otherwise images are copied later on
    img = np.pad(img, ((params.padding,params.padding),(0,0),(0,0)), "constant")
    dn = np.pad(dn, ((params.padding,params.padding),(0,0),(0,0)), "constant")
    desc_img = np.pad(desc_img, ((params.padding,params.padding),(0,0),(0,0)), "constant")

    if params.grid_descs is not None:
        grid_img = np.pad(grid_img, ((params.padding,params.padding),(0,0),(0,0)), "constant")
    
    print(img.shape)
    print(desc_img.shape)

    for lam, right_illum in product(params.lambdas, ([False] if not params.two_sided_illum else [True, False])):

        for xi in range(len(params.x_locs)):

            physical_dims_ = tuple(list(np.array(params.phys_dims)))
            ls_pos_ = np.interp(params.x_locs[xi], (0, params.raw_data_dims[2]), (0, params.phys_dims[0]))

            # only simulate necessary planes if desired
            cut_min, cut_max = get_minmax_cut(params, xi)
            z_subset = None if not params.only_necessary_planes else list(range(int(cut_min[0]), int(cut_max[0])))

            # simulate signal and descriptors
            out_signal, out_desc = sim_lightsheet_img(img, desc_img, dn, right_illum, params.na_illum,
                                                      params.na_detect, physical_dims_, ls_pos_,
                                                      lam, params.ri_medium,
                                                      params.padding, True, params.conv_subblocks, params.bprop_nvolumes,
                                                      z_plane_subset=z_subset)

            # save the whole simulated volume
            out_dir_all = params.save_fstring.format(x=xi, y='all', z='all', lam=lam, illum=(1 if not right_illum else 2))
            save_as_sequence(out_signal, out_dir_all, file_pattern='bbeam_sim_c1_z{plane}.tif')
            save_as_sequence(out_desc, out_dir_all, file_pattern='bbeam_sim_c2_z{plane}.tif')

            if params.grid_descs is not None:
                out_grid, _ = sim_lightsheet_img(grid_img, None, dn, right_illum, params.na_illum,
                                                          params.na_detect, physical_dims_, ls_pos_,
                                                          lam, params.ri_medium,
                                                          params.padding, True, params.conv_subblocks,
                                                          params.bprop_nvolumes,
                                                          z_plane_subset=z_subset)
                save_as_sequence(out_grid, out_dir_all, file_pattern='bbeam_sim_c3_z{plane}.tif')

            yidx, zidx = np.meshgrid(range(len(params.y_locs)), range(len(params.z_locs)))
            for yi, zi in zip(yidx.flat, zidx.flat):

                field_info = params.fields[','.join(map(str, (xi, yi, zi)))]
                off = field_info['off']
                loc = [params.z_locs[zi], params.y_locs[yi], params.x_locs[xi]]
                min_ = np.array(loc) + np.array(off) - np.ceil(np.array(params.fov_size) / 2)
                max_ = np.array(loc) + np.array(off) + np.floor(np.array(params.fov_size) / 2)

                # out dir
                out_dir = params.save_fstring.format(x=xi, y=yi, z=zi, lam=lam, illum=(1 if not right_illum else 2))

                # cut signal and descriptors
                # NB: downsampling
                min_ = min_//params.downsampling
                max_ = max_//params.downsampling

                # correct for offset due to subset of planes simulated
                if params.only_necessary_planes:
                    min_[0] -= cut_min[0]
                    max_[0] -= cut_min[0]

                idxs = tuple([slice(int(mi), int(ma)) for mi, ma in zip(min_, max_)])

                # save
                save_as_sequence(out_signal[idxs], out_dir, file_pattern='bbeam_sim_c1_z{plane}.tif')
                save_as_sequence(out_desc[idxs], out_dir, file_pattern='bbeam_sim_c2_z{plane}.tif')
                if params.grid_descs is not None:
                    save_as_sequence(out_grid[idxs], out_dir, file_pattern='bbeam_sim_c3_z{plane}.tif')


def get_minmax_cut(params, xi):
    """
    get minimum / maximum of intervals we want to cut at a given x position
    takes downsampling into account!

    :param params: parameter dict
    :param xi: index of x position
    :return:  min and max as int-arrays
    """

    min_ = np.full(3, np.iinfo(np.int).max, dtype=np.int)
    max_ = np.full(3, np.iinfo(np.int).min, dtype=np.int)

    yidx, zidx = np.meshgrid(range(len(params.y_locs)), range(len(params.z_locs)))
    for yi, zi in zip(yidx.flat, zidx.flat):
        field_info = params.fields[','.join(map(str, (xi, yi, zi)))]
        off = field_info['off']
        loc = [params.z_locs[zi], params.y_locs[yi], params.x_locs[xi]]
        min_i = np.array(loc) + np.array(off) - np.ceil(np.array(params.fov_size) / 2)
        max_i = np.array(loc) + np.array(off) + np.floor(np.array(params.fov_size) / 2)
        min_i = min_i // params.downsampling
        max_i = max_i // params.downsampling

        min_ = np.min(np.stack([min_, min_i]), axis=0)
        max_ = np.max(np.stack([max_, max_i]), axis=0)

    return min_, max_


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


def dn_from_signal(signal, ri_medium=RI_DEFAULT, ri_range=RI_DELTA_RANGE, clip_max=None):

    # clip signal for ri generation
    if clip_max is not None:
        signal = np.clip(signal, 0, clip_max)

    dn = signal / np.max(signal) * ri_medium * (ri_range[1] - ri_range[0]) + ri_medium * ri_range[0]
    # no ri change where we have no signal/tissue
    dn[signal==0] = 0
    return dn
