import numpy as np


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