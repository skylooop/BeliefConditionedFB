import numpy as np

from skimage.draw import disk


def morris_water_maze(radius, platform_center, platform_radius):
    x = np.ones([2 * radius, 2 * radius], dtype=np.uint8)

    rr, cc = disk((radius, radius), radius - 1)
    x[rr, cc] = 0

    platform = np.zeros_like(x)
    rr, cc = disk(platform_center, platform_radius)
    platform[rr, cc] = 3  # goal
    x += platform

    return x
