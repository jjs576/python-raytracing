import cupy as cp

rng = cp.random


def degrees_to_radians(degrees):
    return degrees * cp.pi / 180


def random_float(_min = 0, _max = 1):
    return rng.uniform(_min, _max)


def random_float_list(size: int, _min = 0, _max = 1):
    return rng.uniform(_min, _max, size).astype(cp.float32, copy=False)
