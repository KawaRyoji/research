import math
import numba

def point2sec(fs, point):
    return point / fs

def sec2point(fs, sec):
    return math.floor(sec * fs)

def calc_split_point(length, ratio):
    assert(ratio > 0 and ratio < 1)
    return int(length * ratio)

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def square(x):
    return x.real**2 + x.imag**2