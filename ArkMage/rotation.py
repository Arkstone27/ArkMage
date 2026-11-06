import numpy as np
from numba import njit, prange

def rotation_2d(data, angle: int):
    data_tourner = rotation_data(data[:, :, np.newaxis], angle)
    return np.squeeze(data_tourner)

@njit
def rotation_data(data, angle: int):

    if angle == 90:
        return rotation_90(data)
    elif angle == 180:
        return rotation_180(data)
    elif angle == 270:
        return rotation_270(data)

    return data.copy()

@njit(parallel=True)
def rotation_90(data):
    dimensions = data.shape
    w, h, n_canaux = dimensions

    data_tourner = np.empty((h, w, n_canaux), dtype=data.dtype)
    for x in prange(w):
        for y in range(h):
            data_tourner[y,w-1-x] = data[x, y]

    return data_tourner

@njit(parallel=True)
def rotation_180(data):
    dimensions = data.shape
    w, h, n_canaux = dimensions

    data_tourner = np.empty((w, h, n_canaux), dtype=data.dtype)
    for x in prange(w):
        for y in range(h):
            data_tourner[w-1-x,h-1-y] = data[x, y]

    return data_tourner

@njit(parallel=True)
def rotation_270(data):
    dimensions = data.shape
    w, h, n_canaux = dimensions

    data_tourner = np.empty((h, w, n_canaux), dtype=data.dtype)

    for x in prange(w):
        for y in range(h):
            data_tourner[h-1-y,x] = data[x, y]

    return data_tourner

if __name__ == '__main__':
    data = np.ones((5,6,1), dtype=np.float32)
    data[0,1] = 0.0

    data_tourner = rotation_data(data, 180).astype(int)

    for i in range(5):
        print(list(data_tourner[i]))