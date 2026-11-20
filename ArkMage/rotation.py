import numpy as np
from numba import njit, prange

from ArkMage.image import ouvrir_image, Pixels

def rotation_2d(data, angle: int):
    data_tourner = rotation_data(data[:, :, np.newaxis], angle)
    return np.squeeze(data_tourner)

def rotation(pixels, angle: int | float):
    if isinstance(angle, float):
        data_tourner = rotation_fast(pixels.data, angle)
    else:
        data_tourner = rotation_data(pixels.data, angle)

    return Pixels(data_tourner, pixels.mode_couleur)

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

@njit(parallel=True)
def rotation_fast(data, angle):
    # Rotation sur sois même dans le même espace 
    # IMPORTANT : Les dimensions ne change pas (il vaut mieux évité les grandes rotations)
    # Rajouter une technique d'interpolation ?
    w, h, n_canaux = data.shape

    data_tourner = np.empty((w, h, n_canaux), dtype=data.dtype)
    rad = np.radians(angle)
    cos_angle = np.cos(rad)
    sin_angle = np.sin(rad)

    milieu_x, milieu_y = w/2, h/2

    for x in prange(w):
        for y in range(h):
                x_relatif, y_relatif = x - milieu_x, y - milieu_y
                x_data_relatif = x_relatif * cos_angle - y_relatif * sin_angle
                y_data_relatif = x_relatif * sin_angle + y_relatif * cos_angle

                x_data, y_data = int(x_data_relatif + milieu_x), int(y_data_relatif + milieu_y)

                if 0 <= x_data < w and 0 <= y_data < h:
                    data_tourner[x, y] = data[x_data, y_data]

    return data_tourner

if __name__ == '__main__':
    pixels = ouvrir_image(True)
    pixels.show()

    rotation(pixels, 90).show()
    rotation(pixels, 5.0).show()