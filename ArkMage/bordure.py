from ArkMage.image import Pixels, ouvrir_image, convolution_vecteurs
from ArkMage.commun import flou_classique_fast, calcul_distance, conversion_gris
from ArkMage.scaling import meilleur_winsorisation

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def calcul_bord_fast(data, table_sommes):
    dimensions = data.shape
    w, h = dimensions[0], dimensions[1]

    data_flou = flou_classique_fast(table_sommes, rayon=1)
    data_bord = np.empty((w,h,1), dtype=np.float32)

    for x in prange(w):
        for y in range(h):
            data_bord[x, y, 0] = calcul_distance(data[x, y], data_flou[x, y])

    return data_bord

def calcul_bord(pixels: Pixels):
    data_bord = calcul_bord_fast(pixels.data, pixels.get_table_sommes())
    data_bord = meilleur_winsorisation(data_bord)

    return Pixels(data_bord, "L")

@njit(parallel=True)
def addition_pythagore(gradient_x, gradient_y):
    dimensions = gradient_x.shape
    w, h = dimensions[0], dimensions[1]

    data_bord = np.empty(dimensions, dtype=np.float32)

    for x in prange(w):
        for y in range(h):
            somme = gradient_x[x,y,0]**2 + gradient_y[x,y,0]**2
            data_bord[x,y,0] = np.sqrt(somme)
    return data_bord

@njit(parallel=True)
def calcul_direction(gradient_x, gradient_y):
    dimensions = gradient_x.shape
    w, h = dimensions[0], dimensions[1]

    data_direction = np.empty(dimensions, dtype=np.float32)

    for x in prange(w):
        for y in range(h):
            data_direction[x,y,0] = np.atan2( gradient_y[x,y,0], gradient_x[x,y,0])
    return data_direction

def operateur_sobel(pixels: Pixels, direction=False):
    data_gris = conversion_gris(pixels).data

    vecteurs_x = (np.array([[1], [0], [-1]], dtype=np.float32), np.array([[0.25, 0.5, 0.25]], dtype=np.float32))
    gradient_x = convolution_vecteurs(data_gris, vecteurs_x)

    vecteurs_y = (np.array([[0.25], [0.5], [0.25]], dtype=np.float32), np.array([[1, 0, -1]], dtype=np.float32))
    gradient_y = convolution_vecteurs(data_gris, vecteurs_y)

    data_bord = addition_pythagore(gradient_x, gradient_y)
    if direction:
        data_direction = calcul_direction(gradient_x, gradient_y)
    else:
        data_direction = None
    return data_bord, data_direction

def affichage_direction(data_direction):
    data_angle = (data_direction + np.pi) / (2*np.pi)
    w, h, n = data_angle.shape
    data_hsv = np.ones((w,h, 3))

    data_hsv[:, :, 0] = data_angle[:, :, 0]


    Pixels(data_hsv, "HSV").show()
if __name__ == '__main__':
    pixels = ouvrir_image()

    pixels_bord = calcul_bord(pixels)
    pixels_bord.show()

    data_bord, data_direction = operateur_sobel(pixels, True)

    Pixels(data_bord, "L").show()
    affichage_direction(data_direction)