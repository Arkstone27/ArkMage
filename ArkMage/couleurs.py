import numpy as np
from numba import njit

@njit
def minmax(couleur):
    minimum, maximum = couleur[2], couleur[2]
    for i in range(2):
        if couleur[i] < minimum:
            minimum = couleur[i]
        elif couleur[i] > maximum:
            maximum = couleur[i]
    return minimum, maximum

@njit
def calcul_luminosite(couleur):
    minimum, maximum = minmax(couleur)
    return maximum - minimum

@njit
def ajuster_luminosite(couleur, facteur):
    if facteur > 0:
        return couleur + (1-couleur)*facteur
    return couleur + couleur*facteur

@njit
def calcul_saturation(couleur):
    minimum, maximum = minmax(couleur)

    if maximum - minimum < 0.5:
        return (maximum - minimum) / (maximum + minimum)
    return (maximum - minimum) / (2 - maximum - minimum)

@njit
def ajuster_saturation(couleur, facteur):
    """
    Par interpolation lineaire
    facteur: 0 completement desaturé
             1 couleur de base
             > 1 saturé
    """
    luma = calcul_luminosite(couleur)

    return np.clip(luma + (couleur - luma) * facteur, 0.0, 255.0)

@njit
def rgb_to_hsv(couleur):
    min, max = minmax(couleur)
    difference = max - min

    couleur_hsv = np.empty(couleur.shape, dtype=np.float32)

    # Hue (Teinte) (entre 0=0 et 1=360)
    if difference == 0:
        couleur_hsv[0] = 0
    elif max == couleur[0]:
        couleur_hsv[0] = 60 * (((couleur[1] - couleur[2])/difference) % 6) / 360
    elif max == couleur[1]:
        couleur_hsv[0] = 60 * ((couleur[2] - couleur[0])/difference + 2) / 360
    else:
        couleur_hsv[0] = 60 * ((couleur[0] - couleur[1])/difference + 4) / 360

    # Saturation
    if max == 0:
        couleur_hsv[1] = 0
    else:
        couleur_hsv[1] = difference / max

    # Value
    couleur_hsv[2] = max
    return couleur_hsv

@njit
def hsv_to_rgb(couleur):
    C = couleur[1] * couleur[2]
    H = couleur[0] * 360
    X = C * (1 - np.abs((H / 60)  % 2 - 1))

    couleur_rgb = np.empty(couleur.shape, dtype=np.float32)
    if H < 60:
        couleur_rgb[0], couleur_rgb[1], couleur_rgb[2] = C, X, 0
    elif H < 120:
        couleur_rgb[0], couleur_rgb[1], couleur_rgb[2] = X, C, 0
    elif H < 180:
        couleur_rgb[0], couleur_rgb[1], couleur_rgb[2] = 0, C, X
    elif H < 240:
        couleur_rgb[0], couleur_rgb[1], couleur_rgb[2] = 0, X, C
    elif H < 300:
        couleur_rgb[0], couleur_rgb[1], couleur_rgb[2] = X, 0, C
    else:
        couleur_rgb[0], couleur_rgb[1], couleur_rgb[2] = X, 0, C


    return couleur_rgb


if __name__ == '__main__':
    couleur = np.array([0,0.5,0], dtype=np.float32)

    print(calcul_luminosite(couleur), ajuster_luminosite(couleur, 0.5))
    print(calcul_saturation(couleur), ajuster_saturation(couleur, 1.5))

    print(rgb_to_hsv(couleur), hsv_to_rgb(rgb_to_hsv(couleur)))