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


if __name__ == '__main__':
    couleur = np.array([0.5,0.2,0.1], dtype=np.float32)

    print(calcul_luminosite(couleur), ajuster_luminosite(couleur, 0.5))
    print(calcul_saturation(couleur), ajuster_saturation(couleur, 1.5))