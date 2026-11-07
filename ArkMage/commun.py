from ArkMage.image import Pixels, ouvrir_image, convolution_vecteurs

from numba import njit, prange
import numpy as np

@njit
def calcul_distance(valeur_1, valeur_2):
    distance = 0.0

    for c in range(len(valeur_1)):
        distance += (valeur_1[c] - valeur_2[c])**2

    return distance

@njit(parallel=True)
def flou_classique_fast(table_sommes, rayon: int):
    dimensions = table_sommes.shape
    w, h, n_canaux = dimensions[0], dimensions[1], dimensions[2]

    data_flou = np.empty(dimensions, dtype=np.float32)

    for x in prange(w):
        for y in range(h):
            # Coordonnées des 2 coins
            x1, y1 =  max(x - rayon, 0), max(y - rayon, 0)
            x2, y2 = min(x + rayon, w - 1), min(y + rayon, h - 1)

            nb_pixels = (x2 - x1) * (y2- y1)

            for c in range(n_canaux):
                somme = table_sommes[x2, y2, c]

                if x1 >= 0:
                    somme -= table_sommes[x1, y2, c]

                if y1 >= 0:
                    somme -= table_sommes[x2, y1, c]

                if x1 >= 0 and y1 >= 0:
                    somme += table_sommes[x1, y1, c]

                data_flou[x, y, c] = somme / nb_pixels

    return data_flou

def flou_classique(pixels: Pixels, rayon: int=1):
    data_flou = flou_classique_fast(pixels.get_table_sommes(), rayon)

    return Pixels(data_flou, pixels.mode_couleur)

@njit
def creation_vecteurs_gaussian(rayon: int=1):
    taille = rayon*2+1
    vecteur_x = np.zeros((taille, 1), dtype=np.float32)
    vecteur_y = np.zeros((1, taille), dtype=np.float32)

    variancex2 = rayon * rayon * 2
    for x in range(taille):
        i = x - rayon
        valeur = np.exp(-(i*i) / variancex2)

        vecteur_x[x, 0] = valeur
        vecteur_y[0, x] = valeur
    return vecteur_x / vecteur_x.sum(), vecteur_y / vecteur_y.sum()

def flou_gaussian_data(data, rayon=1):
    vecteurs_gaussian = creation_vecteurs_gaussian(rayon)
    data_flou = convolution_vecteurs(data, vecteurs_gaussian)

    return data_flou

def flou_gaussian(pixels, rayon=1):
    data_flou = flou_gaussian_data(pixels.data, rayon)

    return Pixels(data_flou, pixels.mode_couleur)

@njit
def teinte_de_gris(couleur):
    return couleur[0]*0.2126 + couleur[1]*0.7152 + couleur[2]*0.0722

def conversion_gris(pixels):
    if pixels.mode_couleur in ("RGB","RGBA"):
        return pixels.appliquer(teinte_de_gris, "L")
    return pixels

@njit
def denoising_fast(data, table_sommes, rayon: int=1):
    data_flou = flou_classique_fast(table_sommes, rayon)
    dimensions = data.shape
    w, h, n_canaux = dimensions

    distance_max = calcul_distance(np.zeros(n_canaux,dtype=np.float32), np.ones(n_canaux,dtype=np.float32))
    data_denoise = np.empty(dimensions, dtype=np.float32)

    for x in prange(w):
        for y in range(h):
            valeur, valeur_flou = data[x, y], data_flou[x, y]
            poids = calcul_distance(valeur, valeur_flou) / distance_max

            for c in range(n_canaux):
                data_denoise[x, y, c] = valeur[c]*poids + valeur_flou[c]*(1-poids)

    return data_denoise

def denoising(pixels, rayon: int=1):
    data_denoise = denoising_fast(pixels.data, pixels.get_table_sommes(), rayon)

    return Pixels(data_denoise, pixels.mode_couleur)

if __name__ == '__main__':
    pixels = ouvrir_image()
    pixels.show()

    pixels_flou = flou_classique(pixels, 2)
    pixels_flou.show()

    pixels_flou = flou_gaussian(pixels, 2)
    pixels_flou.show()

    pixels_gris = conversion_gris(pixels)
    pixels_gris.show()

    pixels_denoise = denoising(pixels)
    pixels_denoise.show()
