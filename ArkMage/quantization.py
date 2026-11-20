import numpy as np
from numba import njit, prange

import subprocess
from PIL import Image

from ArkMage.commun import calcul_distance
from ArkMage.image import ouvrir_image, Pixels

@njit
def couleur_voisine(palette, couleur, indice_dernier, distance_voisin):
    # Renvoie l'indice de la couleur appartenant à palette qui est voisine à couleur (ainsi que leurs distance)
    # Une couleur est voisine sa distance avec couleur est plus petite que distante voisine

    for indice in range(indice_dernier):
        distance = calcul_distance(palette[indice], couleur)

        if calcul_distance(palette[indice], couleur) < distance_voisin:
            return indice, distance
    return indice_dernier, 0

@njit(parallel=True)
def preparation_data(data_pixels, nb_pixels_max=10000):
    # Aplatit la data et échantillone un nombre de pixels définie (pour éviter de faire trop de calcul inutile)
    w, h, n_canaux = data_pixels.shape
    nb_pixels = w*h
    if nb_pixels <= nb_pixels_max:
        ecart = 1
        nb_pixels_max = nb_pixels
    else:
        ecart = int(nb_pixels / nb_pixels_max)
        nb_pixels_max = int(nb_pixels/ecart)

    data_palette = np.zeros((nb_pixels_max, n_canaux), dtype=np.float32)
    for indice_pixel in prange(nb_pixels_max):
        indice = indice_pixel*ecart
        data_palette[indice_pixel] = data_pixels[indice%w, indice//w]
    return data_palette
        
@njit
def creation_palette(data, distance_voisin, nb_couleurs_max):
    # Ma version plus efficace du mean shift clustering
    nb_pixels, n_canaux = data.shape

    palette = np.empty((nb_couleurs_max, n_canaux), np.float32)
    indice_dernier = 0

    for indice in range(nb_pixels):
        couleur = data[indice]
        indice_palette, distance = couleur_voisine(palette, couleur, indice_dernier, distance_voisin)

        if indice_palette < indice_dernier:
            palette[indice_palette] = palette[indice_palette] * 0.5 + couleur * 0.5

        elif indice_palette < nb_couleurs_max:
            palette[indice_palette] = couleur
            indice_dernier += 1
    return palette[:indice_dernier]

def affichage_palette(palette):
    nb_couleurs, n_canaux = palette.shape
    data = np.empty((nb_couleurs*50, 50, n_canaux), dtype=np.float32)

    for indice_palette in range(nb_couleurs):
        data[indice_palette*50:indice_palette*50+50,:,:] = palette[indice_palette]

    Pixels(data, ("L", "NOP", "RGB", "RGBA")[n_canaux - 1]).show()

def quantization(pixels: Pixels, nb_couleurs_max=256, perte=0.05):

    # Plus la perte (0-1) est faible plus le nombre de couleurs est grand
    distance_voisin = perte * pixels.data.shape[2]
    palette = creation_palette(preparation_data(pixels.data), distance_voisin, nb_couleurs_max)

    data_indices = get_indices_palette(pixels.data, palette, distance_voisin/4)

    return data_indices, palette

@njit
def indice_plus_proche(couleur, palette, distance_voisin):
    # Renvoie la couleur dans la palette la plus proche de couleur
    # Distance voisin permet d'éviter de parcourir toute la palette pour chaque couleur
    
    distance_min, indice_min = 10, 0
    for indice in range(palette.shape[0]):
        distance = calcul_distance(palette[indice], couleur)

        if distance < distance_voisin:
            return indice
        elif distance < distance_min:
            distance_min = distance
            indice_min = indice
    return indice_min

@njit(parallel=True)
def get_indices_palette(data, palette, distance_voisin=0):
    w, h, n_canaux = data.shape

    data_indices = np.empty((w, h), dtype=np.uint8) # ATTENTION pas de palette > 256
    for x in prange(w):
        for y in range(h):
            data_indices[x, y] = indice_plus_proche(data[x, y], palette, distance_voisin)
    return data_indices


@njit(parallel=True)
def application_palette_fast(data_indices, palette):
    w, h = data_indices.shape

    data_palette = np.empty((w, h, palette.shape[1]), dtype=np.float32)
    for x in prange(w):
        for y in range(h):
            data_palette[x, y] = palette[data_indices[x, y]]
    return data_palette

def application_palette(indices_palette, palette):
    data_palette = application_palette_fast(indices_palette, palette)

    return Pixels(data_palette, pixels.mode_couleur)

def sauvegarde_indexer(data_indices, palette, chemin):
    img = Image.fromarray(data_indices, mode="P")
    img.putpalette((palette.flatten() * 255).astype(np.uint8))


    img = img.convert("RGB")

    img.save(chemin, lossless=True, optimize=True)

if __name__ == '__main__':
    pixels = ouvrir_image()
    
    data_indices, palette = quantization(pixels, perte=0.005)
    application_palette(data_indices, palette).show()

    sauvegarde_indexer(data_indices, palette, "ArkMage/test.jxl")