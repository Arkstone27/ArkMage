from ArkMage.image import Pixels, ouvrir_image

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def resize_simple_fast(data, new_w, new_h):
    dimensions = data.shape
    w, h = dimensions[0], dimensions[1]
    data_resize = np.empty((new_w, new_h, dimensions[2]), dtype=np.float32)

    facteur_x, facteur_y = w / new_w, h / new_h

    for x in prange(new_w):
        for y in range(new_h):
            x_reel, y_reel = x * facteur_x + 0.5, y * facteur_y + 0.5

            if x_reel >= w: x_reel = w - 1
            if y_reel >= h: y_reel = h - 1

            x_int = int(x_reel)
            if x_reel - x_int > 0.5 and x_int < w - 1:
                x_int += 1

            y_int = int(y_reel)
            if y_reel - y_int > 0.5 and y_int < h - 1:
                y_int += 1

            data_resize[x, y] = data[x_int, y_int]
    return data_resize

def resize_simple(pixels: Pixels, new_w: int | float, new_h: int | float):
    """
    On prend juste le pixel voisin le plus proche
    """
    data_resize = resize_simple_fast(pixels.data, int(new_h), int(new_w))

    return Pixels(data_resize, pixels.mode_couleur)

@njit(parallel=True)
def resize_moyenne_fast(table_sommes, new_w: int, new_h: int):
    dimensions = table_sommes.shape
    w, h, n_canaux = dimensions[0], dimensions[1], dimensions[2]
    data_resize = np.empty((new_w, new_h, dimensions[2]), dtype=np.float32)

    facteur_x, facteur_y = w / new_w, h / new_h
    rayon_x, rayon_y = int(facteur_x) + 1, int(facteur_y) + 1

    for x in prange(new_w):
        for y in range(new_h):
            x_voisin, y_voisin = int(x * facteur_x + 0.5), int(y * facteur_y + 0.5)

            x1, y1 = max(0, x_voisin - rayon_x), max(0, y_voisin - rayon_y)
            x2, y2 = min(w-1, x_voisin + rayon_x), min(h-1, y_voisin + rayon_y)

            nb_pixels = (x2 - x1 + 1)*(y2 - y1 + 1)
            x1, y1 = x1 - 1, y1 - 1

            for c in range(n_canaux):
                somme = table_sommes[x2, y2, c]

                if x1 >= 0:
                    somme -= table_sommes[x1, y2, c]

                if y1 >= 0:
                    somme -= table_sommes[x2, y1, c]

                if x1 >= 0 and y1 >= 0:
                    somme += table_sommes[x1, y1, c]
                data_resize[x, y, c] = somme / nb_pixels

    return data_resize

def resize_moyenne(pixels: Pixels, new_w: int | float, new_h: int | float):
    """
    On prend une moyenne autour du pixel voisin
    """

    data_resize = resize_moyenne_fast(pixels.get_table_sommes(), int(new_h), int(new_w))

    return Pixels(data_resize, pixels.mode_couleur)

@njit(parallel=True)
def resize_bilineaire_fast(data_pad, new_w, new_h, dimensions):
    w, h = dimensions[0], dimensions[1]
    data_resize = np.empty((new_w, new_h, dimensions[2]), dtype=np.float32)

    facteur_x, facteur_y = w / new_w, h / new_h

    for x in prange(new_w):
        for y in range(new_h):
            x_reel, y_reel = x * facteur_x + 0.5, y * facteur_y + 0.5

            if x_reel >= w-1: x_reel = w - 2
            if y_reel >= h-1: y_reel = h - 2

            x_int, y_int = int(x_reel), int(y_reel)
            poids_x2 = x_reel - x_int

            valeur_y1 = data_pad[x_int, y_int] * (1-poids_x2) + data_pad[x_int + 1, y_int] * poids_x2
            valeur_y2 = data_pad[x_int, y_int+1] * (1 - poids_x2) + data_pad[x_int + 1, y_int+1] * poids_x2

            poids_y2 = y_reel - y_int
            data_resize[x, y] = valeur_y1 * (1-poids_y2) + valeur_y2 * poids_y2
    return data_resize

def resize_bilineaire(pixels: Pixels, new_w: int | float, new_h: int | float):
    new_h, new_w = int(new_w), int(new_h)
    data_pad = np.pad(pixels.data, [(1, 1), (1, 1), (0, 0)], mode="edge")

    data_resize = resize_bilineaire_fast(pixels.data, new_w, new_h, pixels.data.shape)
    return Pixels(data_resize, pixels.mode_couleur)

def conservation_ratio(new_w: int, new_h: int, pixels: Pixels):
    w, h, n_canaux = pixels.data.shape
    ratio = w / h

    if new_w == 0:
        return round(new_h / ratio), new_h
    return new_w, round(ratio * new_w)

if __name__ == '__main__':
    pixels = ouvrir_image(False)

    facteur = 10
    new_h, new_w = pixels.w*facteur, pixels.h*facteur
    pixels_resize = resize_simple(pixels, new_w, new_h)
    pixels_resize.show()

    pixels_resize = resize_moyenne(pixels, new_w, new_h)
    pixels_resize.show()

    pixels_resize = resize_bilineaire(pixels, new_w, new_h)
    pixels_resize.show()
