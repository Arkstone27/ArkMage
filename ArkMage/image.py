from PIL import Image
import plotly.express as px

from PySide6.QtWidgets import QFileDialog, QApplication
import sys

import numpy as np
from numba import njit, prange

from ArkMage.stats_data import Stats_Data, afficher_stats
from ArkMage.rotation import rotation_data

@njit()
def fonction_base(valeur, *args):
    return valeur

class Pixels:
    def __init__(self, data, mode_couleur="RGB"):
        if data.ndim == 2:
            self.data = data[:, :, np.newaxis].astype(np.float32, copy=False)
        else:
            self.data = data.astype(np.float32, copy=False)

        self.mode_couleur = mode_couleur
        self.table_sommes = None
        self.stats = Stats_Data(self.data, True)

        dimensions = self.data.shape
        self.w, self.h = dimensions[0], dimensions[1]

    @classmethod
    def img_to_pixels(cls, img):
        data = np.array(img).astype(np.float32)

        return cls(data / 255.0, img.mode) # Chaque valeur est comprise entre 0 et 1

    @classmethod
    def creation_vide(cls, dimensions=(32,32), couleur=(255,255,255), mode_couleur="RGB"):
        img = Image.new(mode_couleur, dimensions, couleur)

        return cls.img_to_pixels(img)

    def conversion_img(self):
        if self.mode_couleur == "B" or self.mode_couleur == "L":
            img_data = np.squeeze(self.data) * 255.0
            mode_couleur = "L"
        else:
            img_data = np.clip(self.data * 255.0, 0, 255)
            mode_couleur = self.mode_couleur

        img_data = img_data.astype(np.uint8)
        return Image.fromarray(img_data, mode=mode_couleur)

    def __getitem__(self, position: tuple):
        return self.data[position[0], position[1]]

    def __setitem__(self, position: tuple, valeur):
        self.data[position[0], position[1]] = valeur
        self.table_sommes = None

    def __len__(self):
        return self.w * self.h

    def appliquer(self, fonction, new_mode: str = None):
        if new_mode is None:
            new_mode = self.mode_couleur
        new_data = appliquer_fast(self.data, fonction, (self.w, self.h, {"RGBA": 4, "RGB": 3}.get(new_mode, 1)))

        return Pixels(new_data, new_mode)

    def convolution(self, noyau, fonction=fonction_base):
        decalage_x, decalage_y = noyau.shape[0] // 2, noyau.shape[1] // 2
        data_pad = np.pad(self.data, [(decalage_x, decalage_x), (decalage_y, decalage_y), (0, 0)], mode="edge")
        new_data = convolution_fast(data_pad, noyau, self.data.shape, fonction)

        return Pixels(new_data, self.mode_couleur)

    def show(self):
        fig = px.imshow(self.conversion_img(), color_continuous_scale='gray')
        fig.show()


    def sauvegarde(self):
        app = QApplication(sys.argv)
        chemin_sauvegarde = QFileDialog.getOpenFileName(None, "Selectionner une image", "", "Images (*.png *.jpg *.jpeg)")[0]

        if chemin_sauvegarde:
            self.conversion_img().save(chemin_sauvegarde)

    def coller(self, x: int, y: int, other):
        assert self.w - x > other.w and self.h - y > other.h, \
            f"Il n'y a pas la place : {x}, {y} | {self.w}, {self.h} | {other.w}, {other.h}"

        self.data[x:x+other.w, y:y+other.h] = other.data

    def get_table_sommes(self):
        if self.table_sommes is None:
            self.table_sommes = creation_table_sommes(self.data)
        return self.table_sommes

    def rotation(self, angle: int):
        data_tourner = rotation_data(self.data, angle)

        return Pixels(data_tourner, self.mode_couleur)

def charger_image(chemin: str):
    img = Image.open(chemin)
    if img.mode == "P": # P = Palette
        img = img.convert("RGBA")

    if img.mode == "RGBA":
        plan = Image.new("RGB", img.size, (255,255,255))
        plan.paste(img, mask=img.split()[3])

        return plan
    return img

def ouvrir_image(image_test=False):
    if image_test:
        chemin = "C:\\Users\Matteo\Documents\Projets\ArkBrick\sources\socle\\test.jpg"
    else:
        app = QApplication(sys.argv)
        chemin = QFileDialog.getOpenFileName(None, "Selectionner une image", "", "Images (*.png *.jpg *.jpeg)")[0]

    img = charger_image(chemin)
    return Pixels.img_to_pixels(img)

@njit(parallel=True) # On fait une compilation just in time
def appliquer_fast(data, fonction, dimensions: tuple):
    w, h = dimensions[0], dimensions[1]
    new_data = np.empty(dimensions, dtype=np.float32)

    for x in prange(w):
        for y in range(h):
            new_data[x, y] = fonction(data[x, y])
    return new_data

@njit()
def calcul_somme(fenetre, noyau):
    somme = 0.0 * fenetre[0, 0]
    w, h = fenetre.shape[0], fenetre.shape[1]

    for x in range(w):
        for y in range(h):
            somme += fenetre[x, y] * noyau[x, y]
    return somme

@njit(parallel=True) # Calcule en parallèle sur tous les processeurs
def convolution_fast(data, noyau, dimensions, fonction=fonction_base):
    w, h = dimensions[0], dimensions[1]
    new_data = np.empty(dimensions, dtype=np.float32)

    decalage_x, decalage_y = noyau.shape[0] // 2, noyau.shape[1] // 2
    for x in prange(decalage_x, w+decalage_x):
        for y in range(decalage_y, h+decalage_y):
            fenetre = data[x-decalage_x:x+decalage_x+1, y-decalage_y:y+decalage_y+1]

            somme = calcul_somme(fenetre, noyau)
            new_data[x-decalage_x, y-decalage_y] = fonction(somme, noyau)

    return new_data

def convolution_vecteurs(data, vecteurs, fonction=fonction_base):
    vecteur_1, vecteur_2 = vecteurs
    decalage_x, decalage_y = vecteur_1.shape[0] // 2, vecteur_2.shape[1] // 2

    data_pad = np.pad(data, [(decalage_x, decalage_x), (0, 0), (0, 0)], mode="edge")
    data_x = convolution_fast(data_pad, vecteur_1, data.shape, fonction_base)

    data_pad = np.pad(data_x, [(0, 0), (decalage_y, decalage_y), (0, 0)], mode="edge")
    new_data = convolution_fast(data_pad, vecteur_2, data.shape, fonction)

    return new_data

@njit()
def creation_table_sommes(data):
    dimensions = data.shape
    w, h, n_canaux = dimensions[0], dimensions[1], dimensions[2]

    table_sommes = np.empty(dimensions, dtype=np.float64)

    for c in range(n_canaux):
        table_sommes[0, 0, c] = data[0, 0, c]

        # Ligne horizontale
        for x in range(1, w):
            table_sommes[x, 0, c] = table_sommes[x-1, 0, c] + data[x, 0, c]

        # Ligne verticale
        for y in range(1, h):
            table_sommes[0, y, c] = table_sommes[0, y-1, c] + data[0, y, c]

        # Calcul du reste
        for x in range(1, w):
            for y in range(1, h):
                table_sommes[x, y, c] = table_sommes[x-1, y, c] + data[x, y, c] + table_sommes[x, y-1, c] - table_sommes[x-1, y-1, c]
    return table_sommes

if __name__ == '__main__':

    pixels = ouvrir_image(False)

    pixels_tourner = pixels.rotation(270)
    pixels_tourner.show()

    pixels.convolution(np.ones((3,3)), fonction_base)

    rectangle_rouge = Pixels.creation_vide((56,45),(255,0,0),"RGB")
    pixels.coller(0, 0, rectangle_rouge)

    afficher_stats(pixels.stats)
    
    pixels.show()
