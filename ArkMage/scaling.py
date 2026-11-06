from stats_data import Stats_Data
from image import appliquer_fast

import numpy as np
from numba import njit

def norm_min_max(data=None, stats_data: Stats_Data=None):
    if data is None:
        data = stats_data.data
    elif stats_data is None:
        stats_data = Stats_Data(data)

    n_canaux = data.shape[2]
    minimum, maximum = stats_data.get_min(), stats_data.get_max()
    ecart = maximum - minimum

    for c in range(n_canaux):
        if ecart[c] == 0: ecart[c] += 0.00001

    @njit
    def fonction_min_max(valeur):
        return (valeur - minimum) / ecart

    return appliquer_fast(data, fonction_min_max, data.shape)

def stand_score_z(data=None, stats_data: Stats_Data=None):
    if data is None:
        data = stats_data.data
    elif stats_data is None:
        stats_data = Stats_Data(data)

    n_canaux = data.shape[2]
    moyenne, ecart_type = stats_data.get_moyenne(), stats_data.get_ecart_type().copy()

    for c in range(n_canaux):
        if ecart_type[c] == 0: ecart_type[c] += 0.00001

    @njit
    def fonction_score_z(valeur):
        return (valeur - moyenne) / ecart_type
    return appliquer_fast(data, fonction_score_z, data.shape)

def norm_robuste(data=None, stats_data: Stats_Data=None):
    if data is None:
        data = stats_data.data
    elif stats_data is None:
        stats_data = Stats_Data(data)

    n_canaux = data.shape[2]
    medianne, ecart_interquartile = stats_data.get_mediane(), stats_data.get_iqr().copy()

    for c in range(n_canaux):
        if ecart_interquartile[c] == 0: ecart_interquartile[c] += 0.00001

    @njit
    def fonction_robuste(valeur):
        return (valeur - medianne) / ecart_interquartile
    return appliquer_fast(data, fonction_robuste, data.shape)

def winsorisation(data=None, stats_data: Stats_Data=None, min_centile=5, max_centile=95):
    if data is None:
        data = stats_data.data
    elif stats_data is None:
        stats_data = Stats_Data(data)
    a_max = stats_data.get_moyenne() + stats_data.get_ecart_type()
    data_couper = np.clip(data, stats_data.get_centile(min_centile), stats_data.get_centile(max_centile))
    return norm_min_max(data_couper)

def meilleur_winsorisation(data=None, stats_data: Stats_Data=None):
    if data is None:
        data = stats_data.data
    elif stats_data is None:
        stats_data = Stats_Data(data)

    a_max = stats_data.get_moyenne() + stats_data.get_ecart_type()
    data_couper = np.clip(data, stats_data.get_min(), a_max)
    return norm_min_max(data_couper)

if __name__ == '__main__':
    data = np.random.rand(2, 2, 3).astype(np.float32)

    data_normaliser = norm_min_max(data)
    print("Min-Max:\n", data_normaliser)

    data_normaliser = stand_score_z(data)
    print("Z-Score:\n", data_normaliser)

    data_normaliser = norm_robuste(data)
    print("Robuste:\n", data_normaliser)

    data_normaliser = winsorisation(data, min_centile=1, max_centile=99)
    print("Winsorisation:\n", data_normaliser)

    data_normaliser = meilleur_winsorisation(data)
    print("Meilleur Winsorisation:\n", data_normaliser)