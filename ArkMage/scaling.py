from ArkMage.stats_data import Stats_Data, calcul_min_max
import ArkMage.stats_data as sd

import numpy as np
from numba import njit

@njit
def norm_min_max(stats_data: Stats_Data):
    data = stats_data.data

    n_canaux = data.shape[2]
    minimum, maximum = stats_data.get_min(), stats_data.get_max()
    ecart = maximum - minimum

    for c in range(n_canaux):
        if ecart[c] == 0: ecart[c] += 0.00001

    return (data - minimum) / ecart

@njit
def stand_score_z(stats_data: Stats_Data):
    data = stats_data.data

    n_canaux = data.shape[2]
    moyenne, ecart_type = stats_data.get_moyenne(), stats_data.get_ecart_type().copy()

    for c in range(n_canaux):
        if ecart_type[c] == 0: ecart_type[c] += 0.00001
    return (data - moyenne) / ecart_type

@njit
def norm_robuste(stats_data: Stats_Data):
    data = stats_data.data

    n_canaux = data.shape[2]
    medianne, ecart_interquartile = stats_data.get_mediane(), stats_data.get_iqr().copy()

    for c in range(n_canaux):
        if ecart_interquartile[c] == 0: ecart_interquartile[c] += 0.00001

    return (data - medianne) / ecart_interquartile

@njit
def winsorisation(stats_data: Stats_Data, min_centile=5, max_centile=95):
    data = stats_data.data

    a_max = stats_data.get_moyenne() + stats_data.get_ecart_type()
    data_couper = np.clip(data, stats_data.get_centile(min_centile), stats_data.get_centile(max_centile))

    minimum, maximum = calcul_min_max(data_couper)
    data_normaliser, _ = sd.norm_min_max(data_couper, maximum, minimum)
    return data_normaliser

@njit
def meilleur_winsorisation(stats_data: Stats_Data):
    data = stats_data.data

    a_max = stats_data.get_moyenne() + stats_data.get_ecart_type()
    data_couper = np.clip(data, stats_data.get_min(), a_max)

    minimum, maximum = calcul_min_max(data_couper)
    data_normaliser, _ = sd.norm_min_max(data_couper, maximum, minimum)
    return data_normaliser

if __name__ == '__main__':
    data = np.random.rand(2, 2, 3).astype(np.float32)
    stats_data = Stats_Data(data)
    # data_normaliser = norm_min_max(stats_data)
    # print("Min-Max:\n", data_normaliser)
    #
    # data_normaliser = stand_score_z(stats_data)
    # print("Z-Score:\n", data_normaliser)
    #
    # data_normaliser = norm_robuste(stats_data)
    # print("Robuste:\n", data_normaliser)

    data_normaliser = winsorisation(stats_data, min_centile=1, max_centile=99)
    print("Winsorisation:\n", data_normaliser)

    data_normaliser = meilleur_winsorisation(stats_data)
    print("Meilleur Winsorisation:\n", data_normaliser)