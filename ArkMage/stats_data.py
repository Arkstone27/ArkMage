import numpy as np

import plotly.graph_objects as go

from numba import njit, prange, types
from numba.experimental.jitclass.decorators import jitclass

spec = [
    ("data", types.float32[:, :, :]),
    ("nb_pixels", types.int32),
    ("n_canaux", types.uint8),
    ("histogramme", types.float32[:, :]),
    ("ecart_data", types.float32),
    ("ajout_min", types.float32),

    ("variance", types.float32[:]),
    ("ecart_type", types.float32[:]),
    ("moyenne", types.float32[:]),

    ("minimum", types.float32[:]),
    ("maximum", types.float32[:]),

    ("centiles", types.float32[:, :])
]

@jitclass(spec)
class Stats_Data(object):
    def __init__(self, data, normaliser: bool=False):
        self.minimum = np.empty(0, dtype=np.float32)
        self.maximum = np.empty(0, dtype=np.float32)

        self.data = data
        if normaliser:
            self.ecart_data, self.ajout_min = 1, 0
        else:
            self.data, self.ecart_data = norm_min_max(data, self.get_max(), self.get_min())
            self.ajout_min = self.get_min()

        w, h, n = data.shape
        self.nb_pixels, self.n_canaux = w*h, n
        self.histogramme = np.empty((0, 0), dtype=np.float32)

        self.moyenne = np.empty(0, dtype=np.float32)
        self.variance = np.empty(0, dtype=np.float32)
        self.ecart_type = np.empty(0, dtype=np.float32)

        self.centiles = np.empty((0,0), dtype=np.float32)

    def get_histogramme(self):
        if self.histogramme.size == 0:
            self.histogramme = calcul_histogramme(self.data)
        return self.histogramme

    def get_moyenne(self):
        if self.moyenne.size == 0:
            self.moyenne = calcul_moyenne(self.get_histogramme(), self.nb_pixels, self.ecart_data, self.ajout_min)
        return self.moyenne
    def get_variance(self):
        if self.variance.size == 0:
            self.variance = calcul_variance(self.get_histogramme(), self.nb_pixels, self.get_moyenne(), self.ecart_data)
        return self.variance
    def get_ecart_type(self):
        if self.ecart_type.size == 0:
            self.ecart_type = np.sqrt(self.get_variance())
        return self.ecart_type

    def get_min(self):
        if self.minimum.size == 0:
            self.minimum, self.maximum = calcul_min_max(self.data)
        return self.minimum
    def get_max(self):
        if self.maximum.size == 0:
            self.minimum, self.maximum = calcul_min_max(self.data)
        return self.maximum

    def get_centiles(self):
        if self.centiles.size == 0:
            self.centiles = calcul_centiles(self.get_histogramme(), self.nb_pixels, self.ecart_data, self.ajout_min)
        return self.centiles

    def get_centile(self, indice_centile):
        if indice_centile == 0:
            return np.zeros(self.n_canaux, dtype=np.float32)
        return self.get_centiles()[:, indice_centile-1]

    def get_quartiles(self):
        return self.get_centile(25), self.get_centile(50), self.get_centile(75)
    def get_quartile(self, indice_quartile):
        return self.get_quartiles()[indice_quartile-1]
    def get_mediane(self):
        return self.get_quartile(2-1)
    def get_iqr(self):
        # Ecart interquartile
        return self.get_quartile(2) - self.get_quartile(1)

@njit(parallel=True)
def norm_min_max(data, maximum, minimum):
    w, h, n_canaux = data.shape
    data_normaliser = np.empty(data.shape, dtype=np.float32)

    ecart = maximum - minimum
    for c in range(n_canaux):
        if ecart[c] == 0: ecart[c] += 0.00001

    for x in prange(w):
        for y in range(h):
            data_normaliser[x, y] = (data[x, y] - minimum) / ecart
    return data_normaliser, ecart

@njit(parallel=True)
def calcul_histogramme(data):
    w, h, n_canaux = data.shape
    nb_barres = 256
    indice_max = nb_barres - 1

    sub_histogrammes = np.zeros((w, nb_barres, n_canaux), dtype=np.uint32)

    for x in prange(w):
        for y in range(h):
            valeur = data[x, y]

            for c in range(n_canaux):
                indice_barre = int(valeur[c] * indice_max)
                sub_histogrammes[x, indice_barre, c] += 1

    histogramme = np.zeros((nb_barres, n_canaux), dtype=np.float32)
    for x in range(w):
        histogramme += sub_histogrammes[x]

    return histogramme

@njit
def calcul_moyenne(histogramme, nb_pixels, ecart_data, ajout_min):
    nb_barres, n_canaux = histogramme.shape
    facteur_echelle = ecart_data / (nb_barres - 1)

    moyenne = np.zeros(n_canaux, dtype=np.float32)

    for idx_barre in range(nb_barres):
        barre = histogramme[idx_barre]
        valeur_barre = idx_barre * facteur_echelle + ajout_min

        for c in range(n_canaux):
            moyenne[c] += barre[c] * valeur_barre

    return moyenne / nb_pixels

@njit
def calcul_variance(histogramme, nb_pixels, moyenne, ecart_data):
    nb_barres, n_canaux = histogramme.shape
    facteur_echelle = ecart_data / (nb_barres - 1)

    variance = np.zeros(n_canaux, dtype=np.float32)

    for idx_barre in range(nb_barres):
        barre = histogramme[idx_barre]
        valeur_barre = idx_barre * facteur_echelle

        for c in range(n_canaux):
            ecart = (valeur_barre - moyenne[c])**2
            variance[c] += ecart * barre[c]
    return variance / nb_pixels

@njit(parallel=True)
def calcul_min_max(data):
    w, h, n_canaux = data.shape

    minimum = data[0, 0].copy()
    maximum = data[0, 0].copy()

    for x in prange(w):
        for y in range(h):
            valeur = data[x, y]

            for c in range(n_canaux):
                if valeur[c] < minimum[c]:
                    minimum[c] = valeur[c]
                if valeur[c] > maximum[c]:
                    maximum[c] = valeur[c]

    return minimum, maximum

@njit
def calcul_centiles(histogramme, nb_pixels, ecart_data, ajout_min):
    nb_barres, n_canaux = histogramme.shape
    facteur_echelle = ecart_data / (nb_barres - 1)

    centiles, valeur_centile = np.empty((n_canaux, 100), dtype=np.float32), int(nb_pixels / 100)
    indices_centiles, sommes = np.zeros(n_canaux, dtype=np.uint8), np.zeros(n_canaux, dtype=np.uint32)

    for idx_barre in range(nb_barres):
        barre = histogramme[idx_barre]
        valeur_barre = idx_barre * facteur_echelle + ajout_min

        for c in range(n_canaux):
            sommes[c] += barre[c]
            indice_centile = indices_centiles[c]

            while valeur_centile*(indice_centile + 1) < sommes[c] and indice_centile < 100:
                centiles[c, indice_centile] = valeur_barre
                indice_centile += 1
            indices_centiles[c] = indice_centile
    return centiles

def generer_graphe_central(stats_image):
    histogramme = stats_image.get_histogramme()
    moyenne = stats_image.get_moyenne()
    ecart_type = stats_image.get_ecart_type()

    nb_barres, n_canaux = histogramme.shape
    x = np.linspace(0.0, 1.0, nb_barres)
    figure_central = go.Figure()

    if n_canaux == 1:
        infos_canaux = ((0, "Intensité", (100,100,100)),)
    else:
        infos_canaux = ((0, "Rouge - R", (255,105,97)), (1, "Vert - G", (119,221,119)), (2, "Bleu - B", (132,182,244)))

    for indice_canal, nom, couleur in infos_canaux:
        couleur_fond = f"rgba{couleur+(0.5,)}"
        couleur = f"rgba{couleur+(1,)}"

        id_groupe = f"groupe_{indice_canal}"

        # Histogramme
        figure_central.add_trace(go.Scatter(x=x, y=histogramme[:, indice_canal], legendgroup=id_groupe,
                                            mode="lines", name=nom, line=dict(color=couleur, width=1.5), fill="tozeroy", fillcolor=couleur_fond))

        # Moyenne & Ecart-Type
        figure_central.add_trace(go.Scatter(x=[moyenne[indice_canal], moyenne[indice_canal]], y=[0, np.max(histogramme)],
                                            mode="lines", name=f"Moyenne {nom}", line=dict(color=couleur, width=1.5),
                                            legendgroup=id_groupe, showlegend=False, hovertemplate=(f"Moyenne : %{{x:.5f}}<br>"
                                                                                           +f"Écart-type : {ecart_type[indice_canal]:.5f}"
                                                                                           +"<extra></extra>")))


    figure_central.update_layout(title="Histogramme de l'image", hovermode="x unified",
                                 xaxis=dict(title=f"Intensité [0, 1]", range=[0.0, 1.0],dtick=0.01),
                                 yaxis=dict(title="Nombre de pixels"))
    return figure_central

def afficher_stats(stats_image):
    figure_central = generer_graphe_central(stats_image)
    # generer_indicateurs(stats_data)
    figure_central.show()

if __name__ == '__main__':
    data = np.random.rand(500, 500, 3).astype(np.float32) * 100
    stats_data = Stats_Data(data)

    stats_data.get_histogramme()

    print("Moyenne", stats_data.get_moyenne())
    print("Variance", stats_data.get_variance())
    print("Ecart_type", stats_data.get_ecart_type())

    print("Minimum", stats_data.get_min())
    print("Maximum", stats_data.get_max())

    print("Ecart Interquartile",stats_data.get_iqr())
    print("95centile", stats_data.get_centile(95))
