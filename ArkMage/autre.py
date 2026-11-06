from time import perf_counter
import os
import webbrowser

import plotly.express as px

import numpy as np
from numba import njit
from numba.typed import List, Dict

def mesure_vitesse(fonction, arguments: tuple=None, iterations: int=1):
    debut = perf_counter()

    for _ in range(iterations):
        if arguments is None:
            resultat_fonction = fonction()
        else:
            resultat_fonction = fonction(*arguments)

    temps = perf_counter() - debut
    print(f"{fonction.__name__} a pris : {temps}s")

def afficher_fonction(fonction):
    intervalle = [x/10 for x in range(-1000,1000)]

    figure = px.line(x=intervalle, y=[fonction(x) for x in intervalle])
    figure.write_html("graphiques\\graphe_fonction.html")

    chemin_graphe = os.path.abspath("graphiques\\graphe_fonction.html")
    webbrowser.open(f"file://{chemin_graphe}")

@njit
def faire_dict(keys, values: List):
    dico = Dict()
    for i in range(len(keys)):
        dico[keys[i]] = values[i]
    return dico

def conversion_nb_dict(dico):
    keys = np.array(list(dico.keys()))
    values = list(dico.values())

    return faire_dict(keys, values)


if __name__ == '__main__':
    # mesure_vitesse(lambda x: [i for i in range(x)], (10000,), 1000)
    #
    # afficher_fonction(lambda x: x*x)
    dico = {"a": 50, "zo": 40}
    print(conversion_nb_dict(dico))