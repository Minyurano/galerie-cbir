"""
Normalisation des vecteurs et mesure de distance pondérée.

Distance utilisée (équations 3, 13 et 14 de l'article) :

    d_m(x, y) = sqrt( Σ_{i ∈ m} w_i (x_i - y_i)^2  /  Σ_{i ∈ m} w_i )   (par groupe m)
    D(x, y)   = Σ_m e_m · d_m(x, y)          avec  Σ_m e_m = 1

- w_i : poids de la composante i (tous égaux à 1 sans retour de pertinence ;
        on retrouve alors la distance euclidienne normalisée de l'équation 1)
- e_m : importance du groupe m (couleur, texture...), égale à 1/M par défaut

Une distance par groupe évite qu'un groupe à 128 dimensions (histogramme)
écrase un groupe à 9 dimensions (moments).
"""
from __future__ import annotations

import numpy as np

from ..descriptors.extractor import FEATURE_DIM, FEATURE_GROUPS, GROUP_SLICES

GROUP_NAMES = [n for n, _ in FEATURE_GROUPS]
M = len(GROUP_NAMES)
STD_FLOOR = 1e-3


class Normalizer:
    """Normalisation z-score : (x - moyenne) / écart-type, par composante.

    Les statistiques sont calculées sur toute la collection. Elles mettent
    chaque composante à la même échelle (une dimension à grandes valeurs ne
    domine plus la distance). Les histogrammes passent d'abord par une racine
    carrée (transformation de Hellinger), qui réduit l'effet des cases très
    remplies et rend la distance euclidienne plus adaptée aux histogrammes.
    """

    SQRT_GROUPS = ("color_hist", "lbp")

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    @classmethod
    def _transform(cls, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float32, copy=True)
        for g in cls.SQRT_GROUPS:
            s = GROUP_SLICES[g]
            X[..., s] = np.sqrt(np.clip(X[..., s], 0, None))
        return X

    def fit(self, X: np.ndarray) -> "Normalizer":
        T = self._transform(X)
        self.mean = T.mean(axis=0)
        self.std = np.maximum(T.std(axis=0), STD_FLOOR)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (self._transform(X) - self.mean) / self.std


def default_weights() -> tuple[np.ndarray, np.ndarray]:
    """Poids unitaires (aucune préférence) : w_i = 1, e_m = 1/M."""
    return np.ones(FEATURE_DIM, dtype=np.float32), np.full(M, 1.0 / M, dtype=np.float32)


def group_distances(Xn: np.ndarray, q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Distance de chaque image à la requête, groupe par groupe -> (N, M)."""
    diff2 = (Xn - q) ** 2 * w
    out = np.empty((Xn.shape[0], M), dtype=np.float32)
    for j, name in enumerate(GROUP_NAMES):
        s = GROUP_SLICES[name]
        out[:, j] = np.sqrt(diff2[:, s].sum(axis=1) / max(w[s].sum(), 1e-12))
    return out


def weighted_distance(Xn: np.ndarray, q: np.ndarray, w: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Distance globale D = Σ_m e_m d_m (équation 13)."""
    return group_distances(Xn, q, w) @ e
