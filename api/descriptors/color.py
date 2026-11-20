"""
Descripteurs de COULEUR.

    1. Histogramme de couleur HSV quantifié (8 x 4 x 4 = 128 cases)
    2. Moments de couleur (moyenne, écart-type, asymétrie) sur H, S, V -> 9 valeurs
    3. Couleurs dominantes (K-means, K = 5) dans l'espace Lab        -> 5 x 4 valeurs

Toutes les fonctions reçoivent une image OpenCV en BGR (uint8).
"""
from __future__ import annotations

import cv2
import numpy as np

HSV_BINS = (8, 4, 4)          # teinte, saturation, valeur
N_DOMINANT = 5                # nombre de couleurs dominantes
KMEANS_SAMPLE = 4000          # pixels échantillonnés pour accélérer K-means


def color_histogram(bgr: np.ndarray, bins=HSV_BINS) -> np.ndarray:
    """Histogramme HSV 3D normalisé (somme = 1), aplati en vecteur.

    L'espace HSV sépare la teinte (H) de l'intensité (V) : il est plus proche
    de la perception humaine que RGB et moins sensible aux variations d'éclairage.
    La teinte reçoit plus de cases (8) car c'est la composante la plus
    discriminante.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float64)
    total = hist.sum()
    return hist / total if total > 0 else hist


RGB_BINS = 64                 # cases des histogrammes par canal (affichage)


def rgb_histograms(bgr: np.ndarray, bins: int = RGB_BINS) -> dict:
    """Histogrammes 1D normalisés des canaux Rouge, Vert, Bleu et des niveaux de gris.

    Complètent l'histogramme HSV 3D : ils montrent directement la répartition
    de chaque couleur primaire (et de la luminosité) dans l'image.
    Utilisés pour l'affichage ; ils n'entrent pas dans le vecteur de recherche.
    """
    out = {}
    for idx, name in ((2, "R"), (1, "G"), (0, "B")):
        h = cv2.calcHist([bgr], [idx], None, [bins], [0, 256]).ravel().astype(np.float64)
        out[name] = (h / max(h.sum(), 1.0)).round(6).tolist()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype(np.float64)
    out["gray"] = (h / max(h.sum(), 1.0)).round(6).tolist()
    return out


def color_moments(bgr: np.ndarray) -> dict:
    """Moments d'ordre 1, 2 et 3 de chaque canal HSV (valeurs ramenées à [0, 1]).

    - moyenne       : couleur moyenne
    - écart-type    : dispersion des couleurs
    - asymétrie     : racine cubique du moment centré d'ordre 3 (signe conservé)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[..., 0] /= 179.0
    hsv[..., 1:] /= 255.0
    out = {}
    for idx, name in enumerate(("H", "S", "V")):
        ch = hsv[..., idx].ravel()
        mean = ch.mean()
        std = ch.std()
        third = np.mean((ch - mean) ** 3)
        skew = np.cbrt(third)
        out[name] = {"mean": float(mean), "std": float(std), "skewness": float(skew)}
    return out


def color_moments_vector(moments: dict) -> np.ndarray:
    return np.array(
        [moments[c][k] for c in ("H", "S", "V") for k in ("mean", "std", "skewness")],
        dtype=np.float64,
    )


def dominant_colors(bgr: np.ndarray, k: int = N_DOMINANT) -> list[dict]:
    """Couleurs dominantes par K-means dans l'espace Lab.

    Lab est perceptuellement uniforme : une même distance correspond à peu près
    à une même différence de couleur perçue, ce qui convient à K-means
    (distance euclidienne). Les couleurs sont triées par proportion décroissante.
    La graine est fixée pour que le résultat soit reproductible.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(0)
    if lab.shape[0] > KMEANS_SAMPLE:
        lab = lab[rng.choice(lab.shape[0], KMEANS_SAMPLE, replace=False)]
    k = min(k, len(np.unique(lab, axis=0)))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k).astype(np.float64)
    props = counts / counts.sum()
    order = np.argsort(-props)

    colors = []
    for i in order:
        lab_px = np.uint8([[centers[i]]])
        r, g, b = cv2.cvtColor(lab_px, cv2.COLOR_LAB2RGB)[0, 0].tolist()
        L, a, bb = centers[i].tolist()
        colors.append(
            {
                "rgb": [int(r), int(g), int(b)],
                "hex": "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b)),
                # Lab OpenCV 8 bits : L, a, b dans [0, 255]
                "lab": [float(L), float(a), float(bb)],
                "proportion": float(props[i]),
            }
        )
    # Complète si l'image contient moins de k couleurs distinctes.
    while len(colors) < N_DOMINANT:
        colors.append({"rgb": [0, 0, 0], "hex": "#000000", "lab": [0.0, 128.0, 128.0], "proportion": 0.0})
    return colors


def dominant_colors_vector(colors: list[dict]) -> np.ndarray:
    """[L, a, b, proportion] x K, valeurs ramenées à [0, 1]."""
    vec = []
    for c in colors[:N_DOMINANT]:
        L, a, b = c["lab"]
        vec.extend([L / 255.0, a / 255.0, b / 255.0, c["proportion"]])
    return np.array(vec, dtype=np.float64)
