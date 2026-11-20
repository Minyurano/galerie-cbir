"""
Descripteurs de TEXTURE.

    1. Caractéristiques de Haralick issues de la matrice de co-occurrence des
       niveaux de gris (GLCM) : contraste, dissimilarité, homogénéité, énergie,
       corrélation, ASM et entropie, pour deux distances (1 et 2 pixels) et
       moyennées sur 4 orientations (0°, 45°, 90°, 135°)   -> 7 x 2 = 14 valeurs
    2. Histogramme des motifs binaires locaux (LBP uniformes, P = 8, R = 1)
                                                             -> 10 valeurs
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

GLCM_LEVELS = 32                       # quantification des niveaux de gris
GLCM_DISTANCES = (1, 2)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM", "entropy")
LBP_P, LBP_R = 8, 1
LBP_BINS = LBP_P + 2                   # motifs uniformes (P + 1) + classe « non uniforme »


def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def glcm_features(bgr: np.ndarray) -> dict:
    """Calcule les caractéristiques de Haralick.

    La GLCM P(i, j | d, θ) compte la fréquence à laquelle un pixel de niveau i
    a un voisin de niveau j à la distance d dans la direction θ. On la rend
    symétrique et normalisée (somme = 1). Moyenner sur les 4 directions rend
    le descripteur peu sensible à la rotation.

    Retour : {propriété: {"d1": valeur, "d2": valeur, "mean": valeur}}
    """
    gray = _gray(bgr)
    q = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    P = graycomatrix(q, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                     levels=GLCM_LEVELS, symmetric=True, normed=True)
    feats = {}
    for prop in GLCM_PROPS:
        if prop == "entropy":
            # Entropie = -Σ p log2 p (désordre de la texture), non fournie par graycoprops
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.nansum(P * np.log2(P, where=P > 0, out=np.zeros_like(P)), axis=(0, 1))
            values = ent  # shape (n_dist, n_angles)
        else:
            values = graycoprops(P, prop)
            if prop == "correlation":
                # Image uniforme : corrélation indéfinie -> 1 par convention
                values = np.nan_to_num(values, nan=1.0)
        per_dist = values.mean(axis=1)  # moyenne sur les angles
        feats[prop] = {f"d{d}": float(v) for d, v in zip(GLCM_DISTANCES, per_dist)}
        feats[prop]["mean"] = float(per_dist.mean())
    return feats


def glcm_vector(feats: dict) -> np.ndarray:
    return np.array([feats[p][f"d{d}"] for p in GLCM_PROPS for d in GLCM_DISTANCES], dtype=np.float64)


def lbp_histogram(bgr: np.ndarray) -> np.ndarray:
    """Histogramme normalisé des LBP uniformes.

    Pour chaque pixel, on compare ses P voisins (sur un cercle de rayon R) au
    pixel central : 1 si voisin >= centre, 0 sinon. Le code binaire obtenu
    décrit le micro-motif local (bord, coin, point, zone plate...). La variante
    « uniforme » est invariante à la rotation et ne garde que P + 2 classes.
    """
    gray = _gray(bgr)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS))
    hist = hist.astype(np.float64)
    return hist / hist.sum() if hist.sum() > 0 else hist


def lbp_full(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LBP « classique » à 256 motifs (P = 8, R = 1) : histogramme normalisé et image des codes.

    Chaque pixel reçoit un code de 0 à 255 (8 bits, un par voisin). L'image des
    codes rend la texture visible : zones plates, contours et grain apparaissent
    nettement. Utilisé pour l'affichage ; la recherche utilise les LBP uniformes.
    """
    gray = _gray(bgr)
    codes = local_binary_pattern(gray, 8, 1, method="default").astype(np.uint8)
    hist = np.bincount(codes.ravel(), minlength=256).astype(np.float64)
    return hist / max(hist.sum(), 1.0), codes
