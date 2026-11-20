"""
Extraction de tous les descripteurs d'une image et construction du vecteur
global utilisé pour l'indexation et la recherche.

Organisation du vecteur (181 dimensions) en 5 « groupes » de caractéristiques :

    groupe            dimensions   type
    ---------------   ----------   --------
    color_hist             128     couleur
    color_moments            9     couleur
    dominant_colors         20     couleur
    glcm                    14     texture
    lbp                     10     texture

Les groupes sont utilisés par la distance (une distance par groupe, puis une
combinaison pondérée) et par les méthodes de retour de pertinence.
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from . import color, texture

MAX_SIDE = 256  # les images sont réduites (côté max) pour un calcul rapide et homogène

FEATURE_GROUPS = [
    ("color_hist", 128),
    ("color_moments", 9),
    ("dominant_colors", 20),
    ("glcm", 14),
    ("lbp", 10),
]
FEATURE_DIM = sum(n for _, n in FEATURE_GROUPS)


def group_slices() -> dict[str, slice]:
    """Position de chaque groupe dans le vecteur global."""
    out, start = {}, 0
    for name, n in FEATURE_GROUPS:
        out[name] = slice(start, start + n)
        start += n
    return out


GROUP_SLICES = group_slices()


class InvalidImage(ValueError):
    """Fichier illisible ou qui n'est pas une image."""


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImage("Le fichier n'est pas une image lisible.")
    return img


def load_image(path: str | Path) -> np.ndarray:
    return decode_image(Path(path).read_bytes())


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """Réduit l'image pour que son plus grand côté fasse au plus MAX_SIDE pixels."""
    h, w = bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return bgr


def extract(bgr: np.ndarray, types: str = "all", details: bool = False) -> dict:
    """Calcule les descripteurs demandés.

    types : "all", "color" ou "texture".
    Retourne un dictionnaire lisible (pour l'affichage) et, si types == "all",
    le vecteur global ``vector`` (pour l'indexation).
    details=True ajoute, pour l'affichage, les histogrammes R, G, B et gris,
    l'histogramme LBP complet (256 motifs) et l'image LBP.
    """
    img = preprocess(bgr)
    out: dict = {"image_size": {"width": int(bgr.shape[1]), "height": int(bgr.shape[0])}}

    if types in ("all", "color"):
        hist = color.color_histogram(img)
        moments = color.color_moments(img)
        dom = color.dominant_colors(img)
        out["color"] = {
            "histogram": {
                "bins": list(color.HSV_BINS),
                "space": "HSV",
                "values": hist.round(6).tolist(),
                # Projections marginales, plus faciles à lire qu'un histogramme 3D
                "hue": hist.reshape(color.HSV_BINS).sum(axis=(1, 2)).round(6).tolist(),
                "saturation": hist.reshape(color.HSV_BINS).sum(axis=(0, 2)).round(6).tolist(),
                "value": hist.reshape(color.HSV_BINS).sum(axis=(0, 1)).round(6).tolist(),
            },
            "moments": moments,
            "dominant_colors": dom,
        }
        if details:
            out["color"]["rgb_histograms"] = {"bins": color.RGB_BINS, **color.rgb_histograms(img)}

    if types in ("all", "texture"):
        glcm = texture.glcm_features(img)
        lbp = texture.lbp_histogram(img)
        out["texture"] = {
            "glcm": {
                "levels": texture.GLCM_LEVELS,
                "distances": list(texture.GLCM_DISTANCES),
                "angles_deg": [0, 45, 90, 135],
                "features": glcm,
            },
            "lbp": {"P": texture.LBP_P, "R": texture.LBP_R, "method": "uniform",
                    "histogram": lbp.round(6).tolist()},
        }
        if details:
            full, codes = texture.lbp_full(img)
            ok, png = cv2.imencode(".png", codes)
            out["texture"]["lbp_full"] = {
                "P": 8, "R": 1, "method": "default",
                "histogram": full.round(6).tolist(),
                "image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else None,
            }

    if types == "all":
        vec = np.concatenate([
            hist,
            color.color_moments_vector(moments),
            color.dominant_colors_vector(dom),
            texture.glcm_vector(glcm),
            lbp,
        ])
        assert vec.shape[0] == FEATURE_DIM
        out["vector"] = vec.tolist()
        out["groups"] = [{"name": n, "dim": d} for n, d in FEATURE_GROUPS]
    return out


def feature_vector(bgr: np.ndarray) -> np.ndarray:
    """Vecteur global (181 valeurs) d'une image."""
    return np.asarray(extract(bgr, "all")["vector"], dtype=np.float32)
