"""
Lecture de la base GHIM-10K, quelle que soit son organisation :

  1. version « plate » : toutes les images dans un seul dossier, nommées
     « <classe>_<numéro>.jpg » (ex. 6_3007.jpg) ;
  2. version Kaggle : un sous-dossier par classe (aircraft, buildings, ...).

Les deux versions sont ramenées aux mêmes numéros de classe (0 à 19).
"""
from __future__ import annotations

from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Numéro de classe -> (thème, nom français)
GHIM_CLASSES = {
    0: ("Événements", "Feux d'artifice"),
    1: ("Architecture", "Édifices européens"),
    2: ("Architecture", "Grande Muraille"),
    3: ("Transports", "Voitures"),
    4: ("Animaux", "Libellules"),
    5: ("Nature", "Montagnes enneigées"),
    6: ("Nature", "Fleurs"),
    7: ("Nature", "Arbres"),
    8: ("Nature", "Prairies et champs"),
    9: ("Nature", "Lacs et rivages"),
    10: ("Transports", "Avions"),
    11: ("Animaux", "Papillons"),
    12: ("Architecture", "Architecture chinoise"),
    13: ("Nature", "Couchers de soleil"),
    14: ("Transports", "Motos"),
    15: ("Transports", "Voiliers"),
    16: ("Transports", "Navires de croisière"),
    17: ("Animaux", "Poules"),
    18: ("Animaux", "Insectes"),
    19: ("Animaux", "Chevaux"),
}

# Nom des sous-dossiers de la version Kaggle -> numéro de classe
KAGGLE_FOLDERS = {
    "fireworks": 0, "buildings": 1, "walls": 2, "cars": 3, "dragon fly": 4, "dragonfly": 4,
    "mountains": 5, "flowers": 6, "trees": 7, "valleys": 8, "sea shores": 9, "seashores": 9,
    "aircraft": 10, "butterfly": 11, "temples": 12, "sunset": 13, "motorcycles": 14,
    "yacht": 15, "ships": 16, "hen": 17, "insects": 18, "horses": 19,
}


def list_ghim(ghim_dir: str | Path) -> list[tuple[Path, int]]:
    """Liste (chemin de l'image, numéro de classe), triée par classe puis par nom."""
    root = Path(ghim_dir)
    if not root.is_dir():
        return []
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    out: list[tuple[Path, int]] = []

    if subdirs:  # version Kaggle : un dossier par classe
        next_id = max(GHIM_CLASSES) + 1
        for d in subdirs:
            key = d.name.strip().lower()
            if key not in KAGGLE_FOLDERS:          # dossier inconnu : nouvelle classe
                KAGGLE_FOLDERS[key] = next_id
                GHIM_CLASSES[next_id] = ("Autres", d.name)
                next_id += 1
            cls = KAGGLE_FOLDERS[key]
            out += [(f, cls) for f in sorted(d.iterdir()) if f.suffix.lower() in IMAGE_EXT]
    else:        # version plate : « classe_numéro.jpg »
        for f in root.iterdir():
            if f.suffix.lower() in IMAGE_EXT and "_" in f.stem and f.stem.split("_")[0].isdigit():
                out.append((f, int(f.stem.split("_")[0])))

    def order(item):
        f, c = item
        tail = f.stem.split("_")[-1]
        return (c, int(tail) if tail.isdigit() else 0, f.name)
    return sorted(out, key=order)
