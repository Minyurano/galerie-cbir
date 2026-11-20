"""
Extrait le vecteur de descripteurs (181 valeurs) de toutes les images GHIM-10K
et le sauvegarde dans un fichier .npz (utilisé par scripts/evaluate.py).

Usage :
    python scripts/extract_features.py [--ghim CHEMIN] [--out data/eval/ghim_features.npz] [--workers 2]
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from api.descriptors.extractor import feature_vector, load_image  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ghim_utils import list_ghim  # noqa: E402


def _one(path: str):
    try:
        return feature_vector(load_image(path))
    except Exception as exc:  # image corrompue
        print(f"[!] {path}: {exc}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghim", default=str(config.GHIM_DIR))
    ap.add_argument("--out", default=str(config.DATA_DIR / "eval" / "ghim_features.npz"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    items = list_ghim(args.ghim)              # version plate ou Kaggle (un dossier par classe)
    files = [f for f, _ in items]
    print(f"{len(files)} images trouvées dans {args.ghim}")
    t0 = time.time()
    with Pool(args.workers) as pool:
        vecs = []
        for i, v in enumerate(pool.imap(_one, [str(f) for f in files], chunksize=32), 1):
            vecs.append(v)
            if i % 500 == 0:
                print(f"  {i}/{len(files)}  ({time.time() - t0:.0f} s)", flush=True)
    keep = [i for i, v in enumerate(vecs) if v is not None]
    X = np.stack([vecs[i] for i in keep]).astype(np.float32)
    names = np.array([files[i].name for i in keep])
    labels = np.array([items[i][1] for i in keep])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, names=names, labels=labels)
    print(f"Terminé : {X.shape} en {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
