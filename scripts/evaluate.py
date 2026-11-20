"""
Évaluation expérimentale sur GHIM-10K (utilisateur simulé).

Protocole
---------
- Requêtes : NQ images par classe (tirées au hasard, graine fixe), 20 classes.
- Vérité terrain : une image est pertinente si elle a la même classe que la requête.
- Scope = 25 images renvoyées par itération (comme dans l'article).
- Utilisateur simulé : il juge les 25 images affichées (pertinente / non
  pertinente) selon la classe ; avec un niveau de BRUIT b, chaque jugement est
  inversé avec la probabilité b (un vrai utilisateur se trompe).
- Méthodes : recherche simple, QVM, FRE, Re-weighting Type-1 ; 5 itérations.
- Mesures : précision@25 (P@25, celle de l'article), précision moyenne (mAP)
  sur toute la base, tests de Wilcoxon appariés.

Usage :
    python scripts/evaluate.py [--features data/eval/ghim_features.npz] [--nq 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from api.retrieval.distance import GROUP_NAMES, Normalizer, default_weights, weighted_distance  # noqa: E402
from api.retrieval.relevance_feedback import apply_feedback, new_state, rank  # noqa: E402

SCOPE = 25
ITERS = 5
METHODS = ["qvm", "fre", "type1"]


def average_precision(order: np.ndarray, rel_mask: np.ndarray, exclude: int) -> float:
    """AP sur le classement complet (la requête elle-même est exclue)."""
    order = order[order != exclude]
    hits = rel_mask[order]
    if hits.sum() == 0:
        return 0.0
    cum = np.cumsum(hits)
    ranks = np.arange(1, len(order) + 1)
    return float((cum[hits] / ranks[hits]).mean())


def precision_at(order: np.ndarray, rel_mask: np.ndarray, k: int) -> float:
    return float(rel_mask[order[:k]].mean())


def run(Xn, labels, queries, method, noise, rng, iters=ITERS):
    """Retourne P@25 et AP par requête et par itération : tableaux (Q, iters+1)."""
    P = np.zeros((len(queries), iters + 1))
    AP = np.zeros((len(queries), iters + 1))
    for qi, q in enumerate(queries):
        rel_mask = labels == labels[q]
        st = new_state(Xn[q], SCOPE, q)
        st.relevant.add(q)
        order, _ = rank(Xn, st)
        st.last_retrieved = order[:SCOPE].tolist()
        P[qi, 0] = precision_at(order, rel_mask, SCOPE)
        AP[qi, 0] = average_precision(order, rel_mask, q)
        for it in range(1, iters + 1):
            if method == "simple":
                P[qi, it], AP[qi, it] = P[qi, 0], AP[qi, 0]
                continue
            shown = order[:SCOPE]
            truth = rel_mask[shown]
            flip = rng.random(len(shown)) < noise
            judged = np.where(flip, ~truth, truth)
            rel = [int(i) for i, j in zip(shown, judged) if j and i != q]
            non = [int(i) for i, j in zip(shown, judged) if not j and i != q]
            apply_feedback(Xn, st, method, rel, non)
            order, _ = rank(Xn, st)
            st.last_retrieved = order[:SCOPE].tolist()
            P[qi, it] = precision_at(order, rel_mask, SCOPE)
            AP[qi, it] = average_precision(order, rel_mask, q)
    return P, AP


def ablation(Xn, labels, queries):
    """mAP de la recherche simple avec un seul groupe de descripteurs."""
    out = {}
    w, _ = default_weights()
    for j, g in enumerate(GROUP_NAMES + ["couleur", "texture"]):
        e = np.zeros(len(GROUP_NAMES), dtype=np.float32)
        if g == "couleur":
            e[:3] = 1 / 3
        elif g == "texture":
            e[3:] = 0.5
        else:
            e[j] = 1.0
        aps, ps = [], []
        for q in queries:
            order = np.argsort(weighted_distance(Xn, Xn[q], w, e), kind="stable")
            m = labels == labels[q]
            aps.append(average_precision(order, m, q))
            ps.append(precision_at(order, m, SCOPE))
        out[g] = {"mAP": float(np.mean(aps)), "P@25": float(np.mean(ps))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(config.DATA_DIR / "eval" / "ghim_features.npz"))
    ap.add_argument("--nq", type=int, default=20, help="requêtes par classe")
    ap.add_argument("--out", default=str(config.DATA_DIR / "eval" / "results.json"))
    ap.add_argument("--noise", default="0,0.1,0.2,0.3")
    args = ap.parse_args()

    d = np.load(args.features)
    X, labels = d["X"], d["labels"]
    Xn = Normalizer().fit(X)(X).astype(np.float32)
    rng = np.random.default_rng(42)
    queries = np.concatenate([rng.choice(np.where(labels == c)[0], args.nq, replace=False)
                              for c in np.unique(labels)])
    print(f"{len(X)} images, {len(queries)} requêtes, scope={SCOPE}, {ITERS} itérations")

    res = {"n_images": int(len(X)), "n_queries": int(len(queries)), "scope": SCOPE, "iterations": ITERS,
           "ablation": ablation(Xn, labels, queries), "runs": {}, "per_class": {}, "wilcoxon": {}}
    print("Ablation (recherche simple) :", json.dumps(res["ablation"], indent=1))

    raw = {}
    for noise in [float(x) for x in args.noise.split(",")]:
        for m in ["simple"] + METHODS:
            if m == "simple" and noise > 0:
                continue
            t0 = time.time()
            P, AP = run(Xn, labels, queries, m, noise, np.random.default_rng(7))
            key = f"{m}@{noise:.1f}"
            raw[key] = (P, AP)
            res["runs"][key] = {"method": m, "noise": noise,
                                "P25": P.mean(0).round(4).tolist(), "mAP": AP.mean(0).round(4).tolist()}
            print(f"{key:12s} P@25 {np.round(P.mean(0) * 100, 1)}  mAP {np.round(AP.mean(0) * 100, 1)}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)

    # Précision par classe (recherche simple vs chaque méthode, 5e itération, bruit 0,2)
    for key in ["simple@0.0"] + [f"{m}@0.2" for m in METHODS]:
        P, _ = raw[key]
        res["per_class"][key] = [float(P[labels[queries] == c, -1].mean()) for c in np.unique(labels)]

    # Tests de Wilcoxon appariés (AP par requête, 5e itération)
    base_ap = raw["simple@0.0"][1][:, 0]
    for noise in [0.0, 0.2]:
        for m in METHODS:
            ap_m = raw[f"{m}@{noise:.1f}"][1][:, -1]
            res["wilcoxon"][f"{m}@{noise:.1f}_vs_simple"] = float(wilcoxon(ap_m, base_ap).pvalue)
        for a, b in [("fre", "qvm"), ("type1", "qvm"), ("type1", "fre")]:
            res["wilcoxon"][f"{a}_vs_{b}@{noise:.1f}"] = float(
                wilcoxon(raw[f"{a}@{noise:.1f}"][1][:, -1], raw[f"{b}@{noise:.1f}"][1][:, -1]).pvalue)
    print("Wilcoxon :", json.dumps(res["wilcoxon"], indent=1))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"Résultats -> {args.out}")


if __name__ == "__main__":
    main()
