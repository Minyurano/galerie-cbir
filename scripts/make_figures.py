"""
Génère les figures du rapport à partir de data/eval/results.json.

    python scripts/make_figures.py [--results data/eval/results.json] [--out docs/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Palette catégorielle validée (ordre fixe) ; la recherche simple est une
# référence neutre (gris, pointillés), pas une série.
COL = {"qvm": "#2a78d6", "fre": "#eb6834", "type1": "#1baf7a", "type1plus": "#8a5cd1", "simple": "#7a7a75"}
MARK = {"qvm": "o", "fre": "s", "type1": "^", "type1plus": "D"}
NAME = {"qvm": "QVM", "fre": "FRE", "type1": "Re-weighting Type-1", "type1plus": "Type-1 amélioré", "simple": "Recherche simple"}
INK, MUTED, GRID = "#1d2433", "#5f6676", "#e6e8ee"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def _lines(ax, R, noise, key, ylabel):
    it = list(range(R["iterations"] + 1))
    base = R["runs"]["simple@0.0"][key][0] * 100
    ax.axhline(base, color=COL["simple"], ls="--", lw=1.2, label=NAME["simple"])
    ends = []
    for m in ("qvm", "fre", "type1", "type1plus"):
        y = [v * 100 for v in R["runs"][f"{m}@{noise:.1f}"][key]]
        ax.plot(it, y, color=COL[m], marker=MARK[m], ms=5, lw=2, label=NAME[m])
        ends.append([y[-1], y[-1], COL[m]])
    # Étiquettes de fin de courbe : écartées verticalement quand deux valeurs sont proches
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.06
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):
        ends[k][1] = max(ends[k][1], ends[k - 1][1] + gap)
    for v, pos, col in ends:
        ax.annotate(f"{v:.1f}", (it[-1], v), xytext=(it[-1] + 0.12, pos), textcoords="data",
                    va="center", fontsize=8, color=col, fontweight="bold")
    ax.set_xticks(it, ["Simple"] + [f"It. {i}" for i in it[1:]])
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.2, it[-1] + 0.6)


def save(fig, base: Path) -> None:
    """PDF pour le rapport LaTeX, PNG pour le README GitHub."""
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(config.DATA_DIR / "eval" / "results.json"))
    ap.add_argument("--out", default=str(config.BASE_DIR / "docs" / "figures"))
    args = ap.parse_args()
    R = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Précision@25 et mAP selon les itérations (utilisateur parfait puis bruité)
    for noise, tag in ((0.0, "parfait"), (0.2, "bruit20")):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        _lines(axes[0], R, noise, "P25", "Précision@25 (%)")
        _lines(axes[1], R, noise, "mAP", "mAP (%)")
        axes[0].legend(loc="lower right", fontsize=7.5)
        fig.tight_layout()
        save(fig, out / f"rf_iterations_{tag}")
        plt.close(fig)

    # 2. Effet du bruit de l'utilisateur (5e itération)
    noises = sorted({r["noise"] for r in R["runs"].values() if r["method"] != "simple"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, key, lab in ((axes[0], "P25", "Précision@25 (%) — itération 5"), (axes[1], "mAP", "mAP (%) — itération 5")):
        ax.axhline(R["runs"]["simple@0.0"][key][0] * 100, color=COL["simple"], ls="--", lw=1.2, label=NAME["simple"])
        for m in ("qvm", "fre", "type1", "type1plus"):
            ax.plot([n * 100 for n in noises], [R["runs"][f"{m}@{n:.1f}"][key][-1] * 100 for n in noises],
                    color=COL[m], marker=MARK[m], ms=5, lw=2, label=NAME[m])
        ax.set_xlabel("Erreurs de jugement de l'utilisateur (%)")
        ax.set_ylabel(lab)
        ax.set_xticks([n * 100 for n in noises])
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, out / "rf_bruit")
    plt.close(fig)

    # 3. Ablation : apport de chaque groupe de descripteurs (recherche simple)
    names = {"color_hist": "Histogramme HSV", "color_moments": "Moments de couleur",
             "dominant_colors": "Couleurs dominantes", "glcm": "GLCM (Haralick)", "lbp": "LBP",
             "couleur": "Couleur (3 groupes)", "texture": "Texture (2 groupes)"}
    ab = R["ablation"]
    keys = list(names)
    vals = [ab[k]["P@25"] * 100 for k in keys] + [R["runs"]["simple@0.0"]["P25"][0] * 100]
    labels = [names[k] for k in keys] + ["Tous les descripteurs"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    colors = [COL["qvm"]] * len(keys) + ["#184f95"]
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.62)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Précision@25 (%) — recherche simple")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    save(fig, out / "ablation")
    plt.close(fig)

    # 4. Précision par classe (recherche simple vs Type-1 bruit 20 %, itération 5)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ghim_utils import GHIM_CLASSES
    pc_s, pc_t = R["per_class"]["simple@0.0"], R["per_class"]["type1@0.2"]
    order = sorted(range(20), key=lambda c: pc_s[c])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = range(20)
    ax.barh([i + 0.2 for i in y], [pc_s[c] * 100 for c in order], height=0.38, color=COL["simple"], label=NAME["simple"])
    ax.barh([i - 0.2 for i in y], [pc_t[c] * 100 for c in order], height=0.38, color=COL["type1"],
            label="Type-1, 5 itérations, 20 % d'erreurs")
    ax.set_yticks(list(y), [f"{GHIM_CLASSES[c][1]} ({c})" for c in order], fontsize=7.5)
    ax.set_xlabel("Précision@25 (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, out / "par_classe")
    plt.close(fig)
    print(f"Figures -> {out}")


if __name__ == "__main__":
    main()
