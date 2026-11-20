"""
Méthodes de retour de pertinence (Relevance Feedback, RF), d'après :

    Renuka Devi SM, « Reweighing Strategies for Relevance Feedback in CBIR »,
    IJARCCE, vol. 5, n° 11, 2016 (tableau 1 de l'article).

    Méthode                         Requête à l'itération k+1          Poids w_i^{k+1}
    -----------------------------   ---------------------------------  -----------------------------
    Query Vector Modification (QVM) α X^k + β moy(R) - γ moy(N)  (2)   1 (distance euclidienne normalisée)
    Feature Relevance Est. (FRE)    moyenne des images pertinentes     1 / (ε + σ_rel,i)            (4)
    Re-weighting Type-1             moyenne des images pertinentes     (ε + σ_Nr,i) / (ε + σ_rel,i) (5)

Notations :
    R, N      ensembles des images jugées pertinentes / non pertinentes
    σ_rel,i   écart-type de la composante i sur les images pertinentes
    σ_Nr,i    écart-type de la composante i sur les N_r images retournées à l'itération k
    ε         petite constante (0,0001 dans l'article)

Idée commune : une composante qui varie PEU entre les images pertinentes est
ce qu'elles ont en commun ; elle doit donc compter PLUS dans la distance.

Choix d'implémentation (documentés dans le rapport) :
  * les vecteurs sont comparés dans l'espace normalisé (z-score) ;
  * la requête initiale fait toujours partie des images pertinentes ;
  * les jugements sont cumulés d'une itération à l'autre (l'utilisateur ne
    perd pas ce qu'il a déjà indiqué) ;
  * FRE : les poids w_i sont normalisés dans chaque groupe (moyenne 1) puis
    bornés (RF_MAX_WEIGHT) : avec peu d'exemples, un écart-type presque nul
    donnerait sinon un poids démesuré (1/ε = 10 000) ;
  * FRE : l'importance e_m de chaque groupe est proportionnelle au nombre
    d'images pertinentes retrouvées avec ce groupe seul (|R_m|, section III-B) ;
  * Type-1 (« type1 ») : STRICTEMENT l'équation 5 de l'article ; tous les
    descripteurs gardent la même importance.
  * Type-1 amélioré (« type1plus », extension du groupe G11, hors article) :
    poids de l'équation 5 normalisés et plafonnés dans chaque groupe, et
    importance des groupes selon l'équation 14 (e_m = W_m / Σ W_l).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from ..descriptors.extractor import GROUP_SLICES
from .distance import GROUP_NAMES, M, default_weights, group_distances, weighted_distance

METHODS = {
    "qvm": {
        "name": "Query Vector Modification (QVM)",
        "groups": ["G3", "G7", "G9", "G12", "G15"],
        "description": "Déplace le vecteur requête vers les images pertinentes et l'éloigne des "
                       "images non pertinentes (formule de Rocchio). Les poids restent égaux.",
    },
    "fre": {
        "name": "Feature Relevance Estimation (FRE / MARS)",
        "groups": ["G1", "G5", "G10", "G13", "G14", "G16"],
        "description": "La requête devient la moyenne des images pertinentes ; chaque composante "
                       "reçoit le poids 1/(ε+σ_rel) et chaque groupe une importance proportionnelle "
                       "au nombre d'images pertinentes qu'il retrouve seul.",
    },
    "type1": {
        "name": "Re-weighting Type-1",
        "groups": ["G2", "G4", "G6", "G8", "G11"],
        "description": "Méthode de l'article (section III-C, équation 5) : la requête devient la "
                       "moyenne des images pertinentes ; chaque composante reçoit le poids "
                       "(ε+σ_Nr)/(ε+σ_rel), fort si elle varie peu chez les pertinentes mais "
                       "beaucoup parmi les images retournées.",
    },
    "type1plus": {
        "name": "Re-weighting Type-1 amélioré",
        "groups": [],
        "description": "Extension proposée par le groupe G11 : poids de l'équation 5 normalisés "
                       "et plafonnés dans chaque descripteur, plus une importance par descripteur "
                       "(équation 14). Plus stable avec peu d'exemples pertinents.",
    },
}


@dataclass
class RFState:
    """État d'une session de recherche (mémoire à court terme)."""
    q0: np.ndarray                      # requête initiale (normalisée)
    q: np.ndarray                       # requête courante
    w: np.ndarray                       # poids des composantes
    e: np.ndarray                       # importance des groupes
    scope: int
    iteration: int = 0
    relevant: set = field(default_factory=set)       # indices (dans l'index) jugés pertinents
    non_relevant: set = field(default_factory=set)   # indices jugés non pertinents
    last_retrieved: list = field(default_factory=list)
    query_index: int | None = None      # indice de la requête si elle est dans la base


def new_state(q0: np.ndarray, scope: int, query_index: int | None = None) -> RFState:
    w, e = default_weights()
    return RFState(q0=q0.copy(), q=q0.copy(), w=w, e=e, scope=scope, query_index=query_index)


def rank(Xn: np.ndarray, state: RFState) -> tuple[np.ndarray, np.ndarray]:
    """Classe toute la collection par distance croissante à la requête courante."""
    d = weighted_distance(Xn, state.q, state.w, state.e)
    order = np.argsort(d, kind="stable")
    return order, d


# --------------------------------------------------------------------------- #
# Outils
# --------------------------------------------------------------------------- #
def _normalize_within_groups(raw: np.ndarray, max_w: float) -> np.ndarray:
    """Moyenne 1 dans chaque groupe, puis plafonnement à max_w et re-normalisation."""
    w = raw.astype(np.float64).copy()
    for name in GROUP_NAMES:
        s = GROUP_SLICES[name]
        g = w[s]
        g = g / max(g.mean(), 1e-12)
        g = np.minimum(g, max_w)
        w[s] = g / max(g.mean(), 1e-12)
    return w.astype(np.float32)


def _relevant_matrix(Xn: np.ndarray, state: RFState) -> np.ndarray:
    """Vecteurs des images pertinentes, requête initiale comprise."""
    rows = [state.q0[None, :]]
    idx = sorted(state.relevant - ({state.query_index} if state.query_index is not None else set()))
    if idx:
        rows.append(Xn[idx])
    return np.vstack(rows)


# --------------------------------------------------------------------------- #
# Les trois méthodes de l'article
# --------------------------------------------------------------------------- #
def _qvm(Xn, state, rel_now, non_now):
    """Équation 2 : X^{k+1} = α X^k + β moy(R) - γ moy(N)."""
    q = config.QVM_ALPHA * state.q
    if rel_now:
        q = q + config.QVM_BETA * Xn[rel_now].mean(axis=0)
    if non_now:
        q = q - config.QVM_GAMMA * Xn[non_now].mean(axis=0)
    # Les coefficients ne sont pas normalisés dans l'article : on ramène la
    # requête à l'échelle d'un vecteur de la base en divisant par Σ des coefficients positifs.
    q = q / (config.QVM_ALPHA + (config.QVM_BETA if rel_now else 0.0))
    w, e = default_weights()   # poids unitaires : distance euclidienne normalisée
    return q.astype(np.float32), w, e


def _fre(Xn, state, rel_now, non_now):
    """FRE (MARS) : requête = moyenne des pertinentes, w_i = 1 / (ε + σ_rel,i)  (éq. 4)."""
    R = _relevant_matrix(Xn, state)
    q = R.mean(axis=0)
    sigma_rel = R.std(axis=0)
    w = _normalize_within_groups(1.0 / (config.RF_EPSILON + sigma_rel), config.RF_MAX_WEIGHT)

    # Importance des groupes : |R_m| = nombre d'images pertinentes connues
    # retrouvées dans les « scope » premières en utilisant le groupe m seul.
    known_rel = np.array(sorted(state.relevant), dtype=int)
    gd = group_distances(Xn, q, w)
    counts = np.zeros(M)
    for j in range(M):
        top = np.argpartition(gd[:, j], min(state.scope, len(gd) - 1))[: state.scope]
        counts[j] = np.isin(top, known_rel).sum() if known_rel.size else 0
    e = (counts + 1.0) / (counts + 1.0).sum()     # +1 : aucun groupe n'est totalement ignoré
    return q.astype(np.float32), w, e.astype(np.float32)


def _type1(Xn, state, rel_now, non_now):
    """Re-weighting Type-1 de l'article (section III-C), sans ajout.

    - requête  : moyenne des images pertinentes (requête initiale comprise) ;
    - poids    : w_i = (ε + σ_Nr,i) / (ε + σ_rel,i)   (équation 5, ε = 0,0001)
                 σ_Nr,i  : écart-type de la composante i sur les N_r images retournées
                 σ_rel,i : écart-type de la composante i sur les images pertinentes ;
    - distance : euclidienne normalisée pondérée par w_i ; les descripteurs gardent
                 la même importance (pas d'équation 14, propre au Type-3).
    """
    R = _relevant_matrix(Xn, state)
    q = R.mean(axis=0)
    sigma_rel = R.std(axis=0)
    retrieved = state.last_retrieved[: state.scope]
    sigma_nr = Xn[retrieved].std(axis=0) if len(retrieved) > 1 else np.ones_like(sigma_rel)
    w = (config.RF_EPSILON + sigma_nr) / (config.RF_EPSILON + sigma_rel)
    _, e = default_weights()
    return q.astype(np.float32), w.astype(np.float32), e


def _type1plus(Xn, state, rel_now, non_now):
    """Type-1 amélioré (extension G11) : équation 5 + normalisation/plafond + équation 14."""
    R = _relevant_matrix(Xn, state)
    q = R.mean(axis=0)
    sigma_rel = R.std(axis=0)
    retrieved = state.last_retrieved[: state.scope]
    sigma_nr = Xn[retrieved].std(axis=0) if len(retrieved) > 1 else np.ones_like(sigma_rel)
    raw = (config.RF_EPSILON + sigma_nr) / (config.RF_EPSILON + sigma_rel)

    # Équation 14 : importance d'un groupe = poids total du groupe / poids total.
    # On utilise la moyenne (bornée) des poids bruts du groupe pour ne pas
    # favoriser les groupes qui ont beaucoup de composantes.
    W = np.array([np.minimum(raw[GROUP_SLICES[g]], 1.0 / config.RF_EPSILON).mean() for g in GROUP_NAMES])
    W = np.log1p(W)            # atténue les écarts extrêmes entre groupes
    e = W / W.sum()
    w = _normalize_within_groups(raw, config.RF_MAX_WEIGHT)
    return q.astype(np.float32), w, e.astype(np.float32)


_IMPL = {"qvm": _qvm, "fre": _fre, "type1": _type1, "type1plus": _type1plus}


def apply_feedback(Xn: np.ndarray, state: RFState, method: str,
                   relevant: list[int], non_relevant: list[int]) -> RFState:
    """Met à jour la requête et les poids à partir des jugements de l'utilisateur.

    relevant / non_relevant : indices (dans l'index) jugés à cette itération.
    """
    if method not in _IMPL:
        raise ValueError(f"Méthode inconnue : {method}. Choix : {', '.join(_IMPL)}")
    rel_now = [i for i in dict.fromkeys(relevant) if 0 <= i < len(Xn)]
    non_now = [i for i in dict.fromkeys(non_relevant) if 0 <= i < len(Xn) and i not in rel_now]
    state.relevant |= set(rel_now)
    state.non_relevant |= set(non_now)
    state.non_relevant -= state.relevant
    state.q, state.w, state.e = _IMPL[method](Xn, state, rel_now, non_now)
    state.iteration += 1
    return state
