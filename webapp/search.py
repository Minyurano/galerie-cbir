"""
Recherche d'images similaires :
    - recherche simple (poids égaux) à partir d'une image de la galerie ou d'un fichier ;
    - recherche avec retour de pertinence (QVM, FRE ou Re-weighting Type-1) :
      l'utilisateur marque les résultats pertinents / non pertinents, l'API
      met à jour la requête et les poids, puis renvoie de nouveaux résultats.
"""
from __future__ import annotations

import uuid

from flask import Blueprint, abort, flash, redirect, render_template, request, send_file, session, url_for
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

import config
from . import imaging
from .api_client import APIError, api
from .models import Image, db

bp = Blueprint("search", __name__, url_prefix="/search")

METHOD_LABELS = {
    "qvm": "Query Vector Modification (QVM)",
    "fre": "Feature Relevance Estimation (FRE)",
    "type1": "Re-weighting Type-1",
}
QUERY_DIR = config.DATA_DIR / "queries"


def _scope() -> int:
    try:
        return max(5, min(int(request.values.get("scope", config.SCOPE)), 100))
    except (TypeError, ValueError):
        return config.SCOPE


def _method() -> str:
    m = request.values.get("method", config.DEFAULT_RF_METHOD)
    return m if m in METHOD_LABELS else config.DEFAULT_RF_METHOD


def _render(res: dict, state: dict):
    """Associe les résultats de l'API aux images de la base et affiche la page."""
    ids = [int(r["image_id"]) for r in res["results"]]
    imgs = {i.id: i for i in Image.query.filter(Image.user_id == current_user.id, Image.id.in_(ids)).all()}
    results = [dict(r, image=imgs.get(int(r["image_id"]))) for r in res["results"]]
    results = [r for r in results if r["image"] is not None]

    # Précision mesurée seulement si la classe de la requête est connue (base GHIM-10K)
    precision = None
    if state.get("query_label") is not None and results:
        hits = sum(1 for r in results if r["image"].label == state["query_label"])
        precision = hits / len(results)
    state["history"].append({"iteration": res["iteration"], "method": res["method"], "precision": precision})
    session["search"] = state

    query_img = db.session.get(Image, state["query_id"]) if state.get("query_id") else None
    return render_template("search/results.html", res=res, results=results, state=state,
                           query_img=query_img, precision=precision, methods=METHOD_LABELS)


@bp.route("/")
@login_required
def form():
    recent = (Image.query.filter_by(user_id=current_user.id).order_by(Image.id.desc()).limit(12).all())
    try:
        info = api.collection_info(current_user.collection)
    except APIError as exc:
        info = None
        flash(str(exc), "danger")
    return render_template("search/form.html", recent=recent, info=info, methods=METHOD_LABELS,
                           default_method=config.DEFAULT_RF_METHOD, scope=config.SCOPE)


@bp.route("/image/<int:image_id>")
@login_required
def by_image(image_id):
    img = db.session.get(Image, image_id)
    if img is None or img.user_id != current_user.id:
        abort(404)
    if not img.indexed:
        flash("Cette image n'est pas encore indexée : réindexez-la avant de lancer une recherche.", "warning")
        return redirect(url_for("gallery.detail", image_id=img.id))
    try:
        res = api.search_by_id(current_user.collection, img.id, _scope())
    except APIError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("gallery.detail", image_id=img.id))
    state = {"sid": res["session_id"], "query_id": img.id, "query_file": None, "scope": res["scope"],
             "method": _method(), "query_label": img.label, "history": []}
    return _render(res, state)


@bp.route("/upload", methods=["POST"])
@login_required
def by_upload():
    f = request.files.get("query")
    if not f or not f.filename or not imaging.allowed(secure_filename(f.filename)):
        flash("Choisissez une image requête valide.", "warning")
        return redirect(url_for(".form"))
    try:
        pil = imaging.open_checked(f.stream)
    except imaging.ImageError as exc:
        flash(str(exc), "danger")
        return redirect(url_for(".form"))
    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    token = f"{current_user.id}_{uuid.uuid4().hex}.jpg"
    pil.convert("RGB").save(QUERY_DIR / token, "JPEG", quality=95)
    try:
        res = api.search_by_file(current_user.collection, QUERY_DIR / token, _scope())
    except APIError as exc:
        flash(str(exc), "danger")
        return redirect(url_for(".form"))
    state = {"sid": res["session_id"], "query_id": None, "query_file": token, "scope": res["scope"],
             "method": _method(), "query_label": None, "history": []}
    return _render(res, state)


@bp.route("/query-file/<token>")
@login_required
def query_file(token):
    if not token.startswith(f"{current_user.id}_") or "/" in token or ".." in token:
        abort(404)
    path = QUERY_DIR / token
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/jpeg")


@bp.route("/feedback", methods=["POST"])
@login_required
def feedback():
    state = session.get("search")
    if not state or state.get("sid") != request.form.get("session_id"):
        flash("Session de recherche expirée : relancez la recherche.", "warning")
        return redirect(url_for(".form"))
    rel = [int(i) for i in request.form.getlist("relevant") if i.isdigit()]
    non = [int(i) for i in request.form.getlist("non_relevant") if i.isdigit()]
    method = _method()
    if not rel and not non:
        flash("Marquez au moins une image comme pertinente ou non pertinente.", "warning")
        return redirect(request.referrer or url_for(".form"))
    try:
        res = api.feedback(current_user.collection, state["sid"], method, rel, non)
    except APIError as exc:
        flash(str(exc), "danger")
        return redirect(url_for(".form"))
    state["method"] = method
    judged = state.setdefault("judged", {})           # jugements cumulés (pré-cochés ensuite)
    judged.update({str(i): 1 for i in rel})
    judged.update({str(i): 0 for i in non})
    return _render(res, state)
