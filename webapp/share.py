"""
Partage avec des personnes extérieures (sans compte) :

    - le propriétaire crée un lien public pour une catégorie (avec ou sans ses
      sous-catégories) ou pour une seule image ;
    - la personne qui reçoit le lien voit les images en lecture seule et peut,
      si c'est autorisé, les télécharger (une par une ou en ZIP) ;
    - le propriétaire voit la liste de ses liens, leur nombre de visites, et
      peut les révoquer à tout moment.

Pour qu'une personne éloignée ouvre le lien, l'application doit être
accessible depuis Internet (voir README : redirection de port VS Code).
"""
from __future__ import annotations

import io
import secrets
import zipfile
from datetime import datetime, timedelta, timezone

from flask import Blueprint, abort, flash, redirect, render_template, request, send_file, session, url_for
from flask_login import current_user, login_required

import config
from . import imaging
from .models import Category, Image, Share, db

bp = Blueprint("share", __name__)


# --------------------------------------------------------------------------- #
# Côté propriétaire (connecté)
# --------------------------------------------------------------------------- #
@bp.route("/shares/create", methods=["POST"])
@login_required
def create():
    cat_id, img_id = request.form.get("category_id"), request.form.get("image_id")
    share = Share(token=secrets.token_urlsafe(24), user_id=current_user.id,
                  include_sub=bool(request.form.get("include_sub")),
                  allow_download=bool(request.form.get("allow_download")))
    if img_id:
        img = db.session.get(Image, int(img_id))
        if img is None or img.user_id != current_user.id:
            abort(404)
        share.image_id = img.id
    elif cat_id:
        cat = db.session.get(Category, int(cat_id))
        if cat is None or cat.user_id != current_user.id:
            abort(404)
        share.category_id = cat.id
    else:
        abort(400)
    days = request.form.get("days", type=int)
    if days and days > 0:
        share.expires_at = datetime.now(timezone.utc) + timedelta(days=min(days, 365))
    db.session.add(share)
    db.session.commit()
    flash("Lien de partage créé : copiez-le et envoyez-le à la personne de votre choix.", "success")
    return redirect(url_for(".manage", new=share.id))


def public_link(token: str) -> str:
    """Lien complet à envoyer. Si l'utilisateur a indiqué l'adresse publique de
    la galerie (ex. adresse donnée par la redirection de port de VS Code), on
    l'utilise ; sinon on reprend l'adresse utilisée dans le navigateur."""
    base = session.get("public_url") or config.PUBLIC_URL
    path = url_for("share.view", token=token)
    return base.rstrip("/") + path if base else url_for("share.view", token=token, _external=True)


@bp.route("/shares", methods=["GET", "POST"])
@login_required
def manage():
    if request.method == "POST":                       # enregistrement de l'adresse publique
        url = request.form.get("public_url", "").strip()
        if url and not url.startswith(("http://", "https://")):
            url = "https://" + url
        session["public_url"] = url.split("/s/")[0].rstrip("/") if url else ""
        flash("Adresse publique enregistrée : les liens ci-dessous l'utilisent." if url
              else "Adresse publique effacée.", "success")
        return redirect(url_for(".manage"))
    shares = Share.query.filter_by(user_id=current_user.id).order_by(Share.created_at.desc()).all()
    base = session.get("public_url") or config.PUBLIC_URL
    local = not base and request.host.split(":")[0] in ("127.0.0.1", "localhost")
    return render_template("share/manage.html", shares=shares, new_id=request.args.get("new", type=int),
                           links={s.id: public_link(s.token) for s in shares}, public_url=base, local=local)


@bp.route("/shares/<int:share_id>/revoke", methods=["POST"])
@login_required
def revoke(share_id):
    share = db.session.get(Share, share_id)
    if share is None or share.user_id != current_user.id:
        abort(404)
    db.session.delete(share)
    db.session.commit()
    flash("Lien révoqué : il ne fonctionne plus.", "info")
    return redirect(url_for(".manage"))


# --------------------------------------------------------------------------- #
# Côté visiteur (public, sans compte)
# --------------------------------------------------------------------------- #
def _active_share(token: str) -> Share:
    share = Share.query.filter_by(token=token).first()
    if share is None or share.expired:
        abort(404)
    return share


def _shared_image(share: Share, image_id: int) -> Image:
    img = db.session.get(Image, image_id)
    if img is None or img.user_id != share.user_id or not img.in_scope_of(share):
        abort(404)
    return img


def _shared_images(share: Share):
    q = Image.query.filter_by(user_id=share.user_id)
    if share.image_id is not None:
        return q.filter(Image.id == share.image_id)
    ids = share.category.descendants_ids() if share.include_sub else [share.category_id]
    return q.filter(Image.category_id.in_(ids))


@bp.route("/s/<token>")
def view(token):
    share = _active_share(token)
    share.views = (share.views or 0) + 1
    db.session.commit()
    page = max(1, request.args.get("page", 1, type=int))
    pagination = _shared_images(share).order_by(Image.category_id, Image.id).paginate(
        page=page, per_page=48, error_out=False)
    return render_template("share/public.html", share=share, pagination=pagination, hide_sidebar=True)


@bp.route("/s/<token>/thumb/<int:image_id>")
def thumb(token, image_id):
    img = _shared_image(_active_share(token), image_id)
    if not img.thumb_path.exists():
        imaging.make_thumbnail(img.file_path, img.thumb_path)
    return send_file(img.thumb_path, mimetype="image/jpeg")


@bp.route("/s/<token>/file/<int:image_id>")
def file(token, image_id):
    share = _active_share(token)
    img = _shared_image(share, image_id)
    as_attachment = request.args.get("dl") == "1"
    if as_attachment and not share.allow_download:
        abort(403)
    return send_file(img.file_path, as_attachment=as_attachment, download_name=img.original_name)


@bp.route("/s/<token>/zip")
def zip_all(token):
    share = _active_share(token)
    if not share.allow_download:
        abort(403)
    imgs = _shared_images(share).limit(1000).all()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        used = set()
        for img in imgs:
            name = img.original_name if img.original_name not in used else f"{img.id}_{img.original_name}"
            used.add(name)
            zf.write(img.file_path, arcname=name)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="partage.zip", mimetype="application/zip")
