"""
Gestion de la galerie :
    - images : chargement (une ou plusieurs), affichage, téléchargement (une ou
      plusieurs en ZIP), suppression (une ou plusieurs), déplacement ;
    - catégories hiérarchiques : création, renommage, déplacement, suppression ;
    - transformations : création de nouvelles images (recadrage, échelle...) ;
    - descripteurs : appel de l'API REST et affichage (une image ou un lot).
"""
from __future__ import annotations

import csv
import json
import io
import shutil
import tempfile
import zipfile
from pathlib import Path

from flask import (Blueprint, Response, abort, current_app, flash, redirect, render_template, request,
                   send_file, url_for)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

import config
from . import imaging
from .api_client import APIError, api
from .models import Category, Image, Share, db

bp = Blueprint("gallery", __name__)
PER_PAGE = 48


# --------------------------------------------------------------------------- #
# Outils : contrôle d'accès (un utilisateur ne voit que SES données)
# --------------------------------------------------------------------------- #
def own_image(image_id: int) -> Image:
    img = db.session.get(Image, image_id)
    if img is None or img.user_id != current_user.id:
        abort(404)
    return img


def own_category(cat_id) -> Category | None:
    if cat_id in (None, "", "0", 0):
        return None
    cat = db.session.get(Category, int(cat_id))
    if cat is None or cat.user_id != current_user.id:
        abort(404)
    return cat


def selected_ids() -> list[int]:
    ids = request.form.getlist("ids") or request.args.get("ids", "").split(",")
    out = []
    for i in ids:
        try:
            out.append(int(i))
        except (TypeError, ValueError):
            continue
    return out


def index_image(img: Image) -> bool:
    """Envoie l'image à l'API pour calculer et indexer ses descripteurs."""
    try:
        api.index_file(current_user.collection, img.id, img.file_path, img.label)
        img.indexed = True
        return True
    except APIError as exc:
        current_app.logger.warning("Indexation impossible : %s", exc)
        img.indexed = False
        return False


def delete_image(img: Image):
    """Supprime l'image : index de l'API, miniature, fichier (sauf base GHIM) et base."""
    api.unindex(current_user.collection, img.id)
    Share.query.filter_by(image_id=img.id).delete()          # les liens vers cette image disparaissent
    img.thumb_path.unlink(missing_ok=True)
    if not img.external:
        img.file_path.unlink(missing_ok=True)
    Image.query.filter_by(parent_image_id=img.id).update({"parent_image_id": None})
    db.session.delete(img)


# --------------------------------------------------------------------------- #
# Galerie
# --------------------------------------------------------------------------- #
@bp.route("/gallery")
@login_required
def index():
    cat = own_category(request.args.get("cat"))
    include_sub = request.args.get("sub") == "1"
    q = request.args.get("q", "").strip()
    page = max(1, request.args.get("page", 1, type=int))

    query = Image.query.filter_by(user_id=current_user.id)
    if cat is None and not q:
        query = query.filter(Image.category_id.is_(None)) if not include_sub else query
    elif cat is not None:
        query = query.filter(Image.category_id.in_(cat.descendants_ids()) if include_sub
                             else Image.category_id == cat.id)
    if q:
        query = query.filter(Image.original_name.ilike(f"%{q}%"))
    pagination = query.order_by(Image.created_at.desc(), Image.id.desc()).paginate(
        page=page, per_page=PER_PAGE, error_out=False)

    subcats = (Category.query.filter_by(user_id=current_user.id, parent_id=cat.id if cat else None)
               .order_by(Category.name).all())
    all_cats = Category.query.filter_by(user_id=current_user.id).all()
    cat_options = sorted(((" / ".join(x.name for x in c.path()), c.id) for c in all_cats))
    return render_template("gallery/index.html", cat=cat, subcats=subcats, pagination=pagination,
                           include_sub=include_sub, q=q, cat_options=cat_options)


# Limites de sécurité pour les archives ZIP (protection contre les « bombes ZIP »)
ZIP_MAX_FILES = 5000
ZIP_MAX_FILE_MB = 50
ZIP_MAX_TOTAL_MB = 1024


def _clean_parts(relpath: str) -> list[str]:
    """Découpe un chemin relatif (« dossier/sous-dossier/image.jpg ») en parties sûres."""
    parts = [p.strip() for p in relpath.replace("\\", "/").split("/")]
    return [p for p in parts if p and p not in (".", "..")]


def _subcategory(parent: Category | None, names: list[str], cache: dict) -> Category | None:
    """Retrouve ou crée la chaîne de sous-catégories « names » sous « parent »."""
    node = parent
    for name in names:
        name = name[:120]
        key = (node.id if node else None, name)
        if key not in cache:
            found = Category.query.filter_by(user_id=current_user.id, parent_id=key[0], name=name).first()
            if found is None:
                found = Category(user_id=current_user.id, parent_id=key[0], name=name)
                db.session.add(found)
                db.session.flush()
            cache[key] = found
        node = cache[key]
    return node


def _iter_upload(files):
    """Produit (chemin relatif, flux) pour chaque image envoyée : fichiers isolés,
    contenu d'un dossier (le navigateur transmet le chemin relatif) ou d'une archive ZIP."""
    for f in files:
        if f.filename.lower().endswith(".zip"):
            # Copie dans un vrai fichier temporaire : sous Python 3.10, le flux reçu
            # (SpooledTemporaryFile) n'a pas de méthode seekable() et zipfile échoue.
            tmp = tempfile.TemporaryFile()
            shutil.copyfileobj(f.stream, tmp)
            tmp.seek(0)
            try:
                zf = zipfile.ZipFile(tmp)
            except zipfile.BadZipFile:
                yield f.filename, None, "archive ZIP illisible"
                continue
            members = [m for m in zf.infolist() if not m.is_dir()
                       and not any(p.startswith((".", "__MACOSX")) for p in _clean_parts(m.filename))]
            total = sum(m.file_size for m in members)
            if len(members) > ZIP_MAX_FILES or total > ZIP_MAX_TOTAL_MB * 1024 * 1024:
                yield f.filename, None, f"archive trop grande (max {ZIP_MAX_FILES} fichiers, {ZIP_MAX_TOTAL_MB} Mo)"
                continue
            # Le nom de l'archive devient la catégorie de tête, sauf si l'archive
            # contient déjà un unique dossier racine (cas le plus courant).
            tops = {_clean_parts(m.filename)[0] for m in members if len(_clean_parts(m.filename)) > 1}
            single_root = len(tops) == 1 and all(len(_clean_parts(m.filename)) > 1 for m in members)
            root = "" if single_root else f.filename.rsplit("/", 1)[-1][:-4] + "/"
            for m in members:
                if m.file_size > ZIP_MAX_FILE_MB * 1024 * 1024:
                    yield m.filename, None, f"fichier de plus de {ZIP_MAX_FILE_MB} Mo"
                    continue
                yield f"{root}{m.filename}", io.BytesIO(zf.read(m)), None
        else:
            yield f.filename, f.stream, None


@bp.route("/images/upload", methods=["POST"])
@login_required
def upload():
    """Chargement d'une ou plusieurs images, d'un dossier entier ou d'une archive ZIP.

    Avec « garder l'arborescence », les dossiers deviennent des sous-catégories
    de la catégorie de destination.
    """
    cat = own_category(request.form.get("category_id"))
    keep_tree = request.form.get("keep_tree") == "on"
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        flash("Sélectionnez au moins une image, un dossier ou une archive ZIP.", "warning")
        return redirect(request.referrer or url_for(".index"))
    ok, bad, not_indexed, cache, cats_used = 0, [], 0, {}, set()
    for relpath, stream, error in _iter_upload(files):
        parts = _clean_parts(relpath) or ["image"]
        if error:
            bad.append(f"{parts[-1]} ({error})")
            continue
        name = secure_filename(parts[-1]) or "image"
        if not imaging.allowed(name):
            if len(parts) == 1:                             # fichier choisi seul : on le signale
                bad.append(f"{parts[-1]} (extension non autorisée)")
            continue                                        # dans un dossier ou un ZIP : ignoré sans bruit
        try:
            pil = imaging.open_checked(stream)
        except imaging.ImageError as exc:
            bad.append(f"{parts[-1]} ({exc})")
            continue
        target = _subcategory(cat, parts[:-1], cache) if keep_tree else cat
        cats_used.add(target.id if target else None)
        rel, size = imaging.store_new(pil, current_user.id, name.rsplit(".", 1)[1])
        img = Image(user_id=current_user.id, category_id=target.id if target else None, original_name=name,
                    path=rel, width=pil.width, height=pil.height, size_bytes=size, source="upload")
        db.session.add(img)
        db.session.flush()                         # obtient img.id avant l'indexation
        if not index_image(img):
            not_indexed += 1
        ok += 1
    db.session.commit()
    if ok:
        where = f" dans {len(cats_used)} catégorie(s)" if len(cats_used) > 1 else ""
        flash(f"{ok} image(s) chargée(s){where}.", "success")
    if not_indexed:
        flash(f"{not_indexed} image(s) non indexée(s) : API injoignable. Utilisez « Réindexer » plus tard.", "warning")
    for b in bad[:20]:
        flash(f"Refusé : {b}", "danger")
    if len(bad) > 20:
        flash(f"… et {len(bad) - 20} autre(s) fichier(s) refusé(s).", "danger")
    return redirect(url_for(".index", cat=cat.id if cat else None))


@bp.route("/images/<int:image_id>")
@login_required
def detail(image_id):
    img = own_image(image_id)
    parent = db.session.get(Image, img.parent_image_id) if img.parent_image_id else None
    derived = Image.query.filter_by(parent_image_id=img.id).all()
    all_cats = Category.query.filter_by(user_id=current_user.id).all()
    cat_options = sorted(((" / ".join(x.name for x in c.path()), c.id) for c in all_cats))
    return render_template("gallery/detail.html", img=img, parent=parent, derived=derived,
                           cat_options=cat_options)


@bp.route("/images/<int:image_id>/file")
@login_required
def file(image_id):
    img = own_image(image_id)
    if not img.file_path.exists():
        abort(404)
    return send_file(img.file_path)


@bp.route("/images/<int:image_id>/thumb")
@login_required
def thumb(image_id):
    img = own_image(image_id)
    if not img.thumb_path.exists():
        if not img.file_path.exists():
            abort(404)
        imaging.make_thumbnail(img.file_path, img.thumb_path)
    resp = send_file(img.thumb_path, mimetype="image/jpeg")
    resp.cache_control.max_age = 3600
    return resp


@bp.route("/images/<int:image_id>/download")
@login_required
def download(image_id):
    img = own_image(image_id)
    return send_file(img.file_path, as_attachment=True, download_name=img.original_name)


@bp.route("/images/<int:image_id>/delete", methods=["POST"])
@login_required
def delete(image_id):
    img = own_image(image_id)
    cat_id = img.category_id
    delete_image(img)
    db.session.commit()
    flash("Image supprimée.", "success")
    return redirect(url_for(".index", cat=cat_id))


@bp.route("/images/<int:image_id>/move", methods=["POST"])
@login_required
def move(image_id):
    img = own_image(image_id)
    cat = own_category(request.form.get("category_id"))
    img.category_id = cat.id if cat else None
    db.session.commit()
    flash("Image déplacée.", "success")
    return redirect(url_for(".detail", image_id=img.id))


@bp.route("/images/<int:image_id>/reindex", methods=["POST"])
@login_required
def reindex(image_id):
    img = own_image(image_id)
    if index_image(img):
        flash("Descripteurs recalculés et indexés.", "success")
    else:
        flash("API injoignable : indexation impossible.", "danger")
    db.session.commit()
    return redirect(url_for(".detail", image_id=img.id))


@bp.route("/images/bulk", methods=["POST"])
@login_required
def bulk():
    """Actions sur une sélection d'images : supprimer, télécharger (ZIP), déplacer, descripteurs."""
    ids = selected_ids()
    action = request.form.get("action")
    imgs = Image.query.filter(Image.user_id == current_user.id, Image.id.in_(ids)).all()
    if not imgs:
        flash("Aucune image sélectionnée.", "warning")
        return redirect(request.referrer or url_for(".index"))

    if action == "delete":
        for img in imgs:
            delete_image(img)
        db.session.commit()
        flash(f"{len(imgs)} image(s) supprimée(s).", "success")
    elif action == "download":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            used = set()
            for img in imgs:
                name = img.original_name
                if name in used:                     # évite les doublons de nom dans l'archive
                    stem, _, ext = name.rpartition(".")
                    name = f"{stem}_{img.id}.{ext}"
                used.add(name)
                zf.write(img.file_path, arcname=name)
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name="images.zip", mimetype="application/zip")
    elif action == "move":
        cat = own_category(request.form.get("category_id"))
        for img in imgs:
            img.category_id = cat.id if cat else None
        db.session.commit()
        flash(f"{len(imgs)} image(s) déplacée(s).", "success")
    elif action == "descriptors":
        return redirect(url_for(".descriptors_batch", ids=",".join(str(i.id) for i in imgs[:30])))
    elif action == "reindex":
        n = sum(index_image(img) for img in imgs)
        db.session.commit()
        flash(f"{n}/{len(imgs)} image(s) indexée(s).", "success" if n == len(imgs) else "warning")
    else:
        abort(400)
    return redirect(request.referrer or url_for(".index"))


# --------------------------------------------------------------------------- #
# Catégories hiérarchiques
# --------------------------------------------------------------------------- #
def _valid_name(name: str) -> str:
    name = (name or "").strip()
    if not 1 <= len(name) <= 120 or "/" in name:
        flash("Nom de catégorie invalide (1 à 120 caractères, sans « / »).", "danger")
        return ""
    return name


def _sibling_exists(name, parent_id, exclude_id=None):
    q = Category.query.filter_by(user_id=current_user.id, parent_id=parent_id, name=name)
    if exclude_id:
        q = q.filter(Category.id != exclude_id)
    return q.first() is not None


@bp.route("/categories/create", methods=["POST"])
@login_required
def category_create():
    parent = own_category(request.form.get("parent_id"))
    name = _valid_name(request.form.get("name"))
    if name:
        if _sibling_exists(name, parent.id if parent else None):
            flash("Une catégorie de ce nom existe déjà à cet endroit.", "danger")
        else:
            c = Category(name=name, user_id=current_user.id, parent_id=parent.id if parent else None)
            db.session.add(c)
            db.session.commit()
            flash(f"Catégorie « {name} » créée.", "success")
            return redirect(url_for(".index", cat=c.id))
    return redirect(url_for(".index", cat=parent.id if parent else None))


@bp.route("/categories/<int:cat_id>/rename", methods=["POST"])
@login_required
def category_rename(cat_id):
    cat = own_category(cat_id)
    name = _valid_name(request.form.get("name"))
    if name and not _sibling_exists(name, cat.parent_id, cat.id):
        cat.name = name
        db.session.commit()
        flash("Catégorie renommée.", "success")
    elif name:
        flash("Une catégorie de ce nom existe déjà à cet endroit.", "danger")
    return redirect(url_for(".index", cat=cat.id))


@bp.route("/categories/<int:cat_id>/move", methods=["POST"])
@login_required
def category_move(cat_id):
    cat = own_category(cat_id)
    new_parent = own_category(request.form.get("parent_id"))
    if new_parent is not None and new_parent.id in cat.descendants_ids():
        flash("Impossible : une catégorie ne peut pas être placée dans elle-même ou ses sous-catégories.", "danger")
    elif _sibling_exists(cat.name, new_parent.id if new_parent else None, cat.id):
        flash("Une catégorie de ce nom existe déjà à la destination.", "danger")
    else:
        cat.parent_id = new_parent.id if new_parent else None
        db.session.commit()
        flash("Catégorie déplacée.", "success")
    return redirect(url_for(".index", cat=cat.id))


@bp.route("/categories/<int:cat_id>/delete", methods=["POST"])
@login_required
def category_delete(cat_id):
    """Supprime une catégorie. Mode « keep » : son contenu remonte au parent ;
    mode « all » : sous-catégories et images sont supprimées."""
    cat = own_category(cat_id)
    parent_id = cat.parent_id
    Share.query.filter(Share.category_id.in_(cat.descendants_ids() if request.form.get("mode") == "all"
                                             else [cat.id])).delete(synchronize_session=False)
    if request.form.get("mode") == "all":
        ids = cat.descendants_ids()
        for img in Image.query.filter(Image.user_id == current_user.id, Image.category_id.in_(ids)).all():
            delete_image(img)
        for cid in reversed(ids):                       # enfants d'abord
            db.session.delete(db.session.get(Category, cid))
    else:
        Image.query.filter_by(user_id=current_user.id, category_id=cat.id).update({"category_id": parent_id})
        for child in list(cat.children):
            child.parent_id = parent_id
            if _sibling_exists(child.name, parent_id, child.id):
                child.name = f"{child.name} ({child.id})"
        db.session.delete(cat)
    db.session.commit()
    flash("Catégorie supprimée.", "success")
    return redirect(url_for(".index", cat=parent_id))


# --------------------------------------------------------------------------- #
# Transformations : création de nouvelles images
# --------------------------------------------------------------------------- #
@bp.route("/images/<int:image_id>/transform", methods=["GET", "POST"])
@login_required
def transform(image_id):
    img = own_image(image_id)
    if request.method == "POST":
        try:
            pil = imaging.open_checked(img.file_path)
            out, done = imaging.apply_transformations(pil, request.form)
        except (imaging.ImageError, ValueError) as exc:
            flash(f"Transformation impossible : {exc}", "danger")
            return redirect(url_for(".transform", image_id=img.id))
        stem, _, ext = img.original_name.rpartition(".")
        ext = ext.lower() if ext.lower() in config.ALLOWED_EXTENSIONS else "png"
        rel, size = imaging.store_new(out, current_user.id, ext)
        new = Image(user_id=current_user.id, category_id=img.category_id,
                    original_name=f"{stem or 'image'}_transforme.{ 'jpg' if ext in ('jpg', 'jpeg') else ext}",
                    path=rel, width=out.width, height=out.height, size_bytes=size, source="transform",
                    transform=", ".join(done)[:255], parent_image_id=img.id)
        db.session.add(new)
        db.session.flush()
        new.original_name = f"{stem or 'image'}_t{new.id}.{'jpg' if ext in ('jpg', 'jpeg') else ext}"
        index_image(new)
        db.session.commit()
        flash(f"Nouvelle image créée : {', '.join(done)}.", "success")
        return redirect(url_for(".detail", image_id=new.id))
    return render_template("gallery/transform.html", img=img)


# --------------------------------------------------------------------------- #
# Descripteurs (consommation de l'API REST)
# --------------------------------------------------------------------------- #
@bp.route("/images/<int:image_id>/descriptors")
@login_required
def descriptors(image_id):
    img = own_image(image_id)
    types = request.args.get("types", "all")
    if types not in ("all", "color", "texture"):
        types = "all"
    try:
        res = api.descriptors([img.file_path], types, details=True)["results"][0]
    except APIError as exc:
        flash(str(exc), "danger")
        return redirect(url_for(".detail", image_id=img.id))
    if "error" in res:
        flash(res["error"], "danger")
        return redirect(url_for(".detail", image_id=img.id))
    d = res["descriptors"]
    # JSON affiché : l'image LBP (base64, plusieurs Ko) est abrégée pour rester lisible
    raw = json.loads(json.dumps(d))
    if raw.get("texture", {}).get("lbp_full", {}).get("image"):
        raw["texture"]["lbp_full"]["image"] = raw["texture"]["lbp_full"]["image"][:60] + "…"
    return render_template("gallery/descriptors.html", img=img, d=d, types=types,
                           d_json=json.dumps(raw, indent=2, ensure_ascii=False))


def _batch_descriptors(ids):
    imgs = Image.query.filter(Image.user_id == current_user.id, Image.id.in_(ids)).all()
    if not imgs:
        abort(404)
    res = api.descriptors([i.file_path for i in imgs], "all")["results"]
    return list(zip(imgs, res))


@bp.route("/descriptors/batch")
@login_required
def descriptors_batch():
    ids = selected_ids()[:30]
    try:
        rows = _batch_descriptors(ids)
    except APIError as exc:
        flash(str(exc), "danger")
        return redirect(url_for(".index"))
    return render_template("gallery/descriptors_batch.html", rows=rows, ids=",".join(map(str, ids)))


@bp.route("/descriptors/batch.csv")
@login_required
def descriptors_batch_csv():
    """Export CSV (une ligne par image) des descripteurs d'un ensemble d'images."""
    ids = selected_ids()[:30]
    try:
        rows = _batch_descriptors(ids)
    except APIError as exc:
        flash(str(exc), "danger")
        return redirect(url_for(".index"))
    out = io.StringIO()
    w = csv.writer(out, delimiter=";")
    header = ["image_id", "nom"]
    header += [f"moment_{c}_{k}" for c in "HSV" for k in ("mean", "std", "skewness")]
    header += [f"dominante_{i + 1}_{k}" for i in range(5) for k in ("hex", "proportion")]
    header += [f"glcm_{p}" for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM", "entropy")]
    header += [f"lbp_{i}" for i in range(10)] + [f"hsv_hist_{i}" for i in range(128)]
    w.writerow(header)
    for img, r in rows:
        if "error" in r:
            continue
        d = r["descriptors"]
        row = [img.id, img.original_name]
        row += [round(d["color"]["moments"][c][k], 6) for c in "HSV" for k in ("mean", "std", "skewness")]
        for c in d["color"]["dominant_colors"]:
            row += [c["hex"], round(c["proportion"], 4)]
        row += [round(d["texture"]["glcm"]["features"][p]["mean"], 6)
                for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM", "entropy")]
        row += d["texture"]["lbp"]["histogram"] + d["color"]["histogram"]["values"]
        w.writerow(row)
    return Response(out.getvalue().encode("utf-8-sig"), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=descripteurs.csv"})
