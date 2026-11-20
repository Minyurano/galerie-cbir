"""
API REST de calcul des descripteurs et de recherche d'images par le contenu.
Écrite avec Flask et son extension Flask-RESTful.

Points d'accès (tous renvoient du JSON) :

    GET    /api/health                                   état du service
    GET    /api/rf-methods                               méthodes de retour de pertinence
    POST   /api/descriptors                              tous les descripteurs d'une ou plusieurs images
    POST   /api/descriptors/color                        descripteurs de couleur seulement
    POST   /api/descriptors/texture                      descripteurs de texture seulement
    GET    /api/collections/<c>                          statistiques d'une collection
    POST   /api/collections/<c>/images                   indexer une image (fichier) ou un lot (chemins)
    GET    /api/collections/<c>/images/<id>              vecteur indexé d'une image
    DELETE /api/collections/<c>/images/<id>              retirer une image de l'index
    POST   /api/collections/<c>/search                   recherche simple (ouvre une session)
    POST   /api/collections/<c>/feedback                 itération de retour de pertinence

Lancement :  python run_api.py   (port 5001 par défaut)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource, abort

import config
from .descriptors.extractor import (FEATURE_DIM, FEATURE_GROUPS, InvalidImage, decode_image,
                                    extract, load_image)
from .retrieval.engine import SearchEngine, SessionNotFound
from .retrieval.index import IndexStore
from .retrieval.relevance_feedback import METHODS

log = logging.getLogger("cbir.api")

store: IndexStore | None = None
engine = SearchEngine()


# --------------------------------------------------------------------------- #
# Outils
# --------------------------------------------------------------------------- #
def _uploaded_images():
    """Fichiers envoyés en multipart (champ « images » ou « image »)."""
    files = request.files.getlist("images") or request.files.getlist("image")
    if not files:
        abort(400, message="Aucune image reçue (champ multipart « images »).")
    return files


def _safe_path(p: str) -> Path:
    """N'autorise la lecture que dans les dossiers déclarés dans la configuration."""
    path = Path(p).resolve()
    if not any(path.is_relative_to(root) for root in config.API_ALLOWED_ROOTS):
        abort(403, message=f"Chemin non autorisé : {p}")
    if not path.is_file():
        abort(404, message=f"Fichier introuvable : {p}")
    return path


def _collection(name: str):
    try:
        return store.get(name)
    except ValueError as exc:
        abort(400, message=str(exc))


def _scope() -> int:
    data = request.get_json(silent=True) or {}
    raw = request.form.get("scope", data.get("scope", config.SCOPE))
    try:
        return max(1, min(int(raw), 500))
    except (TypeError, ValueError):
        abort(400, message="« scope » doit être un entier.")


# --------------------------------------------------------------------------- #
# Ressources
# --------------------------------------------------------------------------- #
class Health(Resource):
    def get(self):
        return {"status": "ok", "feature_dim": FEATURE_DIM,
                "groups": [{"name": n, "dim": d} for n, d in FEATURE_GROUPS],
                "collections": store.names(), "default_method": config.DEFAULT_RF_METHOD}


class RFMethods(Resource):
    def get(self):
        return {"default": config.DEFAULT_RF_METHOD, "methods": METHODS}


class Descriptors(Resource):
    """Calcule les descripteurs d'une image ou d'un ensemble d'images.

    ?details=1 : ajoute les histogrammes R, G, B, gris, le LBP à 256 motifs et l'image LBP.
    """
    types = "all"

    def post(self):
        results = []
        for f in _uploaded_images():
            t0 = time.perf_counter()
            try:
                d = extract(decode_image(f.read()), self.types,
                            details=request.args.get("details") == "1")
            except InvalidImage as exc:
                results.append({"filename": f.filename, "error": str(exc)})
                continue
            if request.args.get("vector") != "1":      # vecteur brut seulement sur demande
                d.pop("vector", None)
            d["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            results.append({"filename": f.filename, "descriptors": d})
        return {"count": len(results), "types": self.types, "results": results}


class ColorDescriptors(Descriptors):
    types = "color"


class TextureDescriptors(Descriptors):
    types = "texture"


class CollectionInfo(Resource):
    def get(self, collection):
        col = _collection(collection)
        return {"collection": collection, "size": len(col), "feature_dim": FEATURE_DIM}


class CollectionImages(Resource):
    """Indexation : un fichier (multipart) ou un lot de chemins (JSON)."""

    def post(self, collection):
        col = _collection(collection)
        if request.files:
            image_id = request.form.get("image_id")
            if not image_id:
                abort(400, message="Champ « image_id » manquant.")
            f = _uploaded_images()[0]
            try:
                vec = extract(decode_image(f.read()), "all")["vector"]
            except InvalidImage as exc:
                abort(422, message=str(exc))
            col.add(str(image_id), np.asarray(vec), request.form.get("label"))
            return {"indexed": [image_id], "size": len(col)}, 201

        data = request.get_json(silent=True) or {}
        items = data.get("items")
        if not isinstance(items, list) or not items:
            abort(400, message="Envoyez un fichier ou un JSON {\"items\": [{\"image_id\", \"path\", \"label\"}]}.")
        ids, vecs, labels, errors = [], [], [], []
        for it in items:
            try:
                path = _safe_path(str(it["path"]))
                vecs.append(np.asarray(extract(load_image(path), "all")["vector"], dtype=np.float32))
                ids.append(str(it["image_id"]))
                labels.append(it.get("label"))
            except (InvalidImage, KeyError) as exc:
                errors.append({"item": it, "error": str(exc)})
        if ids:
            col.add_many(ids, np.stack(vecs), labels)
        return {"indexed": ids, "errors": errors, "size": len(col)}, 201


class CollectionImage(Resource):
    def get(self, collection, image_id):
        col = _collection(collection)
        k = col.position(image_id)
        if k is None:
            abort(404, message="Image absente de l'index.")
        return {"image_id": image_id, "label": col.labels[k], "vector": col.X[k].tolist()}

    def delete(self, collection, image_id):
        col = _collection(collection)
        if not col.remove(image_id):
            abort(404, message="Image absente de l'index.")
        return {"deleted": image_id, "size": len(col)}


class Search(Resource):
    """Recherche simple : image de la collection (JSON image_id) ou fichier envoyé."""

    def post(self, collection):
        col = _collection(collection)
        scope = _scope()
        try:
            if request.files:
                vec = extract(decode_image(_uploaded_images()[0].read()), "all")["vector"]
                return engine.search(col, query_vector=np.asarray(vec), scope=scope)
            data = request.get_json(silent=True) or {}
            if "image_id" not in data:
                abort(400, message="Indiquez « image_id » ou envoyez une image.")
            return engine.search(col, image_id=str(data["image_id"]), scope=scope)
        except InvalidImage as exc:
            abort(422, message=str(exc))
        except KeyError as exc:
            abort(404, message=str(exc))
        except ValueError as exc:
            abort(400, message=str(exc))


class Feedback(Resource):
    """Itération de retour de pertinence."""

    def post(self, collection):
        col = _collection(collection)
        data = request.get_json(silent=True) or {}
        method = data.get("method", config.DEFAULT_RF_METHOD)
        if method not in METHODS:
            abort(400, message=f"Méthode inconnue. Choix : {', '.join(METHODS)}")
        try:
            return engine.feedback(col, str(data.get("session_id", "")), method,
                                   [str(i) for i in data.get("relevant", [])],
                                   [str(i) for i in data.get("non_relevant", [])])
        except SessionNotFound:
            abort(404, message="Session expirée ou inconnue : relancez une recherche.")
        except ValueError as exc:
            abort(400, message=str(exc))


# --------------------------------------------------------------------------- #
# Fabrique d'application
# --------------------------------------------------------------------------- #
def create_app(index_dir: Path | None = None) -> Flask:
    global store
    config.ensure_dirs()
    store = IndexStore(index_dir or config.API_INDEX_DIR)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_AS_ASCII"] = False
    api = Api(app, prefix="/api")
    api.add_resource(Health, "/health")
    api.add_resource(RFMethods, "/rf-methods")
    api.add_resource(Descriptors, "/descriptors")
    api.add_resource(ColorDescriptors, "/descriptors/color")
    api.add_resource(TextureDescriptors, "/descriptors/texture")
    api.add_resource(CollectionInfo, "/collections/<string:collection>")
    api.add_resource(CollectionImages, "/collections/<string:collection>/images")
    api.add_resource(CollectionImage, "/collections/<string:collection>/images/<string:image_id>")
    api.add_resource(Search, "/collections/<string:collection>/search")
    api.add_resource(Feedback, "/collections/<string:collection>/feedback")
    return app
