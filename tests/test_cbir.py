"""
Tests automatiques (pytest) : descripteurs, retour de pertinence, API REST et
application web (avec l'API simulée par le client de test Flask).

    python -m pytest -q
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image as PILImage


def _png(color=(200, 30, 30), size=(64, 48), noise=False) -> bytes:
    arr = np.zeros((size[1], size[0], 3), np.uint8) + np.array(color, np.uint8)
    if noise:
        arr = (arr + np.random.default_rng(0).integers(0, 60, arr.shape)).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "THUMB_DIR", tmp_path / "thumbs")
    monkeypatch.setattr(config, "API_INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "API_ALLOWED_ROOTS", [(tmp_path / "uploads").resolve()])
    return tmp_path


# ------------------------------------------------------------------ descripteurs
def test_descriptor_vector_shape_and_ranges():
    from api.descriptors.extractor import FEATURE_DIM, decode_image, extract
    d = extract(decode_image(_png(noise=True)))
    assert len(d["vector"]) == FEATURE_DIM == 181
    assert abs(sum(d["color"]["histogram"]["values"]) - 1) < 1e-4
    assert abs(sum(d["texture"]["lbp"]["histogram"]) - 1) < 1e-4
    assert abs(sum(c["proportion"] for c in d["color"]["dominant_colors"]) - 1) < 1e-4
    assert set(d["texture"]["glcm"]["features"]) == {"contrast", "dissimilarity", "homogeneity", "energy",
                                                     "correlation", "ASM", "entropy"}


def test_dominant_color_of_uniform_image():
    from api.descriptors.extractor import decode_image, extract
    d = extract(decode_image(_png(color=(0, 0, 255))), "color")
    top = d["color"]["dominant_colors"][0]
    assert top["proportion"] > 0.99 and top["rgb"][2] > 240


def test_invalid_image_rejected():
    from api.descriptors.extractor import InvalidImage, decode_image
    with pytest.raises(InvalidImage):
        decode_image(b"not an image")


# ------------------------------------------------------- retour de pertinence
@pytest.mark.parametrize("method", ["qvm", "fre", "type1"])
def test_feedback_improves_precision_on_synthetic_data(method):
    """Deux classes séparées sur quelques composantes seulement : le retour de
    pertinence doit améliorer (ou conserver) la précision."""
    from api.descriptors.extractor import FEATURE_DIM
    from api.retrieval.relevance_feedback import apply_feedback, new_state, rank
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, FEATURE_DIM)).astype(np.float32)
    y = np.repeat([0, 1], 200)
    X[y == 1, 140:150] += 1.5          # seule une partie des composantes discrimine
    q = 0
    st = new_state(X[q], 25, q)
    st.relevant.add(q)
    order, _ = rank(X, st)
    p0 = (y[order[:25]] == y[q]).mean()
    for _ in range(3):
        st.last_retrieved = order[:25].tolist()
        shown = order[:25]
        apply_feedback(X, st, method, [int(i) for i in shown if y[i] == y[q] and i != q],
                       [int(i) for i in shown if y[i] != y[q]])
        order, _ = rank(X, st)
    p3 = (y[order[:25]] == y[q]).mean()
    assert p3 >= p0
    assert np.isfinite(st.w).all() and abs(st.e.sum() - 1) < 1e-5


# ----------------------------------------------------------------------- API
def test_api_index_search_feedback(env):
    from api.app import create_app
    c = create_app(env / "index").test_client()
    for i in range(12):
        col = (255, 20 * i, 0) if i < 6 else (0, 20 * i, 255)
        r = c.post("/api/collections/u1/images", data={"image_id": str(i), "label": "r" if i < 6 else "b",
                                                        "image": (io.BytesIO(_png(col, noise=True)), "x.png")},
                   content_type="multipart/form-data")
        assert r.status_code == 201
    r = c.post("/api/collections/u1/search", json={"image_id": "0", "scope": 5})
    assert r.status_code == 200 and len(r.json["results"]) == 5
    sid = r.json["session_id"]
    for m in ("qvm", "fre", "type1"):
        r = c.post("/api/collections/u1/feedback", json={"session_id": sid, "method": m,
                                                         "relevant": ["1", "2"], "non_relevant": ["7"]})
        assert r.status_code == 200, r.json
    assert c.post("/api/collections/u1/feedback", json={"session_id": "x", "relevant": ["1"]}).status_code == 404
    assert c.post("/api/collections/u1/images", json={"items": [{"image_id": "z", "path": "/etc/passwd"}]}).status_code == 403
    assert c.delete("/api/collections/u1/images/3").status_code == 200


# ------------------------------------------------------------- application web
@pytest.fixture()
def web(env, monkeypatch):
    """Application web dont le client HTTP de l'API est redirigé vers l'API de test."""
    import requests
    from api.app import create_app as create_api
    from webapp import api_client
    from webapp.app import create_app

    api_test = create_api(env / "index").test_client()

    class FakeResp:
        def __init__(self, r):
            self.status_code, self._r, self.text = r.status_code, r, r.get_data(as_text=True)

        def json(self):
            return self._r.json

    def fake_request(self, method, url, timeout=None, json=None, data=None, files=None):
        path = url.split("://", 1)[1].split("/", 1)[1]
        kw = {}
        if json is not None:
            kw["json"] = json
        if files is not None:
            form = dict(data or {})
            for field, (name, content, _) in files:
                form.setdefault(field, []).append((io.BytesIO(content), name))
            kw.update(data=form, content_type="multipart/form-data")
        elif data is not None:
            kw["data"] = data
        return FakeResp(api_test.open("/" + path, method=method, **kw))

    monkeypatch.setattr(requests.Session, "request", fake_request)
    app = create_app({"TESTING": True, "WTF_CSRF_ENABLED": False,
                      "SQLALCHEMY_DATABASE_URI": f"sqlite:///{env / 'test.db'}"})
    return app.test_client()


def test_web_full_workflow(web):
    r = web.post("/register", data={"username": "alice", "email": "a@b.cd", "password": "password1",
                                    "password2": "password1"})
    assert r.status_code == 302
    # Catégories hiérarchiques
    web.post("/categories/create", data={"name": "Nature"})
    web.post("/categories/create", data={"name": "Fleurs", "parent_id": "1"})
    # Chargement multiple
    files = [(io.BytesIO(_png((255, 10 * i, 0), noise=True)), f"img{i}.png") for i in range(4)]
    r = web.post("/images/upload", data={"category_id": "2", "files": files}, content_type="multipart/form-data")
    assert r.status_code == 302
    r = web.get("/gallery?cat=2")
    assert r.status_code == 200 and r.data.count(b"thumb-card") == 4
    # Descripteurs (une image + lot + CSV)
    assert b"Couleurs dominantes" in web.get("/images/1/descriptors").data
    assert web.get("/descriptors/batch?ids=1,2").status_code == 200
    assert web.get("/descriptors/batch.csv?ids=1,2").data.decode("utf-8-sig").count("\n") == 3
    # Transformation -> nouvelle image indexée
    r = web.post("/images/1/transform", data={"crop_x": "0", "crop_y": "0", "crop_w": "32", "crop_h": "24",
                                              "scale": "200", "flip_h": "on"})
    assert r.status_code == 302
    assert web.get("/images/5").status_code == 200
    # Recherche simple puis retour de pertinence
    r = web.get("/search/image/1?scope=5")
    assert r.status_code == 200 and b"Recherche simple" in r.data
    with web.session_transaction() as s:
        sid = s["search"]["sid"]
    r = web.post("/search/feedback", data={"session_id": sid, "method": "type1", "relevant": ["2"],
                                           "non_relevant": ["3"]})
    assert r.status_code == 200 and "itération 1".encode() in r.data
    # Téléchargement ZIP puis suppression multiple
    r = web.post("/images/bulk", data={"action": "download", "ids": ["1", "2"]})
    assert r.mimetype == "application/zip"
    web.post("/images/bulk", data={"action": "delete", "ids": ["3", "4"]})
    assert web.get("/images/3").status_code == 404


def test_web_isolation_between_users(web):
    web.post("/register", data={"username": "bob", "email": "b@b.cd", "password": "password1", "password2": "password1"})
    web.post("/images/upload", data={"files": [(io.BytesIO(_png()), "a.png")]}, content_type="multipart/form-data")
    web.post("/logout")
    web.post("/register", data={"username": "eve", "email": "e@b.cd", "password": "password1", "password2": "password1"})
    assert web.get("/images/1").status_code == 404          # image de bob invisible pour eve
    assert web.get("/images/1/file").status_code == 404


def test_public_share_link(web):
    """Lien public : visible sans compte, limité au contenu partagé, révocable."""
    web.post("/register", data={"username": "zoe", "email": "z@b.cd", "password": "password1", "password2": "password1"})
    web.post("/categories/create", data={"name": "Album"})
    web.post("/categories/create", data={"name": "Sous", "parent_id": "1"})
    web.post("/images/upload", data={"category_id": "1", "files": [(io.BytesIO(_png()), "a.png")]},
             content_type="multipart/form-data")
    web.post("/images/upload", data={"category_id": "2", "files": [(io.BytesIO(_png((0, 0, 255))), "b.png")]},
             content_type="multipart/form-data")
    web.post("/images/upload", data={"files": [(io.BytesIO(_png((0, 255, 0))), "privee.png")]},
             content_type="multipart/form-data")
    # Partage de la catégorie Album SANS sous-catégories
    web.post("/shares/create", data={"category_id": "1", "allow_download": "on", "days": "7"})
    web.post("/shares", data={"public_url": "https://exemple-5000.devtunnels.ms"})
    page = web.get("/shares").data.decode()
    assert "https://exemple-5000.devtunnels.ms/s/" in page
    from webapp.models import Share
    with web.application.app_context():
        token = Share.query.first().token
    web.post("/logout")
    # Visiteur sans compte
    r = web.get(f"/s/{token}")
    assert r.status_code == 200 and b"a.png" in r.data and b"b.png" not in r.data
    assert web.get(f"/s/{token}/thumb/1").status_code == 200
    assert web.get(f"/s/{token}/thumb/2").status_code == 404      # sous-catégorie non partagée
    assert web.get(f"/s/{token}/file/3").status_code == 404       # image privée
    assert web.get(f"/s/{token}/zip").mimetype == "application/zip"
    assert web.get("/s/jeton-invente").status_code == 404
    # Révocation par le propriétaire
    web.post("/login", data={"username": "zoe", "password": "password1"})
    web.post("/shares/1/revoke")
    web.post("/logout")
    assert web.get(f"/s/{token}").status_code == 404


def test_secret_key_aleatoire_et_persistante(tmp_path, monkeypatch):
    """Sans CBIR_SECRET_KEY, une clé aléatoire est créée une fois puis réutilisée."""
    import config
    monkeypatch.setattr(config, "SECRET_KEY", "")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    k1 = config.secret_key()
    assert len(k1) == 64 and k1 != "change-me-in-production"
    assert config.secret_key() == k1
    monkeypatch.setattr(config, "SECRET_KEY", "fixee")
    assert config.secret_key() == "fixee"


def test_methode_par_defaut_du_groupe():
    import config
    from api.retrieval.relevance_feedback import METHODS
    assert config.DEFAULT_RF_METHOD in METHODS


def test_descripteurs_detailles_rgb_et_lbp_256():
    """?details=1 : histogrammes R, G, B, gris, LBP à 256 motifs et image LBP."""
    from api.descriptors.extractor import decode_image, extract
    d = extract(decode_image(_png(noise=True)), "all", details=True)
    rgb = d["color"]["rgb_histograms"]
    for ch in ("R", "G", "B", "gray"):
        assert len(rgb[ch]) == rgb["bins"] and abs(sum(rgb[ch]) - 1) < 1e-4
    full = d["texture"]["lbp_full"]
    assert len(full["histogram"]) == 256 and abs(sum(full["histogram"]) - 1) < 1e-4
    assert full["image"].startswith("data:image/png;base64,")
    # sans details, la réponse reste légère et le vecteur de recherche ne change pas
    d0 = extract(decode_image(_png(noise=True)), "all")
    assert "rgb_histograms" not in d0["color"] and "lbp_full" not in d0["texture"]
    assert len(d0["vector"]) == len(d["vector"])


def test_chargement_zip_et_dossier_avec_sous_categories(web):
    import zipfile
    web.post("/register", data={"username": "bob", "email": "b@b.cd", "password": "password1",
                                "password2": "password1"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("chevaux/a.png", _png((120, 80, 40), noise=True))
        zf.writestr("chevaux/b.png", _png((110, 90, 30), noise=True))
        zf.writestr("mer/c.png", _png((20, 90, 200), noise=True))
        zf.writestr("lisez-moi.txt", "ignoré")
        zf.writestr("__MACOSX/chevaux/._a.png", "ignoré")
    buf.seek(0)
    r = web.post("/images/upload", data={"keep_tree": "on", "files": [(buf, "animaux.zip")]},
                 content_type="multipart/form-data")
    assert r.status_code == 302
    # Dossier envoyé par le navigateur : le nom de fichier contient le chemin relatif
    r = web.post("/images/upload", data={"keep_tree": "on", "files": [
        (io.BytesIO(_png((10, 200, 10), noise=True)), "vacances/plage/d.png")]}, content_type="multipart/form-data")
    from webapp.models import Category, Image
    with web.application.app_context():
        names = {c.name: c for c in Category.query.all()}
        assert {"animaux", "chevaux", "mer", "vacances", "plage"} <= set(names)
        assert names["chevaux"].parent.name == "animaux" and names["plage"].parent.name == "vacances"
        assert Image.query.filter_by(category_id=names["chevaux"].id).count() == 2
        assert Image.query.count() == 4


def test_zip_avec_dossier_racine_unique_pas_de_doublon(web):
    """« Animaux.zip » contenant déjà « Animaux/… » ne crée pas Animaux/Animaux."""
    import zipfile
    web.post("/register", data={"username": "dan", "email": "d@b.cd", "password": "password1",
                                "password2": "password1"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Animaux/Chevaux/a.png", _png((120, 80, 40), noise=True))
    buf.seek(0)
    web.post("/images/upload", data={"keep_tree": "on", "files": [(buf, "Animaux.zip")]},
             content_type="multipart/form-data")
    from webapp.models import Category
    with web.application.app_context():
        assert sorted(c.name for c in Category.query.all()) == ["Animaux", "Chevaux"]


def test_page_descripteurs_affiche_rgb_et_lbp(web):
    web.post("/register", data={"username": "carl", "email": "c@b.cd", "password": "password1",
                                "password2": "password1"})
    web.post("/images/upload", data={"files": [(io.BytesIO(_png(noise=True)), "x.png")]},
             content_type="multipart/form-data")
    html = web.get("/images/1/descriptors").data.decode()
    assert "rgbChart" in html and "lbp256Chart" in html and "data:image/png;base64," in html
