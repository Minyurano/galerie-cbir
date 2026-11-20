"""
Importe la base GHIM-10K dans la galerie d'un utilisateur :
    1. crée l'utilisateur s'il n'existe pas ;
    2. crée une hiérarchie de catégories (GHIM-10K > thème > classe) ;
    3. enregistre les 10 000 images (sans les copier : chemin vers la base) ;
    4. demande à l'API de calculer et d'indexer leurs descripteurs (par lots).

L'API doit être lancée :  python run_api.py

Usage :
    python scripts/import_ghim.py --user demo --password demo1234 [--ghim CHEMIN] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from webapp.api_client import APIError, api  # noqa: E402
from webapp.app import create_app  # noqa: E402
from webapp.models import Category, Image, User, db  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ghim_utils import GHIM_CLASSES, list_ghim  # noqa: E402

BATCH = 200


def get_or_create_category(user, name, parent):
    c = Category.query.filter_by(user_id=user.id, name=name, parent_id=parent.id if parent else None).first()
    if c is None:
        c = Category(user_id=user.id, name=name, parent_id=parent.id if parent else None)
        db.session.add(c)
        db.session.flush()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="demo")
    ap.add_argument("--email", default=None)
    ap.add_argument("--password", default="demo1234")
    ap.add_argument("--ghim", default=str(config.GHIM_DIR))
    ap.add_argument("--limit", type=int, default=0, help="nombre maximal d'images par classe (0 = toutes)")
    args = ap.parse_args()

    ghim = Path(args.ghim).resolve()
    files = list_ghim(ghim)                  # [(chemin, classe), ...] : version plate ou Kaggle
    if not files:
        sys.exit(f"Aucune image trouvée dans {ghim}. Téléchargez GHIM-10K et indiquez --ghim.")
    try:
        api.health()
    except APIError as exc:
        sys.exit(str(exc))

    app = create_app()
    with app.app_context():
        user = User.query.filter_by(username=args.user).first()
        if user is None:
            user = User(username=args.user, email=args.email or f"{args.user}@example.org")
            user.set_password(args.password)
            db.session.add(user)
            db.session.commit()
            print(f"Utilisateur créé : {args.user}")

        root = get_or_create_category(user, "GHIM-10K", None)
        cats = {}
        for cls in sorted({c for _, c in files}):
            theme, name = GHIM_CLASSES[cls]
            t = get_or_create_category(user, theme, root)
            cats[cls] = get_or_create_category(user, f"{name} ({cls:02d})", t)
        db.session.commit()

        existing = {p for (p,) in db.session.query(Image.path).filter_by(user_id=user.id, external=True)}
        per_class, todo = {}, []
        for f, cls in files:
            per_class[cls] = per_class.get(cls, 0) + 1
            if args.limit and per_class[cls] > args.limit:
                continue
            if str(f) in existing:
                continue
            with PILImage.open(f) as im:
                w, h = im.size
            img = Image(user_id=user.id, category_id=cats[cls].id, original_name=f.name, path=str(f),
                        external=True, width=w, height=h, size_bytes=f.stat().st_size, source="import",
                        label=str(cls))
            db.session.add(img)
            todo.append(img)
        db.session.commit()
        print(f"{len(todo)} image(s) enregistrée(s) ; indexation par l'API...")

        t0 = time.time()
        for k in range(0, len(todo), BATCH):
            chunk = todo[k:k + BATCH]
            res = api.index_paths(user.collection, [{"image_id": str(i.id), "path": i.path, "label": i.label}
                                                    for i in chunk])
            ok = set(res["indexed"])
            for i in chunk:
                i.indexed = str(i.id) in ok
            db.session.commit()
            print(f"  {min(k + BATCH, len(todo))}/{len(todo)} indexées ({time.time() - t0:.0f} s)", flush=True)
        print(f"Terminé. Connectez-vous avec « {args.user} » sur http://{config.WEB_HOST}:{config.WEB_PORT}")


if __name__ == "__main__":
    main()
