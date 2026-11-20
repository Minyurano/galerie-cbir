"""
Fabrique de l'application web « Galerie CBIR » (Flask).

Composants :
    - Flask-Login     : authentification et sessions utilisateur
    - Flask-SQLAlchemy: base SQLite (utilisateurs, catégories, images)
    - Flask-WTF       : protection CSRF de tous les formulaires
    - api_client      : consommation de l'API REST de descripteurs (Flask-RESTful)

Lancement :  python run_web.py   (port 5000 par défaut)
"""
from __future__ import annotations

from flask import Flask, redirect, render_template, url_for
from flask_login import LoginManager, current_user
from flask_wtf.csrf import CSRFError, CSRFProtect

import config
from .models import Category, User, db

login_manager = LoginManager()
csrf = CSRFProtect()


@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))


def _category_tree(user_id: int):
    """Arborescence des catégories de l'utilisateur, avec le nombre d'images."""
    from sqlalchemy import func
    from .models import Image
    cats = Category.query.filter_by(user_id=user_id).order_by(Category.name).all()
    counts = dict(db.session.query(Image.category_id, func.count(Image.id))
                  .filter(Image.user_id == user_id).group_by(Image.category_id).all())
    nodes = {c.id: {"cat": c, "children": [], "count": counts.get(c.id, 0)} for c in cats}
    roots = []
    for c in cats:
        (nodes[c.parent_id]["children"] if c.parent_id in nodes else roots).append(nodes[c.id])

    def total(n):
        n["total"] = n["count"] + sum(total(ch) for ch in n["children"])
        return n["total"]
    for r in roots:
        total(r)
    return roots, counts.get(None, 0)


def create_app(test_config: dict | None = None) -> Flask:
    config.ensure_dirs()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=config.secret_key(),
        SQLALCHEMY_DATABASE_URI=config.DATABASE_URI,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        WTF_CSRF_TIME_LIMIT=None,
    )
    if test_config:
        app.config.update(test_config)

    # Derrière un tunnel ou un proxy (redirection de port VS Code, ngrok...),
    # on reprend l'adresse publique transmise dans les en-têtes X-Forwarded-*.
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    db.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Connectez-vous pour accéder à votre galerie."
    login_manager.login_message_category = "warning"

    from .auth import bp as auth_bp
    from .gallery import bp as gallery_bp
    from .search import bp as search_bp
    from .share import bp as share_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(gallery_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(share_bp)

    @app.context_processor
    def inject_tree():
        if current_user.is_authenticated:
            tree, uncategorized = _category_tree(current_user.id)
            return {"category_tree": tree, "uncategorized_count": uncategorized}
        return {}

    @app.route("/")
    def home():
        return redirect(url_for("gallery.index") if current_user.is_authenticated else url_for("auth.login"))

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, message="Page ou image introuvable."), 404

    @app.errorhandler(413)
    def too_large(_):
        return render_template("error.html", code=413,
                               message=f"Fichier trop volumineux (maximum {config.MAX_UPLOAD_MB} Mo)."), 413

    @app.errorhandler(CSRFError)
    def csrf_error(e):
        return render_template("error.html", code=400, message="Formulaire expiré, rechargez la page."), 400

    with app.app_context():
        db.create_all()
    return app
