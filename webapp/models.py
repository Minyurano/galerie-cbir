"""
Modèle de données de l'application web (SQLAlchemy, base SQLite).

    User ──< Category (arborescence : parent_id)   User ──< Image >── Category
                                                   Image ──> Image (parent : image d'origine d'une transformation)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

import config

db = SQLAlchemy()


def _now():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=_now)

    def set_password(self, password: str):
        # Mot de passe haché (jamais stocké en clair)
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def collection(self) -> str:
        """Nom de la collection d'index correspondante dans l'API."""
        return f"user_{self.id}"


class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    parent_id = db.Column(db.Integer, db.ForeignKey("category.id"), nullable=True, index=True)
    created_at = db.Column(db.DateTime(timezone=True), default=_now)

    parent = db.relationship("Category", remote_side=[id], backref=db.backref("children", lazy="select",
                                                                             order_by="Category.name"))
    images = db.relationship("Image", backref="category", lazy="dynamic")

    __table_args__ = (db.UniqueConstraint("user_id", "parent_id", "name", name="uq_category_sibling"),)

    def path(self) -> list["Category"]:
        """Chemin depuis la racine (fil d'Ariane)."""
        node, out = self, []
        while node is not None:
            out.append(node)
            node = node.parent
        return list(reversed(out))

    def descendants_ids(self) -> list[int]:
        ids, stack = [], [self]
        while stack:
            c = stack.pop()
            ids.append(c.id)
            stack.extend(c.children)
        return ids


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey("category.id"), nullable=True, index=True)
    original_name = db.Column(db.String(255), nullable=False)
    # Chemin du fichier : relatif à UPLOAD_DIR pour les images chargées / créées,
    # absolu pour les images importées depuis GHIM-10K (non copiées).
    path = db.Column(db.String(512), nullable=False)
    external = db.Column(db.Boolean, default=False)  # True : fichier de la base GHIM-10K
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    size_bytes = db.Column(db.Integer)
    source = db.Column(db.String(20), default="upload")  # upload | transform | import
    transform = db.Column(db.String(255))                # description de la transformation appliquée
    parent_image_id = db.Column(db.Integer, db.ForeignKey("image.id"), nullable=True)
    label = db.Column(db.String(32))                     # classe GHIM-10K (vérité terrain), si connue
    indexed = db.Column(db.Boolean, default=False)       # descripteurs calculés par l'API ?
    created_at = db.Column(db.DateTime(timezone=True), default=_now)

    def in_scope_of(self, share: "Share") -> bool:
        """L'image fait-elle partie de ce qui est partagé par ce lien ?"""
        if share.image_id is not None:
            return self.id == share.image_id
        if self.category_id is None:
            return False
        if share.include_sub:
            return self.category_id in share.category.descendants_ids()
        return self.category_id == share.category_id

    @property
    def file_path(self) -> Path:
        p = Path(self.path)
        return p if self.external or p.is_absolute() else config.UPLOAD_DIR / p

    @property
    def thumb_path(self) -> Path:
        return config.THUMB_DIR / f"{self.id}.jpg"


class Share(db.Model):
    """Lien de partage public (lecture seule) vers une catégorie ou une image.

    Le jeton est aléatoire et impossible à deviner (secrets.token_urlsafe).
    Le propriétaire peut révoquer le lien à tout moment ; une date
    d'expiration facultative le désactive automatiquement.
    """
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(64), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey("category.id"), nullable=True)
    image_id = db.Column(db.Integer, db.ForeignKey("image.id"), nullable=True)
    include_sub = db.Column(db.Boolean, default=True)      # sous-catégories comprises
    allow_download = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime(timezone=True), default=_now)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=True)
    views = db.Column(db.Integer, default=0)

    category = db.relationship("Category")
    image = db.relationship("Image")
    owner = db.relationship("User")

    @property
    def expired(self) -> bool:
        if self.expires_at is None:
            return False
        exp = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=timezone.utc)
        return exp < _now()

    @property
    def title(self) -> str:
        if self.image is not None:
            return self.image.original_name
        return " / ".join(c.name for c in self.category.path()) if self.category else "?"
