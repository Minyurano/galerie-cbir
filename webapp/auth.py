"""Authentification : inscription, connexion, déconnexion."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from .models import User, db

bp = Blueprint("auth", __name__)

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("gallery.index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        pwd, pwd2 = request.form.get("password", ""), request.form.get("password2", "")
        errors = []
        if not USERNAME_RE.match(username):
            errors.append("Nom d'utilisateur : 3 à 32 caractères (lettres, chiffres, . _ -).")
        if not EMAIL_RE.match(email):
            errors.append("Adresse e-mail invalide.")
        if len(pwd) < 8:
            errors.append("Le mot de passe doit contenir au moins 8 caractères.")
        if pwd != pwd2:
            errors.append("Les deux mots de passe sont différents.")
        if User.query.filter((User.username == username) | (User.email == email)).first():
            errors.append("Ce nom d'utilisateur ou cet e-mail est déjà utilisé.")
        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("auth/register.html", username=username, email=email), 400
        user = User(username=username, email=email)
        user.set_password(pwd)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Compte créé. Bienvenue dans votre galerie !", "success")
        return redirect(url_for("gallery.index"))
    return render_template("auth/register.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("gallery.index"))
    if request.method == "POST":
        ident = request.form.get("username", "").strip()
        user = User.query.filter((User.username == ident) | (User.email == ident.lower())).first()
        if user and user.check_password(request.form.get("password", "")):
            login_user(user, remember=bool(request.form.get("remember")))
            nxt = request.args.get("next", "")
            # Redirection uniquement vers une page interne (évite les redirections ouvertes)
            if not nxt or urlparse(nxt).netloc or not nxt.startswith("/"):
                nxt = url_for("gallery.index")
            return redirect(nxt)
        flash("Identifiants incorrects.", "danger")
        return render_template("auth/login.html", username=ident), 401
    return render_template("auth/login.html")


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("Vous êtes déconnecté.", "info")
    return redirect(url_for("auth.login"))
