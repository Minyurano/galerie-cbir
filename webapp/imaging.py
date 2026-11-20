"""
Traitement des fichiers images côté application web (Pillow) :
validation, enregistrement, miniatures et TRANSFORMATIONS (création de
nouvelles images à partir d'images existantes).
"""
from __future__ import annotations

import uuid
from pathlib import Path

from PIL import Image as PILImage, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

import config

THUMB_SIZE = (320, 320)


class ImageError(ValueError):
    pass


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def open_checked(path_or_file) -> PILImage.Image:
    """Ouvre et VÉRIFIE qu'il s'agit bien d'une image (sécurité)."""
    try:
        img = PILImage.open(path_or_file)
        img.verify()                      # détecte les fichiers corrompus / falsifiés
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageError("Fichier image invalide ou corrompu.") from exc
    if hasattr(path_or_file, "seek"):
        path_or_file.seek(0)
    img = PILImage.open(path_or_file)
    img.load()
    return ImageOps.exif_transpose(img)   # respecte l'orientation des photos de téléphone


def store_new(img: PILImage.Image, user_id: int, ext: str) -> tuple[str, int]:
    """Enregistre une image dans data/uploads/<user>/ sous un nom unique.

    Retourne (chemin relatif à UPLOAD_DIR, taille en octets).
    """
    ext = ext.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    fmt = {"jpeg": "JPEG", "png": "PNG", "bmp": "BMP", "webp": "WEBP", "tif": "TIFF", "tiff": "TIFF"}.get(ext, "PNG")
    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    rel = Path(str(user_id)) / f"{uuid.uuid4().hex}.{'jpg' if fmt == 'JPEG' else ext}"
    dest = config.UPLOAD_DIR / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return rel.as_posix(), dest.stat().st_size


def make_thumbnail(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with PILImage.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail(THUMB_SIZE)
        im.save(dest, "JPEG", quality=85)
    return dest


# --------------------------------------------------------------------------- #
# Transformations
# --------------------------------------------------------------------------- #
def apply_transformations(img: PILImage.Image, ops: dict) -> tuple[PILImage.Image, list[str]]:
    """Applique les opérations demandées, dans cet ordre :
    recadrage -> changement d'échelle -> rotation -> miroir -> couleurs -> filtres.

    ops : dictionnaire issu du formulaire (valeurs vides = opération ignorée).
    Retourne l'image transformée et la liste lisible des opérations appliquées.
    """
    done = []
    w, h = img.size

    # 1) Recadrage (coordonnées en pixels de l'image d'origine)
    if ops.get("crop_w") and ops.get("crop_h"):
        x, y = max(0, int(float(ops.get("crop_x") or 0))), max(0, int(float(ops.get("crop_y") or 0)))
        cw, ch = int(float(ops["crop_w"])), int(float(ops["crop_h"]))
        x2, y2 = min(w, x + cw), min(h, y + ch)
        if x2 - x < 2 or y2 - y < 2:
            raise ImageError("Zone de recadrage trop petite ou hors de l'image.")
        if (x, y, x2, y2) != (0, 0, w, h):
            img = img.crop((x, y, x2, y2))
            done.append(f"recadrage {x2 - x}x{y2 - y} @({x},{y})")

    # 2) Changement d'échelle : pourcentage OU largeur / hauteur
    if ops.get("scale"):
        s = float(ops["scale"]) / 100.0
        if not 0.05 <= s <= 5:
            raise ImageError("L'échelle doit être comprise entre 5 % et 500 %.")
        if s != 1:
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), PILImage.LANCZOS)
            done.append(f"échelle {float(ops['scale']):g} %")
    elif ops.get("resize_w") or ops.get("resize_h"):
        nw = int(ops["resize_w"]) if ops.get("resize_w") else None
        nh = int(ops["resize_h"]) if ops.get("resize_h") else None
        ratio = img.width / img.height
        if ops.get("keep_ratio"):          # proportions conservées : la largeur prime
            if nw:
                nh = round(nw / ratio)
            else:
                nw = round(nh * ratio)
        else:                              # dimension manquante déduite des proportions
            nw = nw or round(nh * ratio)
            nh = nh or round(nw / ratio)
        if not (1 <= nw <= 10000 and 1 <= nh <= 10000):
            raise ImageError("Dimensions demandées invalides.")
        img = img.resize((nw, nh), PILImage.LANCZOS)
        done.append(f"redimensionnement {nw}x{nh}")

    # 3) Rotation (degrés, sens anti-horaire, l'image est agrandie pour ne rien couper)
    if ops.get("rotate") and float(ops["rotate"]) % 360 != 0:
        angle = float(ops["rotate"])
        img = img.rotate(angle, expand=True, resample=PILImage.BICUBIC,
                         fillcolor=(255, 255, 255) if img.mode == "RGB" else None)
        done.append(f"rotation {angle:g}°")

    # 4) Miroirs
    if ops.get("flip_h"):
        img = ImageOps.mirror(img)
        done.append("miroir horizontal")
    if ops.get("flip_v"):
        img = ImageOps.flip(img)
        done.append("miroir vertical")

    # 5) Couleurs
    if ops.get("grayscale"):
        img = ImageOps.grayscale(img).convert("RGB")
        done.append("niveaux de gris")
    for key, enhancer, label in (("brightness", ImageEnhance.Brightness, "luminosité"),
                                 ("contrast", ImageEnhance.Contrast, "contraste"),
                                 ("saturation", ImageEnhance.Color, "saturation")):
        if ops.get(key) and float(ops[key]) != 100:
            img = enhancer(img.convert("RGB")).enhance(float(ops[key]) / 100.0)
            done.append(f"{label} {float(ops[key]):g} %")

    # 6) Filtres
    if ops.get("blur") and float(ops["blur"]) > 0:
        img = img.filter(ImageFilter.GaussianBlur(float(ops["blur"])))
        done.append(f"flou gaussien r={float(ops['blur']):g}")
    if ops.get("sharpen"):
        img = img.filter(ImageFilter.SHARPEN)
        done.append("netteté")
    if ops.get("equalize"):
        img = ImageOps.equalize(img.convert("RGB"))
        done.append("égalisation d'histogramme")

    if not done:
        raise ImageError("Aucune transformation sélectionnée.")
    return img, done
