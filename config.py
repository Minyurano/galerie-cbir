"""
Configuration centrale du projet « Galerie CBIR ».

Toutes les valeurs peuvent être surchargées par des variables d'environnement,
ce qui permet de lancer l'API et l'application web sur des machines différentes
sans modifier le code.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

# Dossier des données générées (base SQLite, images chargées, miniatures, index).
# Il est recréé automatiquement et n'est PAS livré avec le code.
DATA_DIR = Path(os.environ.get("CBIR_DATA_DIR", BASE_DIR / "data"))

# Emplacement de la base GHIM-10K (10 000 images, 20 classes de 500 images,
# fichiers nommés « <classe>_<numéro>.jpg »). Non livrée avec le code.
GHIM_DIR = Path(os.environ.get("GHIM_DIR", BASE_DIR / "GHIM-10K"))

# --------------------------------------------------------------------------- #
# API REST (Flask + Flask-RESTful)
# --------------------------------------------------------------------------- #
API_HOST = os.environ.get("CBIR_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("CBIR_API_PORT", 5001))
API_URL = os.environ.get("CBIR_API_URL", f"http://{API_HOST}:{API_PORT}")
API_INDEX_DIR = DATA_DIR / "index"

# Dossiers dont l'API accepte de lire des fichiers par chemin (indexation en
# masse). Toute autre lecture par chemin est refusée (sécurité).
API_ALLOWED_ROOTS = [GHIM_DIR.resolve(), (DATA_DIR / "uploads").resolve()]

# --------------------------------------------------------------------------- #
# Application web
# --------------------------------------------------------------------------- #
WEB_HOST = os.environ.get("CBIR_WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.environ.get("CBIR_WEB_PORT", 5000))
SECRET_KEY = os.environ.get("CBIR_SECRET_KEY", "")  # vide : clé aléatoire créée au 1er lancement (voir secret_key())
UPLOAD_DIR = DATA_DIR / "uploads"
THUMB_DIR = DATA_DIR / "thumbs"
DATABASE_URI = os.environ.get("CBIR_DATABASE_URI", f"sqlite:///{DATA_DIR / 'galerie.db'}")
MAX_UPLOAD_MB = int(os.environ.get("CBIR_MAX_UPLOAD_MB", 512))   # dossiers et archives ZIP
# Adresse publique de la galerie (ex. https://xxxx-5000.euw.devtunnels.ms) utilisée
# dans les liens de partage. Peut aussi être saisie dans la page « Mes partages ».
PUBLIC_URL = os.environ.get("CBIR_PUBLIC_URL", "")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"}

# --------------------------------------------------------------------------- #
# Recherche et retour de pertinence
# --------------------------------------------------------------------------- #
# Nombre d'images renvoyées à chaque itération (« scope » dans l'article).
SCOPE = int(os.environ.get("CBIR_SCOPE", 25))

# Méthode de retour de pertinence du groupe :
#   "qvm"   -> Query Vector Modification   (G3, G7, G9, G12, G15)
#   "fre"   -> Feature Relevance Estimation (G1, G5, G10, G13, G14)
#   "type1" -> Re-weighting Type-1          (G2, G4, G6, G8, G11)
# Méthode du groupe G11 : Re-weighting Type-1.
DEFAULT_RF_METHOD = os.environ.get("CBIR_RF_METHOD", "type1")

# Paramètres de la formule de Rocchio (QVM).
QVM_ALPHA = 1.0
QVM_BETA = 0.75
QVM_GAMMA = 0.25

# Petite constante de l'article (équations 4 et 5) pour éviter la division par 0.
RF_EPSILON = 1e-4
# Borne supérieure d'un poids de composante (après normalisation dans son groupe).
RF_MAX_WEIGHT = 10.0


def secret_key() -> str:
    """Clé de signature des sessions Flask.

    Priorité à CBIR_SECRET_KEY ; sinon une clé aléatoire est créée une fois dans
    data/secret_key (jamais versionnée) puis réutilisée : les sessions restent
    valides après un redémarrage et personne ne peut deviner la clé.
    """
    if SECRET_KEY:
        return SECRET_KEY
    import secrets
    f = DATA_DIR / "secret_key"
    if f.exists():
        key = f.read_text(encoding="utf-8").strip()
        if key:
            return key
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    f.write_text(key, encoding="utf-8")
    return key


def ensure_dirs() -> None:
    """Crée les dossiers de données s'ils n'existent pas encore."""
    for d in (DATA_DIR, UPLOAD_DIR, THUMB_DIR, API_INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)
