"""Lance l'application web de la galerie (port 5000 par défaut).

L'API de descripteurs doit être lancée en parallèle :  python run_api.py
"""
import config
from webapp.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=False, threaded=True)
