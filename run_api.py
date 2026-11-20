"""Lance l'API REST de descripteurs et de recherche (port 5001 par défaut)."""
import logging

import config
from api.app import create_app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [API] %(message)s")
    app = create_app()
    # threaded=True : plusieurs requêtes en parallèle (indexation + recherche)
    app.run(host=config.API_HOST, port=config.API_PORT, debug=False, threaded=True)
