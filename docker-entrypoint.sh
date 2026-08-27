#!/bin/sh
# ------------------------------------------------------------------------------
# Point d'entree du conteneur : "api" (defaut) ou "app".
# ------------------------------------------------------------------------------
set -e

case "${1:-api}" in
  api)
    echo "Demarrage de l'API plumber sur le port ${DOGCLF_API_PORT:-5000}"
    exec Rscript /app/api/run_api.R --host 0.0.0.0 --port "${DOGCLF_API_PORT:-5000}"
    ;;
  app)
    echo "Demarrage de la WebApp Shiny sur le port ${DOGCLF_APP_PORT:-3838}"
    exec Rscript /app/app/run_app.R --host 0.0.0.0 --port "${DOGCLF_APP_PORT:-3838}"
    ;;
  *)
    exec "$@"
    ;;
esac
