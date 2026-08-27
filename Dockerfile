# ==============================================================================
# Dockerfile -- Deploiement de l'API R (plumber) et de la WebApp (Shiny)
#
# Construction :
#     docker build -t dogclf-r .
#
# API seule :
#     docker run -p 5000:5000 -v "$(pwd)/models:/app/models:ro" dogclf-r
#
# WebApp seule :
#     docker run -p 3838:3838 -v "$(pwd)/models:/app/models:ro" dogclf-r app
#
# Le dossier models/ est monte en lecture seule plutot que copie dans l'image :
# le modele pese plusieurs dizaines de Mo et change a chaque reentrainement, il
# n'a pas a declencher la reconstruction de toute l'image.
# ==============================================================================

FROM rocker/r-ver:4.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    DOGCLF_ROOT=/app \
    RETICULATE_PYTHON=/opt/venv/bin/python

# --- Dependances systeme ------------------------------------------------------
# libmagick++ : moteur d'ImageMagick derriere le paquet R magick
# libcurl / libssl : httr2, plumber, telechargements
# python3-venv : environnement TensorFlow pilote par reticulate
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagick++-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libpng-dev \
        libjpeg-dev \
        libtiff5-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        python3 python3-pip python3-venv \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- Paquets R ----------------------------------------------------------------
# Installes avant la copie du code : cette couche est mise en cache et n'est
# reconstruite que si la liste des paquets change.
RUN install2.r --error --skipinstalled \
        keras3 tensorflow tfdatasets reticulate \
        magick digest jsonlite data.table \
        ggplot2 scales \
        plumber shiny bslib httr2 curl base64enc

# --- Backend Python TensorFlow ------------------------------------------------
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir "tensorflow-cpu==2.17.*"

WORKDIR /app

# --- Code applicatif ----------------------------------------------------------
COPY R/     /app/R/
COPY api/   /app/api/
COPY app/   /app/app/
COPY data/breeds_fr.csv /app/data/breeds_fr.csv

# models/ est attendu en volume monte. Le repertoire est cree pour que l'API
# demarre proprement en mode degrade si aucun volume n'est fourni.
RUN mkdir -p /app/models /app/reports

EXPOSE 5000 3838

# --- Sonde de sante -----------------------------------------------------------
# Utilise l'endpoint /api/health, qui renvoie 503 tant que le modele n'est pas
# charge : l'orchestrateur ne routera pas de trafic vers un conteneur inapte.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5000/api/health || exit 1

COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["api"]
