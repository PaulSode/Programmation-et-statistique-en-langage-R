# ==============================================================================
# api/run_api.R -- Lanceur de l'API
#
# Usage :
#     Rscript api/run_api.R
#     Rscript api/run_api.R --port 8000
#
# Variables d'environnement reconnues :
#     DOGCLF_API_HOST   defaut 0.0.0.0
#     DOGCLF_API_PORT   defaut 5000
#     DOGCLF_ROOT       racine du projet (deduite automatiquement sinon)
# ==============================================================================

# --- Racine du projet, deduite de l'emplacement de ce fichier -----------------
.script <- {
  a <- commandArgs(trailingOnly = FALSE)
  h <- grep("^--file=", a, value = TRUE)
  if (length(h)) sub("^--file=", "", h[1]) else "api/run_api.R"
}
root <- normalizePath(file.path(dirname(.script), ".."), winslash = "/", mustWork = TRUE)
Sys.setenv(DOGCLF_ROOT = root)
setwd(root)

source(file.path(root, "R", "common", "config.R"))

suppressPackageStartupMessages(library(plumber))

# --- Arguments de ligne de commande -------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- which(args == flag)
  if (length(i) && length(args) > i[1]) args[i[1] + 1L] else default
}
host <- get_arg("--host", API_HOST)
port <- as.integer(get_arg("--port", API_PORT))

# --- Verifications de demarrage -----------------------------------------------
# On previent AVANT de lancer le serveur plutot que de laisser l'API demarrer
# puis echouer requete par requete.
if (!file.exists(PATH_MODEL)) {
  log_warn("Modele absent : ", PATH_MODEL)
  log_warn("L'API demarrera en mode degrade (503 sur /api/predict).")
  log_warn("Entrainez le modele avec : Rscript R/05_train_model.R")
}
if (!file.exists(PATH_CLASS_NAMES)) {
  log_warn("class_names.json absent : ", PATH_CLASS_NAMES)
}

cat(sprintf("
+---------------------------------------------------------------+
|  API de classification de races de chiens (plumber)           |
+---------------------------------------------------------------+
|  Ecoute            http://%s:%-4d                       |
|  Documentation     http://127.0.0.1:%d/__docs__/              |
|  Sonde de sante    http://127.0.0.1:%d/api/health             |
+---------------------------------------------------------------+

Exemple d'appel :
  curl -X POST http://127.0.0.1:%d/api/predict -F \"image=@photo.jpg\"

", host, port, port, port, port))

# --- Lancement ----------------------------------------------------------------
pr <- plumber::plumb(file.path(root, "api", "plumber.R"))

# Taille maximale du corps des requetes, alignee sur MAX_UPLOAD_MB. La valeur
# est appliquee par httpuv en amont de plumber : une image trop grosse est
# rejetee avant meme d'etre chargee en memoire.
options(plumber.maxRequestSize = MAX_UPLOAD_MB * 1024^2)

pr$run(host = host, port = port, docs = TRUE)
