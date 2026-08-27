# ------------------------------------------------------------------------------
# bootstrap.R — Amorcage commun a tous les scripts du pipeline.
#
# Usage (depuis la RACINE du projet) :
#     source("R/common/bootstrap.R")
#
# Charge config.R puis utils_image.R. config.R determine lui-meme la racine du
# projet a partir de son propre chemin, donc tous les chemins restent corrects
# meme si le repertoire de travail change ensuite.
# ------------------------------------------------------------------------------

if (!file.exists("R/common/config.R")) {
  stop(
    "Fichier R/common/config.R introuvable.\n",
    "Executez les scripts depuis la racine du projet, par exemple :\n",
    "    setwd('/chemin/vers/dog-breed-r'); source('R/01_download_data.R')\n",
    "ou definissez la variable d'environnement DOGCLF_ROOT.",
    call. = FALSE
  )
}

source("R/common/config.R")
source("R/common/utils_image.R")
