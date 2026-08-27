# ------------------------------------------------------------------------------
# 00_install_dependencies.R — Installation de l'environnement R
#
# Deux briques distinctes :
#   1. des paquets R purs (CRAN) ;
#   2. le backend Python TensorFlow, installe par keras3 dans un venv isole
#      via reticulate. R ne reimplemente pas TensorFlow : il le pilote.
# ------------------------------------------------------------------------------

cran_packages <- c(
  # -- coeur modelisation
  "keras3",        # interface R vers Keras 3 (remplace l'ancien paquet 'keras')
  "tensorflow",    # acces bas niveau au graphe TF
  "tfdatasets",    # pipelines tf.data (prefetch, cache, map)
  "reticulate",    # pont R <-> Python

  # -- donnees & images
  "magick",        # lecture/redimensionnement/inspection d'images (ImageMagick)
  "digest",        # hachage MD5 pour la detection de doublons
  "jsonlite",      # lecture/ecriture JSON
  "data.table",    # manipulation rapide du manifeste

  # -- visualisation & rapports
  "ggplot2",
  "scales",

  # -- API & WebApp
  "plumber",       # API REST
  "shiny",         # WebApp
  "bslib",         # theme Bootstrap 5 pour Shiny
  "httr2",         # client HTTP (la WebApp appelle l'API)
  "curl",          # envoi multipart depuis la WebApp vers l'API
  "base64enc"      # encodage des apercus d'image dans la WebApp
)

install_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) == 0L) {
    message("Tous les paquets CRAN sont deja installes.")
    return(invisible(NULL))
  }
  message("Installation de : ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
}

install_if_missing(cran_packages)

# --- Backend Python -----------------------------------------------------------
# A n'executer qu'une seule fois. Cree un environnement virtuel dedie et y
# installe tensorflow + keras. C'est l'etape la plus fragile de l'installation :
# comptez ~1,5 Go, et voir R/common/config.R (DOGCLF_PYTHON) pour placer
# l'environnement sur un disque autre que le disque systeme.
setup_python_backend <- function() {
  message("Installation du backend TensorFlow (peut prendre plusieurs minutes)...")
  keras3::install_keras(backend = "tensorflow")
}

# Verification : affiche la version de TensorFlow reellement chargee.
check_backend <- function() {
  tf <- reticulate::import("tensorflow")
  message("TensorFlow version : ", tf$`__version__`)
  message("Keras version      : ", keras3::keras$`__version__`)
  message("GPU detecte(s)     : ", length(tf$config$list_physical_devices("GPU")))
  invisible(TRUE)
}

if (identical(Sys.getenv("DOGCLF_SETUP_PYTHON"), "1")) {
  setup_python_backend()
  check_backend()
} else {
  message("\nPour installer le backend Python, executez :\n",
          "  Sys.setenv(DOGCLF_SETUP_PYTHON = '1'); source('R/00_install_dependencies.R')\n",
          "ou directement : keras3::install_keras(backend = 'tensorflow')")
}
