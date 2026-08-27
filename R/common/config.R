# ------------------------------------------------------------------------------
# config.R — Configuration centrale du projet
#
# Toutes les constantes du pipeline vivent ici. Les scripts numerotes
# (01_ a 07_), l'API plumber et la WebApp Shiny sourcent ce fichier, ce qui
# garantit qu'entrainement et service utilisent EXACTEMENT les memes reglages.
#
# Regle : aucune de ces constantes ne doit etre redefinie ailleurs. Une taille
# d'image ou une convention de normalisation dupliquee dans deux fichiers finit
# toujours par diverger, et le desalignement entrainement / service qui en
# resulte ne produit aucune erreur -- seulement des predictions silencieusement
# fausses.
# ------------------------------------------------------------------------------

# --- Racine du projet ---------------------------------------------------------
# Resolue via la variable d'environnement DOGCLF_ROOT si elle existe, sinon en
# remontant depuis l'emplacement de ce fichier. Evite toute dependance a getwd().
.this_script_path <- function() {
  # 1) fichier passe a Rscript
  args <- commandArgs(trailingOnly = FALSE)
  hit  <- grep("^--file=", args, value = TRUE)
  if (length(hit)) return(normalizePath(sub("^--file=", "", hit[1]), winslash = "/", mustWork = FALSE))

  # 2) fichier en cours de source() : on remonte la pile d'appels
  n <- sys.nframe()
  if (n >= 1) for (i in seq(n, 1L)) {
    f <- sys.frame(i)$ofile
    if (!is.null(f)) return(normalizePath(f, winslash = "/", mustWork = FALSE))
  }
  NA_character_
}

.resolve_project_root <- function() {
  env_root <- Sys.getenv("DOGCLF_ROOT", unset = "")
  if (nzchar(env_root)) return(normalizePath(env_root, winslash = "/", mustWork = FALSE))

  # Ce fichier vit en <root>/R/common/config.R -> on remonte de deux crans.
  this_file <- .this_script_path()
  if (is.na(this_file) || basename(this_file) != "config.R") {
    # Dernier recours : le repertoire de travail est suppose etre la racine.
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(file.path(dirname(this_file), "..", ".."), winslash = "/", mustWork = FALSE)
}

PROJECT_ROOT <- .resolve_project_root()

path_from_root <- function(...) file.path(PROJECT_ROOT, ...)

# --- Repertoires --------------------------------------------------------------
DIR_DATA        <- path_from_root("data")           # manifestes, tables de reference
DIR_RAW         <- path_from_root("data", "raw")    # archive + Images/ brutes
DIR_IMAGES      <- file.path(DIR_RAW, "Images")     # 120 dossiers nXXXXXXXX-race
DIR_MODELS      <- path_from_root("models")         # artefacts entraines
DIR_REPORTS     <- path_from_root("reports")        # audit, figures, metriques

for (d in c(DIR_DATA, DIR_RAW, DIR_MODELS, DIR_REPORTS)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# --- Donnees ------------------------------------------------------------------
DATA_URL      <- "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
DATA_TAR      <- file.path(DIR_RAW, "images.tar")
N_CLASSES_REF <- 120L   # Stanford Dogs : 120 races, 20 580 images

# --- Pretraitement image ------------------------------------------------------
IMG_HEIGHT <- 224L
IMG_WIDTH  <- 224L
IMG_SIZE   <- c(IMG_HEIGHT, IMG_WIDTH)
N_CHANNELS <- 3L

# Filtres du nettoyage (voir 03_clean_data.R)
MIN_SIDE_PX   <- 64L        # une image plus petite que 64 px de cote est du bruit
MAX_FILE_MB   <- 10         # garde-fou lecture
MIN_PER_CLASS <- 50L        # une classe sous ce seuil n'est pas apprenable

# --- Split reproductible ------------------------------------------------------
SEED        <- 123L
SPLIT_TRAIN <- 0.70
SPLIT_VAL   <- 0.15
SPLIT_TEST  <- 0.15         # jamais regarde avant l'evaluation finale

# --- Entrainement -------------------------------------------------------------
BATCH_SIZE        <- 32L
EPOCHS_HEAD       <- 10L    # phase 1 : base gelee
EPOCHS_FINETUNE   <- 8L     # phase 2 : fine-tuning
LR_HEAD           <- 1e-3
LR_FINETUNE       <- 1e-5   # 100x plus faible : on ne detruit pas les features
UNFREEZE_LAST_N   <- 30L    # nb de couches de MobileNetV2 degelees en phase 2
DROPOUT_RATE      <- 0.2
LABEL_SMOOTHING   <- 0.0
TOP_K             <- 3L     # nb de predictions renvoyees par l'API

# --- Artefacts produits -------------------------------------------------------
# class_names.json est la SOURCE UNIQUE DE VERITE de l'ordre des classes. Il
# est ecrit par le meme script que le modele, donc les deux ne peuvent pas
# diverger. Une liste de classes maintenue a la main, decalee d'un seul indice,
# rendrait toutes les predictions fausses sans lever la moindre erreur.
PATH_MODEL        <- file.path(DIR_MODELS, "model.keras")
PATH_CLASS_NAMES  <- file.path(DIR_MODELS, "class_names.json")
PATH_METADATA     <- file.path(DIR_MODELS, "model_metadata.json")
PATH_MANIFEST_RAW <- file.path(DIR_DATA, "manifest_raw.csv")
PATH_MANIFEST     <- file.path(DIR_DATA, "manifest_clean.csv")
PATH_BREEDS_FR    <- file.path(DIR_DATA, "breeds_fr.csv")

# --- API / WebApp -------------------------------------------------------------
API_HOST     <- Sys.getenv("DOGCLF_API_HOST", "0.0.0.0")
API_PORT     <- as.integer(Sys.getenv("DOGCLF_API_PORT", "5000"))
API_BASE_URL <- Sys.getenv("DOGCLF_API_URL", "http://127.0.0.1:5000")
MAX_UPLOAD_MB <- 5          # applique par l'API et par la WebApp

CORS_ORIGINS <- c("http://localhost:3000", "http://localhost:5173",
                  "http://127.0.0.1:3000", "http://127.0.0.1:5173")

# --- Groupes de races visuellement ambigues -----------------------------------
# Hypothese posee AVANT tout entrainement (probleme P9 de l'audit) : ces races
# sont difficiles a distinguer sur une seule photo, parfois parce que le critere
# reel n'y est pas visible (la taille, pour les caniches et les schnauzers).
#
# La liste est definie ici, une seule fois, parce que 02_audit_data.R l'enonce
# et 06_evaluate_model.R la CONFRONTE aux confusions reellement observees. Une
# hypothese qu'on ne mesure pas ensuite ne vaut rien : voir la part d'erreur
# calculee dans le rapport d'evaluation.
CONFUSABLE_GROUPS <- list(
  c("Siberian_husky", "malamute", "Eskimo_dog"),
  c("Italian_greyhound", "whippet"),
  c("Norfolk_terrier", "Norwich_terrier"),
  c("Appenzeller", "EntleBucher", "Greater_Swiss_Mountain_dog", "Bernese_mountain_dog"),
  c("Lhasa", "Shih-Tzu", "Tibetan_terrier"),
  c("toy_poodle", "miniature_poodle", "standard_poodle"),
  c("miniature_schnauzer", "standard_schnauzer", "giant_schnauzer"),
  c("Staffordshire_bullterrier", "American_Staffordshire_terrier"),
  c("Pembroke", "Cardigan"),
  c("collie", "Border_collie", "Shetland_sheepdog")
)

# --- Backend Python -----------------------------------------------------------
# keras3 pilote TensorFlow via reticulate, dans un environnement virtuel Python.
# Par defaut celui-ci est cree sous ~/.virtualenvs/r-keras, donc sur C: sous
# Windows. TensorFlow et ses dependances pesent environ 1,5 Go : sur un disque
# systeme sature, l'installation echoue avec un "No space left on device" au
# beau milieu du telechargement.
#
# DOGCLF_PYTHON permet de pointer vers un interpreteur situe ailleurs :
#
#   # installation, une fois, sur un disque qui a de la place
#   Sys.setenv(WORKON_HOME = "D:/venvs", PIP_CACHE_DIR = "D:/pip-cache")
#   keras3::install_keras(backend = "tensorflow", envname = "D:/venvs/r-keras")
#
#   # utilisation, ensuite
#   Sys.setenv(DOGCLF_PYTHON = "D:/venvs/r-keras/Scripts/python.exe")
#
# Sans cette variable, reticulate applique sa logique de decouverte habituelle.
.python_bin <- Sys.getenv("DOGCLF_PYTHON", unset = "")
if (nzchar(.python_bin)) {
  if (file.exists(.python_bin)) {
    Sys.setenv(RETICULATE_PYTHON = .python_bin)
  } else {
    warning("DOGCLF_PYTHON pointe vers un fichier inexistant : ", .python_bin,
            call. = FALSE)
  }
}

# --- Journalisation -----------------------------------------------------------
log_info  <- function(...) cat(sprintf("[INFO ] %s\n", paste0(...)))
log_warn  <- function(...) cat(sprintf("[WARN ] %s\n", paste0(...)))
log_error <- function(...) cat(sprintf("[ERROR] %s\n", paste0(...)))
log_step  <- function(...) cat(sprintf("\n=== %s ===\n", paste0(...)))
