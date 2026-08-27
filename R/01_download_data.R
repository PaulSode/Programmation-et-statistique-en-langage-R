# ------------------------------------------------------------------------------
# 01_download_data.R — Recuperation du jeu de donnees Stanford Dogs
#
# Telecharge l'archive images.tar (~757 Mo) et l'extrait dans data/raw/.
#
#   - telechargement idempotent : l'archive n'est pas retelechargee a chaque run ;
#   - controle de ce qui a reellement ete extrait (nb de dossiers, nb d'images) ;
#   - tous les chemins viennent de config.R, donc aucun chemin absolu code en
#     dur : le script tourne a l'identique sur n'importe quelle machine.
# ------------------------------------------------------------------------------

source("R/common/bootstrap.R")

log_step("01 — Telechargement du jeu de donnees Stanford Dogs")

# --- Telechargement -----------------------------------------------------------
# Taille attendue de l'archive, verifiee apres coup. Un telechargement tronque
# produit un tar qui s'extrait partiellement sans erreur : on veut le detecter
# ici, pas trois etapes plus loin.
EXPECTED_BYTES <- 793579520

#' Telecharge l'archive, en reprenant un fichier partiel s'il en existe un.
#'
#' curl::multi_download() gere la reprise via l'en-tete HTTP Range et n'impose
#' pas de delai global. C'est indispensable ici : utils::download.file() applique
#' getOption("timeout"), qui vaut 60 secondes par defaut -- soit bien moins que
#' le temps necessaire a 757 Mo sur une liaison ordinaire. Le telechargement
#' echoue alors a mi-parcours en laissant un fichier tronque.
download_archive <- function() {
  if (requireNamespace("curl", quietly = TRUE)) {
    already <- if (file.exists(DATA_TAR)) file.info(DATA_TAR)$size else 0
    if (already > 0) {
      log_info(sprintf("Fichier partiel detecte (%.0f Mo) : reprise du telechargement.",
                       already / 1024^2))
    }
    res <- curl::multi_download(DATA_URL, destfiles = DATA_TAR,
                                resume = TRUE, progress = TRUE)
    if (isTRUE(res$success[1])) return(TRUE)
    log_error("curl : ", res$error[1])
    return(FALSE)
  }

  # Repli sans curl : on relache le delai global, sans quoi l'echec est certain.
  old <- options(timeout = max(3600, getOption("timeout")))
  on.exit(options(old), add = TRUE)
  # mode = "wb" est indispensable sous Windows, sinon l'archive est corrompue
  # par la conversion de fins de ligne.
  tryCatch({
    utils::download.file(DATA_URL, destfile = DATA_TAR, mode = "wb", quiet = FALSE)
    TRUE
  }, error = function(e) {
    log_error("Echec du telechargement : ", conditionMessage(e)); FALSE
  })
}

archive_complete <- function() {
  file.exists(DATA_TAR) && file.info(DATA_TAR)$size >= EXPECTED_BYTES
}

if (archive_complete()) {
  log_info("Archive deja presente et complete : ", DATA_TAR,
           sprintf(" (%.0f Mo)", file.info(DATA_TAR)$size / 1024^2))
} else {
  log_info("Telechargement depuis ", DATA_URL, " (~757 Mo)...")

  # Jusqu'a 3 tentatives : le serveur coupe regulierement les connexions
  # longues, et chaque reprise repart de l'octet atteint.
  for (attempt in 1:3) {
    if (attempt > 1) log_warn("Nouvelle tentative (", attempt, "/3)...")
    download_archive()
    if (archive_complete()) break
  }

  if (!archive_complete()) {
    got <- if (file.exists(DATA_TAR)) file.info(DATA_TAR)$size else 0
    stop(sprintf(paste0("Telechargement incomplet : %.0f Mo sur %.0f Mo attendus.\n",
                        "Relancez ce script -- il reprendra la ou il s'est arrete.\n",
                        "Ou telechargez l'archive manuellement et placez-la dans %s"),
                 got / 1024^2, EXPECTED_BYTES / 1024^2, DIR_RAW), call. = FALSE)
  }
  log_info(sprintf("Archive complete : %.0f Mo", file.info(DATA_TAR)$size / 1024^2))
}

# --- Extraction ---------------------------------------------------------------
already_extracted <- dir.exists(DIR_IMAGES) &&
  length(list.dirs(DIR_IMAGES, recursive = FALSE)) >= N_CLASSES_REF

if (already_extracted) {
  log_info("Images deja extraites dans ", DIR_IMAGES)
} else {
  log_info("Extraction de l'archive vers ", DIR_RAW, "...")
  utils::untar(DATA_TAR, exdir = DIR_RAW)
}

# --- Verification -------------------------------------------------------------
if (!dir.exists(DIR_IMAGES)) {
  stop("Le dossier ", DIR_IMAGES, " est absent apres extraction. ",
       "Verifiez la structure de l'archive.", call. = FALSE)
}

class_dirs <- list.dirs(DIR_IMAGES, recursive = FALSE, full.names = FALSE)
n_images   <- length(list.files(DIR_IMAGES, recursive = TRUE,
                                pattern = "[.]jpe?g$", ignore.case = TRUE))

log_info("Dossiers de classes : ", length(class_dirs), " (attendu : ", N_CLASSES_REF, ")")
log_info("Fichiers images     : ", n_images, " (attendu : 20580)")

if (length(class_dirs) != N_CLASSES_REF) {
  log_warn("Nombre de classes inattendu. L'extraction est peut-etre incomplete.")
}

log_info("Etape 01 terminee. Enchainez avec R/02_audit_data.R")
