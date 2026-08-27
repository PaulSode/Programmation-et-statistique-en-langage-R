# ==============================================================================
# api/plumber.R -- API REST de classification de races de chiens
#
# plumber transforme des fonctions R annotees en endpoints HTTP.
# Lancement : Rscript api/run_api.R
#
#   POST /api/predict   multipart/form-data, champ "image"
#   -> { model_type, predictions: [...], top_prediction: {...} }
#
#   GET /api/health     etat reel du service (sonde de disponibilite)
#   GET /api/breeds     les 120 races connues du modele
#   GET /api/metadata   provenance et contrat d'entree du modele deploye
#   GET /__docs__/      documentation OpenAPI interactive, generee par plumber
#
# Principe directeur des reponses : une erreur doit toujours etre distinguable
# d'un resultat. L'API ne renvoie jamais de valeur de repli mise en forme comme
# une prediction ; elle repond 503 si le modele manque, 400 si l'entree est
# invalide, 501 si la fonctionnalite n'existe pas.
# ==============================================================================

# --- Amorcage -----------------------------------------------------------------
# run_api.R positionne DOGCLF_ROOT. En cas de lancement direct, on suppose que
# le repertoire de travail est la racine du projet.
if (!nzchar(Sys.getenv("DOGCLF_ROOT")) && file.exists("R/common/config.R")) {
  Sys.setenv(DOGCLF_ROOT = normalizePath(".", winslash = "/"))
}
.root <- Sys.getenv("DOGCLF_ROOT", unset = normalizePath(".", winslash = "/"))

source(file.path(.root, "R", "common", "config.R"))
source(file.path(.root, "R", "common", "utils_image.R"))
source(file.path(.root, "R", "common", "predict_service.R"))

suppressPackageStartupMessages({
  library(plumber)
  library(jsonlite)
})

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0L) b else a

# Chargement au demarrage, pas a la premiere requete : le premier client ne
# doit pas payer les ~10 s de chargement de TensorFlow, et la latence doit etre
# previsible d'un appel a l'autre.
log_step("Demarrage de l'API")
.ready <- load_service()
if (!.ready) {
  log_warn("Le modele n'a pas pu etre charge. L'API demarre quand meme et ",
           "repondra 503 sur /api/predict. Consultez GET /api/health.")
}

# ------------------------------------------------------------------------------
# Metadonnees OpenAPI
# ------------------------------------------------------------------------------

#* @apiTitle Classification de races de chiens
#* @apiDescription API de classification d'images en 120 races de chiens (jeu Stanford Dogs), servie par un modele MobileNetV2 affine par transfer learning. Le pretraitement des pixels est integre au modele : envoyez l'image telle quelle.
#* @apiVersion 1.0.0

# ------------------------------------------------------------------------------
# Filtre CORS
# ------------------------------------------------------------------------------
# Les navigateurs refusent "Access-Control-Allow-Origin: *" des lors que la
# requete transporte des credentials. Il faut donc renvoyer l'origine exacte de
# la requete, apres l'avoir validee contre la liste blanche CORS_ORIGINS.

#* @filter cors
function(req, res) {
  origin <- req$HTTP_ORIGIN

  if (!is.null(origin) && origin %in% CORS_ORIGINS) {
    res$setHeader("Access-Control-Allow-Origin", origin)
    res$setHeader("Access-Control-Allow-Credentials", "true")
    res$setHeader("Vary", "Origin")
  }

  if (identical(req$REQUEST_METHOD, "OPTIONS")) {
    res$setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    res$setHeader("Access-Control-Allow-Headers",
                  req$HTTP_ACCESS_CONTROL_REQUEST_HEADERS %||% "Content-Type")
    res$setHeader("Access-Control-Max-Age", "86400")
    res$status <- 204L
    return(list())
  }

  plumber::forward()
}

#* @filter logger
function(req) {
  cat(sprintf("[%s] %s %s\n", format(Sys.time(), "%H:%M:%S"),
              req$REQUEST_METHOD, req$PATH_INFO))
  plumber::forward()
}

# ------------------------------------------------------------------------------
# Extraction du fichier envoye
# ------------------------------------------------------------------------------

#' Recupere le contenu binaire de l'image d'une requete.
#'
#' Accepte trois formes, dans cet ordre :
#'   1. multipart/form-data avec un champ "image" (ce qu'envoie le frontend) ;
#'   2. multipart/form-data avec un champ nomme autrement (premiere piece
#'      binaire trouvee) ;
#'   3. corps JSON { "image_base64": "..." }, pratique pour les clients qui ne
#'      savent pas construire de multipart.
#'
#' @return liste(raw, filename) ou NULL
extract_image <- function(req) {
  body <- req$body

  as_raw <- function(part) {
    if (is.raw(part)) return(part)
    if (is.list(part)) {
      if (!is.null(part$value) && is.raw(part$value)) return(part$value)
      if (!is.null(part$body)  && is.raw(part$body))  return(part$body)
    }
    NULL
  }
  name_of <- function(part, fallback) {
    if (is.list(part) && !is.null(part$filename)) part$filename else fallback
  }

  if (is.list(body)) {
    # 1. champ "image", le nom attendu par defaut
    if (!is.null(body$image)) {
      r <- as_raw(body$image)
      if (!is.null(r)) return(list(raw = r, filename = name_of(body$image, "image")))
    }
    # 3. base64
    if (!is.null(body$image_base64) && is.character(body$image_base64)) {
      r <- tryCatch(jsonlite::base64_dec(body$image_base64), error = function(e) NULL)
      if (!is.null(r)) return(list(raw = r, filename = "upload.b64"))
    }
    # 2. n'importe quelle piece binaire
    for (nm in names(body)) {
      r <- as_raw(body[[nm]])
      if (!is.null(r) && length(r) > 0L) return(list(raw = r, filename = name_of(body[[nm]], nm)))
    }
  }

  # Corps binaire brut (Content-Type: image/jpeg)
  if (is.raw(body) && length(body) > 0L) return(list(raw = body, filename = "upload"))

  NULL
}

#' Reponse d'erreur homogene.
error_response <- function(res, status, message, detail = NULL) {
  res$status <- status
  out <- list(error = message, status = status)
  if (!is.null(detail)) out$detail <- detail
  out
}

# ------------------------------------------------------------------------------
# POST /api/predict
# ------------------------------------------------------------------------------

#* Classifie la race du chien present sur une image.
#*
#* @param type:str Type d'analyse. Seul "dog_breed" est implemente.
#* @param top_k:int Nombre de candidats a renvoyer (1 a 10, defaut 3).
#* @parser multi
#* @parser octet
#* @parser json
#* @serializer unboxedJSON list(na = "null")
#* @post /api/predict
function(req, res, type = "dog_breed", top_k = TOP_K) {

  # --- Type d'analyse ---------------------------------------------------------
  if (identical(type, "binary")) {
    # Aucun modele binaire n'est entraine dans ce projet. Un 501 dit au client
    # que la fonctionnalite n'existe pas ; renvoyer une valeur plausible mais
    # constante lui ferait croire a un resultat.
    return(error_response(res, 501L,
      "Classification binaire chat/chien non implementee",
      "Aucun modele binaire n'est entraine dans ce projet."))
  }
  if (!identical(type, "dog_breed")) {
    return(error_response(res, 400L, "Type d'analyse invalide",
                          "Valeurs acceptees : 'dog_breed'."))
  }

  # --- Service disponible ? ---------------------------------------------------
  if (!service_ready() && !load_service()) {
    return(error_response(res, 503L, "Modele indisponible",
      paste0("Le fichier ", basename(PATH_MODEL), " est absent ou illisible. ",
             "Consultez GET /api/health.")))
  }

  # --- Fichier ----------------------------------------------------------------
  img <- extract_image(req)
  if (is.null(img)) {
    return(error_response(res, 400L, "Aucune image envoyee",
      "Attendu : multipart/form-data avec un champ 'image', ou JSON {\"image_base64\": \"...\"}."))
  }

  # Garde-fou de taille : sans lui, un client peut saturer la memoire du
  # serveur avec un seul envoi. httpuv applique deja la meme limite en amont
  # (voir run_api.R) ; ce controle est la seconde ligne de defense.
  size_mb <- length(img$raw) / 1024^2
  if (size_mb > MAX_UPLOAD_MB) {
    return(error_response(res, 413L, "Image trop volumineuse",
                          sprintf("%.1f Mo recus, limite %d Mo.", size_mb, MAX_UPLOAD_MB)))
  }

  k <- suppressWarnings(as.integer(top_k))
  if (is.na(k) || k < 1L) k <- TOP_K
  k <- min(k, 10L)

  # --- Inference --------------------------------------------------------------
  result <- tryCatch(
    classify_dog_breed(img$raw, top_k = k),
    error = function(e) e
  )

  if (inherits(result, "error")) {
    msg <- conditionMessage(result)
    # Une image indecodable est une erreur du client (400), pas du serveur :
    # repondre 500 enverrait le client enqueter sur une panne inexistante.
    if (grepl("indecodable|decode|magick", msg, ignore.case = TRUE)) {
      return(error_response(res, 400L, "Image illisible", msg))
    }
    log_error("Echec d'inference : ", msg)
    return(error_response(res, 500L, "Erreur de traitement", msg))
  }

  res$status <- 200L
  list(
    model_type     = result$model_type,
    predictions    = result$predictions,
    top_prediction = result$top_prediction,
    meta = list(
      filename     = img$filename,
      size_kb      = round(length(img$raw) / 1024, 1),
      inference_ms = result$inference_ms,
      top_k        = k
    )
  )
}

# ------------------------------------------------------------------------------
# GET /api/health
# ------------------------------------------------------------------------------

#* Etat du service.
#*
#* Renvoie 200 si le modele est charge et pret, 503 sinon. Un orchestrateur
#* (Docker, Kubernetes) peut s'en servir comme sonde de disponibilite.
#* @serializer unboxedJSON list(na = "null")
#* @get /api/health
function(res) {
  st <- service_status()
  res$status <- if (isTRUE(st$ready)) 200L else 503L
  list(
    status    = if (isTRUE(st$ready)) "ok" else "degraded",
    ready     = st$ready,
    n_classes = st$n_classes,
    loaded_at = st$loaded_at,
    features  = list(traduction_fr = st$translations, synsets = st$synsets),
    r_version = as.character(getRversion()),
    time      = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  )
}

# ------------------------------------------------------------------------------
# GET /api/breeds
# ------------------------------------------------------------------------------

#* Liste des races reconnues par le modele, dans l'ordre de ses sorties.
#* @serializer unboxedJSON list(na = "null")
#* @get /api/breeds
function(res) {
  if (!service_ready() && !load_service()) {
    return(error_response(res, 503L, "Modele indisponible"))
  }
  breeds <- list_breeds()
  list(count = length(breeds), breeds = breeds)
}

# ------------------------------------------------------------------------------
# GET /api/metadata
# ------------------------------------------------------------------------------

#* Provenance et contrat d'entree du modele deploye.
#*
#* Permet a un client de verifier QUEL modele repond, comment il a ete
#* entraine et ce qu'il attend en entree. Sans cet endpoint, le modele servi
#* serait une boite noire : impossible de savoir quelle version repond.
#* @serializer unboxedJSON list(na = "null")
#* @get /api/metadata
function(res) {
  st <- service_status()
  if (is.null(st$metadata)) {
    return(error_response(res, 404L, "Metadonnees absentes",
      "model_metadata.json n'a pas ete trouve. Il est produit par R/05_train_model.R."))
  }
  st$metadata
}
