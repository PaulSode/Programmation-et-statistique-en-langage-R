# ==============================================================================
# predict_service.R -- SERVICE DE PREDICTION
#
# Point d'entree unique de l'inference. Utilise par :
#   - R/07_predict.R  (ligne de commande)
#   - api/plumber.R   (API REST)
#   - app/app.R       (WebApp Shiny)
#
# Quatre garanties, chacune destinee a rendre impossible une classe de panne
# silencieuse -- celles qui ne provoquent aucune erreur mais faussent les
# resultats :
#
#  1. AUCUNE PREDICTION FICTIVE. Si le modele est absent ou illisible, le
#     service echoue explicitement. Il ne renvoie jamais de valeur de repli
#     mise en forme comme une vraie reponse : un client ne doit jamais avoir a
#     deviner s'il lit une prediction ou une invention.
#
#  2. UN SEUL CHARGEMENT, dans un environnement dedie plutot que dans des
#     variables globales. Le modele n'est pas recharge a chaque requete.
#
#  3. COHERENCE VERIFIEE AU CHARGEMENT. Le nombre de sorties du modele doit
#     egaler le nombre de noms de classes ; sinon le service refuse de demarrer.
#     Un decalage d'un seul indice rendrait toutes les predictions fausses sans
#     lever la moindre erreur.
#
#  4. PRETRAITEMENT UNIQUE. Voir R/common/utils_image.R : la mise a l'echelle
#     des pixels appartient au modele lui-meme, donc entrainement et service ne
#     peuvent pas diverger.
# ==============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

# Etat du service, isole dans son propre environnement : aucune variable
# globale mutable qui trainerait dans l'espace de travail de l'appelant.
.svc <- new.env(parent = emptyenv())
.svc$model       <- NULL
.svc$class_names <- NULL
.svc$class_index <- NULL
.svc$breeds_fr   <- NULL
.svc$metadata    <- NULL
.svc$loaded_at   <- NULL

# ------------------------------------------------------------------------------
# Chargement
# ------------------------------------------------------------------------------

#' Charge le modele et ses tables de reference. Idempotent.
#'
#' @param force recharge meme si deja charge (utile apres un reentrainement)
#' @return TRUE si le service est operationnel, FALSE sinon
load_service <- function(force = FALSE) {
  if (!is.null(.svc$model) && !force) return(TRUE)

  if (!file.exists(PATH_MODEL)) {
    log_error("Modele introuvable : ", PATH_MODEL)
    return(FALSE)
  }

  ok <- tryCatch({
    suppressPackageStartupMessages(library(keras3))
    .svc$model <- keras3::load_model(PATH_MODEL)
    TRUE
  }, error = function(e) {
    log_error("Chargement du modele impossible : ", conditionMessage(e))
    FALSE
  })
  if (!ok) return(FALSE)

  # --- Noms de classes : ordre = ordre des sorties du modele ------------------
  if (!file.exists(PATH_CLASS_NAMES)) {
    log_error("class_names.json introuvable : ", PATH_CLASS_NAMES,
              ". Sans lui, les indices de sortie ne peuvent pas etre nommes.")
    .svc$model <- NULL
    return(FALSE)
  }
  .svc$class_names <- unlist(jsonlite::read_json(PATH_CLASS_NAMES))

  # --- Verrou de coherence ----------------------------------------------------
  # Verrou indispensable : sans lui, une liste de classes desynchronisee du
  # modele produirait des predictions fausses en silence. Mieux vaut refuser de
  # demarrer que servir des resultats faux.
  n_out <- as.integer(tail(unlist(.svc$model$output_shape), 1))
  if (length(.svc$class_names) != n_out) {
    log_error(sprintf(
      "Incoherence : le modele a %d sorties mais class_names.json contient %d noms.",
      n_out, length(.svc$class_names)))
    .svc$model <- NULL
    return(FALSE)
  }

  # --- Tables optionnelles ----------------------------------------------------
  ci_path <- file.path(DIR_MODELS, "class_index.csv")
  if (!file.exists(ci_path)) ci_path <- file.path(DIR_DATA, "class_index.csv")
  if (file.exists(ci_path)) {
    .svc$class_index <- fread(ci_path, encoding = "UTF-8")
  } else {
    log_warn("class_index.csv absent : les synsets ImageNet ne seront pas renvoyes.")
    .svc$class_index <- NULL
  }

  if (file.exists(PATH_BREEDS_FR)) {
    .svc$breeds_fr <- fread(PATH_BREEDS_FR, encoding = "UTF-8")
  } else {
    log_warn("breeds_fr.csv absent : pas de traduction francaise.")
    .svc$breeds_fr <- NULL
  }

  .svc$metadata <- if (file.exists(PATH_METADATA))
    jsonlite::read_json(PATH_METADATA) else NULL

  .svc$loaded_at <- Sys.time()
  log_info("Service pret : ", length(.svc$class_names), " classes, modele charge depuis ",
           PATH_MODEL)
  TRUE
}

#' Le service peut-il servir des predictions ?
service_ready <- function() !is.null(.svc$model)

#' Etat detaille, pour l'endpoint /api/health.
service_status <- function() {
  list(
    ready        = service_ready(),
    model_path   = PATH_MODEL,
    model_exists = file.exists(PATH_MODEL),
    n_classes    = if (is.null(.svc$class_names)) NA_integer_ else length(.svc$class_names),
    loaded_at    = if (is.null(.svc$loaded_at)) NA_character_
                   else format(.svc$loaded_at, "%Y-%m-%dT%H:%M:%S"),
    translations = !is.null(.svc$breeds_fr),
    synsets      = !is.null(.svc$class_index),
    metadata     = .svc$metadata
  )
}

# ------------------------------------------------------------------------------
# Enrichissement d'une classe
# ------------------------------------------------------------------------------

#' Construit la description complete d'une classe a partir de son indice.
#' @param k indice base 0 (comme les sorties Keras)
describe_class <- function(k) {
  label <- .svc$class_names[k + 1L]

  synset <- NA_character_
  raw    <- label
  if (!is.null(.svc$class_index)) {
    row <- .svc$class_index[index == k]
    if (nrow(row) == 1L) {
      synset <- row$synset[1]
      raw    <- row$breed_raw[1]
    }
  }

  fr <- NA_character_
  if (!is.null(.svc$breeds_fr)) {
    row <- .svc$breeds_fr[breed_label == label]
    if (nrow(row) >= 1L) fr <- row$breed_fr[1]
  }

  list(
    class_index = k,
    class_id    = if (is.na(synset)) NA_character_ else synset,
    class_name  = raw,
    class_label = label,
    class_name_fr = if (is.na(fr)) label else fr,
    # Identifiant complet "<synset>-<race>", pratique pour un client qui veut
    # joindre les metadonnees ImageNet. Le synset est bien celui de la race
    # predite ; il vaut NA plutot qu'une valeur inventee quand class_index.csv
    # est absent.
    synset      = if (is.na(synset)) NA_character_ else paste0(synset, "-", raw)
  )
}

# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------

#' Classifie une image.
#'
#' @param src chemin de fichier (character) ou contenu binaire (raw)
#' @param top_k nombre de candidats a renvoyer
#' @return liste : model_type, predictions (liste de top_k elements),
#'   top_prediction, inference_ms
#' @details Leve une erreur si le service n'est pas pret ou si l'image est
#'   indecodable. Aucune valeur de repli n'est inventee.
classify_dog_breed <- function(src, top_k = TOP_K) {

  if (!service_ready() && !load_service()) {
    stop("Le modele n'est pas disponible. Verifiez ", PATH_MODEL,
         " puis relancez le service.", call. = FALSE)
  }

  # --- Pretraitement ----------------------------------------------------------
  x <- tryCatch(
    image_to_tensor(src, IMG_SIZE),
    error = function(e) stop("Image indecodable : ", conditionMessage(e), call. = FALSE)
  )

  # --- Inference --------------------------------------------------------------
  t0 <- Sys.time()
  probs <- predict(.svc$model, x, verbose = 0L)
  ms <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000

  probs <- as.numeric(probs[1, ])

  # La sortie est deja une distribution de probabilite : la couche finale du
  # modele est un softmax. Ne PAS en appliquer un second -- cela laisserait le
  # classement intact mais tasserait toutes les confiances autour de 1/120,
  # rendant les pourcentages affiches denues de sens.
  top <- order(probs, decreasing = TRUE)[seq_len(min(top_k, length(probs)))]

  predictions <- lapply(top, function(i) {
    k <- i - 1L                                  # R indexe a 1, Keras a 0
    c(describe_class(k), list(confidence = round(probs[i], 6)))
  })

  list(
    model_type     = "dog_breed_classifier",
    predictions    = predictions,
    top_prediction = predictions[[1]],
    inference_ms   = round(ms, 1)
  )
}

#' Liste des races connues du modele.
list_breeds <- function() {
  if (!service_ready() && !load_service()) {
    stop("Le modele n'est pas disponible.", call. = FALSE)
  }
  lapply(seq_along(.svc$class_names) - 1L, describe_class)
}

# ------------------------------------------------------------------------------
# Classification binaire chat / chien
# ------------------------------------------------------------------------------
# Volontairement non implementee : aucun modele binaire n'est entraine dans ce
# projet. L'API repond 501 sur ce type d'analyse plutot que de renvoyer une
# valeur plausible mais constante -- un 501 est une information exploitable par
# un client, une constante deguisee en prediction ne l'est pas.
