# ==============================================================================
# 04_build_datasets.R -- PIPELINE D'ENTREE (tf.data)
#
# Definit le pipeline d'entree tf.data. Ce fichier ne fait que DECLARER des
# fonctions ; il est source par 05_train_model.R et 06_evaluate_model.R.
#
#   build_datasets()  -- construit train / val / test depuis le manifeste nettoye
#   make_dataset()    -- brique de base : chemins + labels -> dataset batche
#   build_data_augmentation() -- bloc d'augmentation integre au modele
# ==============================================================================

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(tfdatasets)
  library(data.table)
})

#' Verifie que le backend Python est disponible, avec un message actionnable.
#'
#' Sans ce garde-fou, la premiere expression touchant `tf$...` remonte une trace
#' Python d'une trentaine de lignes dont on ne tire rien.
require_backend <- function() {
  if (!reticulate::py_module_available("tensorflow")) {
    stop("Backend TensorFlow introuvable.\n",
         "  Installez-le : keras3::install_keras(backend = 'tensorflow')\n",
         "  Voir aussi   : R/00_install_dependencies.R",
         call. = FALSE)
  }
  invisible(TRUE)
}

# Evaluation differee : sourcer ce fichier ne doit pas exiger un backend Python.
# Seul l'appel effectif a un constructeur de dataset en a besoin.
delayedAssign("AUTOTUNE", tensorflow::tf$data$AUTOTUNE)

# ------------------------------------------------------------------------------
# Decodage
# ------------------------------------------------------------------------------

#' Lit un fichier image et renvoie un tenseur float32 (H, W, 3) dans [0, 255].
#'
#' Deux garde-fous explicites :
#'   - channels = 3 force la conversion des images en niveaux de gris et des
#'     PNG avec alpha (probleme P3 de l'audit) ;
#'   - expand_animations = FALSE evite qu'un GIF anime produise un tenseur de
#'     rang 4 qui ferait planter le batch entier (probleme P4).
#'
#' Les valeurs restent dans [0, 255] : la mise a l'echelle est faite par une
#' couche du modele, pas ici. Voir 05_train_model.R.
decode_and_resize <- function(path) {
  raw <- tf$io$read_file(path)
  img <- tf$io$decode_image(raw, channels = 3L, expand_animations = FALSE)
  # tf$image$resize etire l'image sans preserver le ratio d'aspect, exactement
  # comme image_to_tensor() cote service. Entrainement et inference deforment
  # donc les images de facon identique.
  img <- tf$image$resize(img, size = as.integer(IMG_SIZE), method = "bilinear")
  img <- tf$ensure_shape(img, shape = c(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))
  tf$cast(img, tf$float32)
}

# ------------------------------------------------------------------------------
# Construction d'un dataset a partir de chemins + labels
# ------------------------------------------------------------------------------

#' @param paths   vecteur de chemins de fichiers
#' @param labels  vecteur d'entiers (base 0)
#' @param shuffle melanger a chaque epoch (TRUE pour l'entrainement uniquement)
#' @param batch_size taille de lot
#' @param cache   mettre en cache les images decodees en RAM
make_dataset <- function(paths, labels, shuffle = FALSE,
                         batch_size = BATCH_SIZE, cache = FALSE) {

  require_backend()

  # Liste NOMMEE : l'element du dataset est un dictionnaire, la fonction de
  # mapping recoit donc un seul argument. C'est la forme la moins ambigue avec
  # tfdatasets, qui ne desempaquette pas toujours les tuples de la meme facon.
  ds <- tfdatasets::tensor_slices_dataset(list(
    path  = tf$constant(paths,              dtype = tf$string),
    label = tf$constant(as.integer(labels), dtype = tf$int32)
  ))

  ds <- tfdatasets::dataset_map(
    ds,
    function(x) list(decode_and_resize(x$path), x$label),
    num_parallel_calls = AUTOTUNE
  )

  # cache() AVANT shuffle() : le decodage JPEG n'est paye qu'une seule fois,
  # puis chaque epoch ne fait que relire de la memoire. C'est le principal
  # levier de performance sur ce jeu de donnees -- sans cache, l'essentiel du
  # temps d'entrainement part dans le decodage, repete a chaque epoch.
  if (cache) ds <- tfdatasets::dataset_cache(ds)

  if (shuffle) {
    ds <- tfdatasets::dataset_shuffle(ds, buffer_size = 1000L,
                                      reshuffle_each_iteration = TRUE)
  }

  ds <- tfdatasets::dataset_batch(ds, batch_size, drop_remainder = FALSE)
  tfdatasets::dataset_prefetch(ds, buffer_size = AUTOTUNE)
}

# ------------------------------------------------------------------------------
# Voie normale : datasets construits depuis le manifeste nettoye
# ------------------------------------------------------------------------------

#' @param cache_train mettre les images d'entrainement en cache RAM.
#'   ~14 000 images x 224 x 224 x 3 en float32 pesent environ 8 Go : a ne
#'   demander que si la machine suit. Passer a FALSE sinon.
#' @return liste(train, val, test, class_names, class_index, n_classes)
build_datasets <- function(cache_train = FALSE) {

  if (!file.exists(PATH_MANIFEST)) {
    stop("Manifeste nettoye absent. Executez d'abord R/03_clean_data.R", call. = FALSE)
  }

  man <- fread(PATH_MANIFEST)
  idx <- fread(file.path(DIR_DATA, "class_index.csv"))

  # Verrou de coherence : les labels du manifeste doivent correspondre a
  # l'index des classes. Si ce n'est pas le cas, tout le reste est faux.
  stopifnot(
    all(man$label >= 0L),
    max(man$label) == nrow(idx) - 1L,
    identical(sort(unique(man$label)), idx$index)
  )

  tr <- man[split == "train"]
  va <- man[split == "val"]
  te <- man[split == "test"]

  log_info(sprintf("Datasets : train %d / val %d / test %d images, %d classes",
                   nrow(tr), nrow(va), nrow(te), nrow(idx)))

  list(
    train       = make_dataset(tr$path, tr$label, shuffle = TRUE,  cache = cache_train),
    val         = make_dataset(va$path, va$label, shuffle = FALSE, cache = TRUE),
    test        = make_dataset(te$path, te$label, shuffle = FALSE, cache = FALSE),
    class_names = idx$breed_label,
    class_index = idx,
    n_classes   = nrow(idx),
    counts      = list(train = nrow(tr), val = nrow(va), test = nrow(te))
  )
}

# ------------------------------------------------------------------------------
# Augmentation de donnees
# ------------------------------------------------------------------------------

#' Bloc d'augmentation de donnees.
#'
#' Ces couches sont ACTIVES a l'entrainement et INERTES en inference : Keras
#' les court-circuite quand training = FALSE. Elles sont donc integrees au
#' modele lui-meme, ce qui evite d'avoir a les reappliquer -- ou a oublier de
#' les desactiver -- au moment du service.
#'
#' Le contenu est calibre sur le probleme : un chien photographie a l'envers
#' n'existe pas, donc pas de RandomFlip vertical ; les rotations et zooms
#' restent faibles pour ne pas couper la tete de l'animal.
build_data_augmentation <- function() {
  keras3::keras_model_sequential(name = "data_augmentation") |>
    keras3::layer_random_flip("horizontal") |>
    keras3::layer_random_rotation(0.1) |>
    keras3::layer_random_zoom(0.1) |>
    keras3::layer_random_contrast(0.1)   # ajout : robustesse aux expositions
}
