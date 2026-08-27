# ------------------------------------------------------------------------------
# utils_image.R — Lecture, inspection et vectorisation des images
#
# Ce fichier contient l'UNIQUE implementation du pretraitement d'inference du
# projet. Elle est utilisee par :
#   - 06_evaluate_model.R  (evaluation)
#   - 07_predict.R         (prediction en lot / ligne de commande)
#   - api/plumber.R        (API REST)
#   - app/app.R            (WebApp Shiny)
#
# Pourquoi une seule implementation : des que deux chemins de code preparent les
# images differemment -- l'un normalisant vers [-1, 1], l'autre vers [0, 1] par
# exemple --, le modele recoit en service des entrees d'une distribution qu'il
# n'a jamais vue a l'entrainement. Rien ne plante : les predictions deviennent
# simplement fausses, et le classement reste plausible.
#
# La parade retenue va plus loin qu'une simple fonction partagee : la mise a
# l'echelle des pixels est DEPLACEE A L'INTERIEUR du modele (layer_rescaling,
# voir 05_train_model.R). Le pretraitement externe se limite alors a "decoder,
# convertir en RGB, redimensionner, rendre un tableau 0-255", et le fichier
# .keras devient autosuffisant : quiconque le charge obtient forcement la bonne
# normalisation. Le desalignement devient structurellement impossible.
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(magick)
})

# --- Inspection sans decodage complet -----------------------------------------

#' Lit les metadonnees d'un fichier image sans jamais lever d'exception.
#'
#' @param path chemin du fichier
#' @return liste : ok, format, width, height, colorspace, channels, n_frames, error
inspect_image <- function(path) {
  fail <- function(msg) list(
    ok = FALSE, format = NA_character_, width = NA_integer_, height = NA_integer_,
    colorspace = NA_character_, n_frames = NA_integer_, error = msg
  )

  if (!file.exists(path)) return(fail("fichier absent"))
  size_mb <- file.info(path)$size / 1024^2
  if (is.na(size_mb) || size_mb == 0)      return(fail("fichier vide"))
  if (size_mb > MAX_FILE_MB)               return(fail(sprintf("fichier > %s Mo", MAX_FILE_MB)))

  info <- tryCatch(
    {
      img <- magick::image_read(path)
      on.exit(rm(img), add = TRUE)
      magick::image_info(img)
    },
    error   = function(e) e,
    warning = function(w) w
  )

  if (inherits(info, "condition")) return(fail(conditionMessage(info)))
  if (!is.data.frame(info) || nrow(info) == 0) return(fail("metadonnees illisibles"))

  list(
    ok         = TRUE,
    format     = as.character(info$format[1]),
    width      = as.integer(info$width[1]),
    height     = as.integer(info$height[1]),
    colorspace = as.character(info$colorspace[1]),
    n_frames   = nrow(info),          # > 1 => GIF anime / TIFF multipage
    error      = NA_character_
  )
}

# --- Pretraitement d'inference -------------------------------------------------

#' Decode une image et renvoie un tableau 4D pret pour le modele.
#'
#' @param src chemin de fichier (character) OU contenu binaire (raw), tel que
#'   recu par l'API depuis un upload multipart.
#' @param target_size c(hauteur, largeur)
#' @return tableau numerique de dimensions (1, H, W, 3), valeurs dans [0, 255].
#'   La mise a l'echelle finale est faite PAR LE MODELE (layer_rescaling).
image_to_tensor <- function(src, target_size = IMG_SIZE) {
  img <- magick::image_read(src)

  # Une image animee ou multipage donne plusieurs frames : on garde la premiere.
  if (length(img) > 1L) img <- img[1]

  # Force 3 canaux. Le dataset contient des images en niveaux de gris, en CMYK
  # et des PNG avec canal alpha ; sans cette conversion, image_data() renverrait
  # 1, 4 ou 4 canaux et le tableau final n'aurait pas la bonne forme.
  img <- magick::image_convert(img, colorspace = "sRGB")
  img <- magick::image_flatten(img)   # aplatit un eventuel canal alpha sur fond

  # "!" force les dimensions exactes en ignorant le ratio d'aspect : c'est
  # precisement ce que fait image_dataset_from_directory() cote entrainement,
  # donc train et serving deforment les images de la meme facon.
  geom <- sprintf("%dx%d!", target_size[2], target_size[1])   # magick = LxH
  img  <- magick::image_resize(img, geom)

  # image_data() renvoie un tableau `raw` de classe "bitmap", de dimensions
  # (canaux, largeur, hauteur), indexable en d[canal, x, y].
  d <- magick::image_data(img, channels = "rgb")

  # PIEGE : magick definit une methode as.integer.bitmap() qui TRANSPOSE
  # silencieusement le tableau en (hauteur, largeur, canaux). Appeler
  # as.integer() puis reimposer dim(3, W, H) melange donc les axes et les
  # canaux -- l'image transmise au modele devient du bruit, sans aucune erreur.
  # unclass() court-circuite cette methode : on recupere l'ordre memoire natif,
  # qu'on reorganise nous-memes, explicitement.
  arr <- as.integer(unclass(d))
  dim(arr) <- c(3L, target_size[2], target_size[1])            # (C, W, H)
  arr <- aperm(arr, c(3L, 2L, 1L))                             # -> (H, W, C)

  out <- array(as.numeric(arr), dim = c(1L, target_size[1], target_size[2], 3L))
  stopifnot(identical(dim(out), c(1L, target_size[1], target_size[2], 3L)))
  out
}

#' Empile plusieurs images en un seul batch (utile pour l'evaluation).
images_to_batch <- function(paths, target_size = IMG_SIZE) {
  n   <- length(paths)
  out <- array(0, dim = c(n, target_size[1], target_size[2], 3L))
  for (i in seq_len(n)) out[i, , , ] <- image_to_tensor(paths[i], target_size)[1, , , ]
  out
}

# --- Noms de classes -----------------------------------------------------------

#' Extrait l'identifiant synset ImageNet d'un nom de dossier Stanford Dogs.
#' "n02085620-Chihuahua" -> "n02085620"
synset_from_dirname <- function(x) sub("^(n[0-9]+)-.*$", "\\1", x)

#' Extrait le nom de race brut d'un nom de dossier Stanford Dogs.
#' "n02085620-Chihuahua" -> "Chihuahua"
breed_from_dirname <- function(x) sub("^n[0-9]+-", "", x)

#' Rend un nom de race lisible par un humain.
#' "German_short-haired_pointer" -> "German short-haired pointer"
pretty_breed <- function(x) {
  x <- gsub("_", " ", x)
  trimws(x)
}
