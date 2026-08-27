# ==============================================================================
# 07_predict.R -- PREDICTION
#
# Classifie une ou plusieurs images depuis la ligne de commande, et produit une
# figure recapitulant les meilleurs candidats.
#
# Trois points d'attention, chacun correspondant a une erreur facile a commettre
# et impossible a reperer sans verification :
#
#   1. Pas de softmax supplementaire. La couche de sortie du modele en applique
#      deja un ; en reappliquer un ecraserait toutes les confiances vers 1/120
#      et rendrait les pourcentages affiches denues de sens -- sans changer le
#      classement, donc sans que rien ne paraisse anormal.
#
#   2. Pas de division par 255 : la mise a l'echelle appartient au modele. Le
#      code appelant fournit des pixels bruts 0-255.
#
#   3. Top-3 plutot que top-1, coherent avec ce que renvoie l'API et pertinent
#      vu les groupes de races indiscernables identifies a l'audit (P9).
#
# Usage :
#     Rscript R/07_predict.R chemin/vers/image.jpg [autre.jpg ...]
#   ou, en session :
#     source("R/07_predict.R"); predict_image("photo.jpg")
# ==============================================================================

source("R/common/bootstrap.R")
source("R/common/predict_service.R")

suppressPackageStartupMessages({
  library(magick)
  library(ggplot2)
  library(data.table)
})

# ------------------------------------------------------------------------------
# Prediction sur une image + rendu
# ------------------------------------------------------------------------------

#' Classifie une image et affiche le resultat.
#'
#' @param img_path chemin de l'image
#' @param plot TRUE pour produire une figure (image annotee + barres de score)
#' @param save_to chemin de sauvegarde de la figure, ou NULL pour affichage seul
#' @return la liste renvoyee par classify_dog_breed(), invisiblement
predict_image <- function(img_path, plot = TRUE, save_to = NULL) {

  if (!file.exists(img_path)) stop("Fichier introuvable : ", img_path, call. = FALSE)

  res <- classify_dog_breed(img_path, top_k = TOP_K)

  top <- res$top_prediction
  cat(sprintf("\n%s\n", img_path))
  cat(sprintf("  Predicted class: %s (%.2f%%)  [%s]\n",
              top$class_label, 100 * top$confidence, top$class_name_fr))
  for (i in seq_along(res$predictions)) {
    p <- res$predictions[[i]]
    cat(sprintf("    %d. %-32s %6.2f%%   %s\n", i, p$class_label,
                100 * p$confidence, p$class_name_fr))
  }
  cat(sprintf("  (inference : %.1f ms)\n", res$inference_ms))

  if (plot) {
    dt <- data.table(
      race       = vapply(res$predictions, function(p) p$class_name_fr, character(1)),
      confiance  = vapply(res$predictions, function(p) p$confidence, numeric(1))
    )
    # Le titre doit reprendre la PREMIERE ligne (la prediction la mieux classee).
    # `res$predictions` est deja trie par confiance decroissante.
    titre <- sprintf("%s -- %.1f %%", dt$race[1], 100 * dt$confiance[1])

    # Les niveaux sont inverses pour que coord_flip() affiche le meilleur en
    # haut ; cela ne change pas l'ordre des lignes du tableau.
    dt[, race := factor(race, levels = rev(race))]

    g <- ggplot(dt, aes(x = race, y = confiance)) +
      geom_col(fill = "#2c7fb8") +
      geom_text(aes(label = sprintf("%.1f %%", 100 * confiance)),
                hjust = -0.1, size = 3.5) +
      coord_flip() +
      scale_y_continuous(labels = scales::percent, limits = c(0, 1.12),
                         breaks = seq(0, 1, 0.25)) +
      labs(title = titre, subtitle = basename(img_path), x = NULL, y = "Confiance") +
      theme_minimal(base_size = 11)

    if (is.null(save_to)) {
      print(g)
      # Apercu de l'image elle-meme dans le viewer.
      print(magick::image_read(img_path) |> magick::image_scale("400"))
    } else {
      ggsave(save_to, g, width = 7, height = 3.5, dpi = 150)
      log_info("Figure ecrite : ", save_to)
    }
  }

  invisible(res)
}

#' Classifie un lot d'images et renvoie un tableau.
predict_batch <- function(paths, top_k = 1L) {
  rbindlist(lapply(paths, function(p) {
    r <- tryCatch(classify_dog_breed(p, top_k = top_k),
                  error = function(e) NULL)
    if (is.null(r)) return(data.table(path = p, race = NA_character_,
                                      race_fr = NA_character_, confiance = NA_real_))
    data.table(path = p,
               race       = r$top_prediction$class_label,
               race_fr    = r$top_prediction$class_name_fr,
               confiance  = r$top_prediction$confidence)
  }))
}

# ------------------------------------------------------------------------------
# Execution en ligne de commande
# ------------------------------------------------------------------------------
# Ce bloc ne s'execute QUE si le fichier a ete lance par Rscript. Un
# source() en session se contente de definir les fonctions ci-dessus.
.invoked_directly <- {
  a <- commandArgs(trailingOnly = FALSE)
  h <- grep("^--file=", a, value = TRUE)
  length(h) > 0L && basename(sub("^--file=", "", h[1])) == "07_predict.R"
}

if (.invoked_directly) {
  args <- commandArgs(trailingOnly = TRUE)

  if (length(args) > 0L) {
    if (!load_service()) {
      stop("Service indisponible. Entrainez le modele (R/05_train_model.R) ",
           "ou placez un fichier .keras valide dans ", DIR_MODELS, call. = FALSE)
    }
    for (a in args) {
      out <- file.path(DIR_REPORTS, paste0("prediction_", tools::file_path_sans_ext(basename(a)), ".png"))
      predict_image(a, plot = TRUE, save_to = out)
    }
  } else {
    # Sans argument, on illustre l'usage sur quelques images du jeu de test,
    # dont les chemins sont resolus depuis le manifeste.
    message("Aucun argument. Exemples tires du jeu de test :\n",
            "  Rscript R/07_predict.R photo1.jpg photo2.jpg")

    if (file.exists(PATH_MANIFEST) && load_service()) {
      man <- fread(PATH_MANIFEST)
      demo <- man[split == "test" &
                    breed_raw %in% c("Chihuahua", "bloodhound", "Irish_terrier"),
                  .SD[1], by = breed_raw]
      for (i in seq_len(nrow(demo))) predict_image(demo$path[i], plot = FALSE)
    }
  }
}
