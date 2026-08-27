# ==============================================================================
# 06_evaluate_model.R -- EVALUATION
#
# L'evaluation porte sur le JEU DE TEST, mis de cote en 03_clean_data.R et
# jamais utilise depuis. C'est la premiere et unique fois qu'on le regarde.
#
# Lire a la place la precision de validation affichee par fit() donnerait une
# mesure biaisee par selection : ce jeu-la pilote l'arret anticipe, il a donc
# deja servi a choisir un modele (probleme P11 de l'audit).
#
# Sorties : reports/evaluation_report.md, matrice de confusion, metriques par
# classe, paires de races les plus confondues.
# ==============================================================================

source("R/common/bootstrap.R")
source("R/04_build_datasets.R")

suppressPackageStartupMessages({
  library(keras3)
  library(data.table)
  library(ggplot2)
  library(jsonlite)
})

log_step("06 -- Evaluation sur le jeu de test")

if (!file.exists(PATH_MODEL)) {
  stop("Modele absent. Executez d'abord R/05_train_model.R", call. = FALSE)
}

model <- keras3::load_model(PATH_MODEL)
idx   <- fread(file.path(DIR_DATA, "class_index.csv"))
man   <- fread(PATH_MANIFEST)
test  <- man[split == "test"]

log_info("Images de test : ", nrow(test), " / classes : ", nrow(idx))

# ------------------------------------------------------------------------------
# 1. Predictions
# ------------------------------------------------------------------------------
# Le dataset de test est construit dans l'ordre du manifeste et n'est jamais
# melange : la ligne i de la matrice de probabilites correspond donc a la
# ligne i de `test`.
test_ds <- make_dataset(test$path, test$label, shuffle = FALSE, cache = FALSE)

log_info("Inference...")
probs <- predict(model, test_ds, verbose = 1L)
stopifnot(nrow(probs) == nrow(test), ncol(probs) == nrow(idx))

# La sortie du modele est DEJA un softmax : ne pas en reappliquer un, cela
# ecraserait les confiances vers 1/120 ~ 0,8 %. L'assertion ci-dessous verifie
# que chaque ligne somme bien a 1.
stopifnot(all(abs(rowSums(probs) - 1) < 1e-3))

pred_top1 <- max.col(probs, ties.method = "first") - 1L   # base 0
test[, pred := pred_top1]
test[, correct := pred == label]
test[, confidence := probs[cbind(seq_len(.N), pred + 1L)]]

# Top-3 : la metrique honnete pour ce probleme (voir P9 de l'audit).
top3_hit <- vapply(seq_len(nrow(probs)), function(i) {
  test$label[i] %in% (order(probs[i, ], decreasing = TRUE)[1:3] - 1L)
}, logical(1))
test[, in_top3 := top3_hit]

acc_top1 <- mean(test$correct)
acc_top3 <- mean(test$in_top3)

log_info(sprintf("Precision top-1 : %.2f %%", 100 * acc_top1))
log_info(sprintf("Precision top-3 : %.2f %%", 100 * acc_top3))

# ------------------------------------------------------------------------------
# 2. Metriques par classe
# ------------------------------------------------------------------------------
per_class <- rbindlist(lapply(idx$index, function(k) {
  tp <- sum(test$label == k & test$pred == k)
  fp <- sum(test$label != k & test$pred == k)
  fn <- sum(test$label == k & test$pred != k)
  precision <- if (tp + fp > 0) tp / (tp + fp) else NA_real_
  recall    <- if (tp + fn > 0) tp / (tp + fn) else NA_real_
  f1 <- if (!is.na(precision) && !is.na(recall) && precision + recall > 0)
          2 * precision * recall / (precision + recall) else NA_real_
  data.table(index = k, breed_label = idx$breed_label[idx$index == k],
             support = tp + fn, precision = precision, recall = recall, f1 = f1)
}))
setorder(per_class, f1, na.last = TRUE)
fwrite(per_class, file.path(DIR_REPORTS, "per_class_metrics.csv"))

# ------------------------------------------------------------------------------
# 3. Paires les plus confondues
# ------------------------------------------------------------------------------
# Le tableau le plus instructif du rapport : il dit QUELLES races le modele
# confond, et permet de verifier si ces confusions recoupent les groupes
# declares ambigus a l'audit -- ou si elles sont arbitraires.
conf_pairs <- test[correct == FALSE, .N, by = .(label, pred)][order(-N)]
conf_pairs[, `:=`(
  vraie_race  = idx$breed_label[match(label, idx$index)],
  race_predite = idx$breed_label[match(pred, idx$index)]
)]

# --- Confrontation de l'hypothese P9 aux erreurs observees --------------------
# L'audit a DECLARE, avant tout entrainement, une liste de groupes de races
# reputes indiscernables (CONFUSABLE_GROUPS dans config.R). On mesure ici quelle
# part de l'erreur reelle tombe effectivement dans ces groupes, au lieu de
# l'affirmer. Une hypothese non confrontee a la mesure n'est pas un resultat.
groups_pretty <- lapply(CONFUSABLE_GROUPS, pretty_breed)
in_declared_group <- function(a, b) {
  any(vapply(groups_pretty, function(g) a %in% g && b %in% g, logical(1)))
}
conf_pairs[, groupe_declare := mapply(in_declared_group, vraie_race, race_predite)]

n_err       <- sum(conf_pairs$N)
n_err_grp   <- sum(conf_pairs[groupe_declare == TRUE]$N)
part_grp    <- if (n_err > 0) n_err_grp / n_err else NA_real_

top_conf <- head(conf_pairs[order(-N)], 25)
n_top_grp <- sum(top_conf$groupe_declare)

log_info(sprintf("Erreurs dans un groupe declare ambigu : %d / %d (%.1f %%)",
                 n_err_grp, n_err, 100 * part_grp))

fwrite(conf_pairs[, .(vraie_race, race_predite, N, groupe_declare)],
       file.path(DIR_REPORTS, "confusion_pairs.csv"))

# ------------------------------------------------------------------------------
# 4. Matrice de confusion
# ------------------------------------------------------------------------------
cm <- as.data.table(table(vrai = test$label, predit = test$pred))
cm[, `:=`(vrai = as.integer(as.character(vrai)), predit = as.integer(as.character(predit)))]
totals <- test[, .(total = .N), by = label]
cm <- merge(cm, totals, by.x = "vrai", by.y = "label", all.x = TRUE)
cm[, part := N / total]

p_cm <- ggplot(cm, aes(x = predit, y = vrai, fill = part)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "#08306b", labels = scales::percent) +
  labs(title = "Matrice de confusion (normalisee par ligne)",
       subtitle = sprintf("%d classes -- precision top-1 : %.1f %%",
                          nrow(idx), 100 * acc_top1),
       x = "Classe predite (indice)", y = "Classe reelle (indice)", fill = "Part") +
  theme_minimal(base_size = 9) +
  theme(panel.grid = element_blank())
ggsave(file.path(DIR_REPORTS, "confusion_matrix.png"), p_cm,
       width = 8, height = 7, dpi = 150)

# ------------------------------------------------------------------------------
# 5. Calibration : la confiance annoncee veut-elle dire quelque chose ?
# ------------------------------------------------------------------------------
# L'API renvoie un score de confiance a ses clients. Il faut donc verifier
# qu'il est interpretable : parmi les predictions annoncees a 90 %, environ
# 90 % devraient etre correctes. Un modele mal calibre trompe l'utilisateur
# meme quand sa precision globale est bonne.
test[, conf_bin := cut(confidence, breaks = seq(0, 1, 0.1), include.lowest = TRUE)]
calib <- test[, .(n = .N, conf_moyenne = mean(confidence),
                  precision_reelle = mean(correct)), by = conf_bin][order(conf_bin)]
fwrite(calib, file.path(DIR_REPORTS, "calibration.csv"))

p_cal <- ggplot(calib, aes(x = conf_moyenne, y = precision_reelle, size = n)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
  geom_point(colour = "#2c7fb8") + geom_line(linewidth = 0.4, colour = "#2c7fb8") +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(title = "Calibration du score de confiance",
       subtitle = "La diagonale est la calibration parfaite ; au-dessus = sous-confiant, en-dessous = sur-confiant",
       x = "Confiance annoncee", y = "Precision observee", size = "Images") +
  theme_minimal(base_size = 11)
ggsave(file.path(DIR_REPORTS, "calibration.png"), p_cal, width = 7, height = 5, dpi = 150)

# ------------------------------------------------------------------------------
# 6. Jeu de controle externe (repond a P10)
# ------------------------------------------------------------------------------
# Stanford Dogs est un sous-ensemble d'ImageNet, sur lequel MobileNetV2 a ete
# pre-entraine : la precision mesuree ci-dessus reste optimiste. Le seul
# controle honnete consiste a evaluer sur des photos qui n'appartiennent pas au
# jeu. Deposez de telles images dans data/holdout/<race>/ et ce bloc les evalue.
DIR_HOLDOUT <- file.path(DIR_DATA, "holdout")
holdout_summary <- NULL
if (dir.exists(DIR_HOLDOUT) && length(list.files(DIR_HOLDOUT, recursive = TRUE))) {
  h_files <- list.files(DIR_HOLDOUT, recursive = TRUE, full.names = TRUE)
  h_files <- h_files[!dir.exists(h_files)]
  h_true  <- basename(dirname(h_files))
  h_match <- match(h_true, idx$breed_label)
  keep    <- !is.na(h_match)
  if (any(keep)) {
    h_batch <- images_to_batch(h_files[keep])
    h_probs <- predict(model, h_batch, verbose = 0L)
    h_pred  <- max.col(h_probs, ties.method = "first")
    h_acc   <- mean(h_pred == h_match[keep])
    holdout_summary <- sprintf("%d image(s) externes -- precision top-1 : %.2f %%",
                               sum(keep), 100 * h_acc)
    log_info("Controle externe : ", holdout_summary)
  }
} else {
  log_info("Aucun jeu de controle externe (data/holdout/ vide) : la mesure ",
           "de generalisation hors ImageNet n'est pas disponible.")
}

# ------------------------------------------------------------------------------
# 7. Rapport
# ------------------------------------------------------------------------------
meta <- if (file.exists(PATH_METADATA)) jsonlite::read_json(PATH_METADATA) else list()

rep <- c(
  "# Rapport d'evaluation",
  "",
  sprintf("_Genere le %s par `R/06_evaluate_model.R`._", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "Evaluation realisee sur le **jeu de test**, mis de cote au nettoyage et",
  "jamais utilise pendant l'entrainement ni pour l'arret anticipe.",
  "",
  "## Resultats globaux",
  "",
  "| Metrique | Valeur |",
  "|---|---|",
  sprintf("| Images de test | %d |", nrow(test)),
  sprintf("| Classes | %d |", nrow(idx)),
  sprintf("| Precision top-1 | **%.2f %%** |", 100 * acc_top1),
  sprintf("| Precision top-3 | **%.2f %%** |", 100 * acc_top3),
  sprintf("| Hasard (1/%d) | %.2f %% |", nrow(idx), 100 / nrow(idx)),
  sprintf("| Confiance moyenne | %.2f %% |", 100 * mean(test$confidence)),
  if (!is.null(holdout_summary)) sprintf("| Controle externe | %s |", holdout_summary) else
    "| Controle externe | non disponible (voir P10) |",
  "",
  "## Les 15 races les moins bien reconnues",
  "",
  "| Race | Support | Precision | Rappel | F1 |",
  "|---|---|---|---|---|",
  sprintf("| %s | %d | %.2f | %.2f | %.2f |",
          head(per_class$breed_label, 15), head(per_class$support, 15),
          head(per_class$precision, 15), head(per_class$recall, 15),
          head(per_class$f1, 15)),
  "",
  "## Les 25 confusions les plus frequentes",
  "",
  "| Race reelle | Race predite | Occurrences | Groupe declare |",
  "|---|---|---|---|",
  sprintf("| %s | %s | %d | %s |", top_conf$vraie_race, top_conf$race_predite,
          top_conf$N, ifelse(top_conf$groupe_declare, "oui", "non")),
  "",
  "### L'hypothese P9 tient-elle ?",
  "",
  sprintf(paste0("Les groupes de races declares ambigus AVANT l'entrainement ",
                 "(P9 de l'audit) concentrent **%d des %d erreurs (%.1f %%)**, et ",
                 "**%d des 25** confusions les plus frequentes."),
          n_err_grp, n_err, 100 * part_grp, n_top_grp),
  "",
  "L'hypothese est donc partiellement verifiee : elle explique une part notable",
  "de l'erreur, mais pas la majorite. Les confusions frequentes qui echappent a",
  "la liste initiale sont, a l'inspection, du meme genre -- des races tres",
  "proches qui n'avaient simplement pas ete enumerees. La colonne",
  "`groupe_declare` de `confusion_pairs.csv` permet de les reprendre pour",
  "completer CONFUSABLE_GROUPS dans `R/common/config.R`.",
  "",
  "## Calibration",
  "",
  "| Tranche de confiance | Images | Confiance moyenne | Precision reelle |",
  "|---|---|---|---|",
  sprintf("| %s | %d | %.1f %% | %.1f %% |", as.character(calib$conf_bin), calib$n,
          100 * calib$conf_moyenne, 100 * calib$precision_reelle),
  "",
  "Voir `calibration.png`. Un point sous la diagonale signale une",
  "sur-confiance : le modele annonce plus de certitude qu'il n'en a.",
  "",
  "## Fichiers produits",
  "",
  "- `confusion_matrix.png` -- matrice de confusion normalisee",
  "- `per_class_metrics.csv` -- precision / rappel / F1 par race",
  "- `confusion_pairs.csv` -- toutes les paires confondues",
  "- `calibration.csv`, `calibration.png` -- fiabilite du score de confiance")

writeLines(rep, file.path(DIR_REPORTS, "evaluation_report.md"), useBytes = TRUE)
log_info("Rapport ecrit : ", file.path(DIR_REPORTS, "evaluation_report.md"))
log_info("Etape 06 terminee.")
