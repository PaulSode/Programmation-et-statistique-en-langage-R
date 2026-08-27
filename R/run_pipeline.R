# ==============================================================================
# run_pipeline.R -- Orchestration du pipeline complet
#
# Usage (depuis la racine du projet) :
#     Rscript R/run_pipeline.R              # tout, du telechargement a l'evaluation
#     Rscript R/run_pipeline.R --from 03    # reprend a partir du nettoyage
#     Rscript R/run_pipeline.R --only 02    # une seule etape
#
# Chaque etape ecrit ses artefacts sur disque et les suivantes les relisent :
# le pipeline est donc reprenable a n'importe quel point, sans tout rejouer.
# ==============================================================================

if (!file.exists("R/common/config.R")) {
  stop("Executez ce script depuis la racine du projet.", call. = FALSE)
}
source("R/common/config.R")

steps <- c(
  "01" = "R/01_download_data.R",
  "02" = "R/02_audit_data.R",
  "03" = "R/03_clean_data.R",
  "05" = "R/05_train_model.R",
  "06" = "R/06_evaluate_model.R"
)
# 04_build_datasets.R ne definit que des fonctions : il est source par 05 et 06.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag) {
  i <- which(args == flag)
  if (length(i) && length(args) > i[1]) args[i[1] + 1L] else NULL
}

only <- get_arg("--only")
from <- get_arg("--from")

todo <- names(steps)
if (!is.null(only)) {
  todo <- only
} else if (!is.null(from)) {
  todo <- todo[todo >= from]
}

stopifnot(all(todo %in% names(steps)))

t_start <- Sys.time()
for (s in todo) {
  f <- steps[[s]]
  log_step(sprintf("PIPELINE -- etape %s : %s", s, f))
  t0 <- Sys.time()
  # Chaque etape tourne dans son propre environnement pour eviter que des
  # variables d'une etape ne fuient vers la suivante.
  env <- new.env(parent = globalenv())
  source(f, local = env, echo = FALSE)
  log_info(sprintf("Etape %s terminee en %.1f min", s,
                   as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  rm(env); gc(verbose = FALSE)
}

log_step(sprintf("PIPELINE TERMINE en %.1f min",
                 as.numeric(difftime(Sys.time(), t_start, units = "mins"))))

cat("
Artefacts produits :
  models/model.keras           modele entraine (pretraitement inclus)
  models/class_names.json      ordre des classes en sortie
  models/model_metadata.json   provenance et contrat d'entree
  data/manifest_clean.csv      images retenues et leur split
  reports/audit_report.md      problemes du jeu de donnees
  reports/cleaning_report.md   nettoyage applique
  reports/evaluation_report.md resultats sur le jeu de test

Etapes suivantes :
  Rscript api/run_api.R        demarre l'API REST
  Rscript app/run_app.R        demarre la WebApp
")
