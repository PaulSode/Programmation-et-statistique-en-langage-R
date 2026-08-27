# ==============================================================================
# 03_clean_data.R -- NETTOYAGE DES DONNEES
#
#   >>> Ce script repond a la question : "Comment les donnees ont-elles ete
#       nettoyees ?" A chaque probleme Pxx releve par l'audit correspond ici
#       une regle explicite, tracee, et un comptage des lignes ecartees.
#
# Principe : on ne supprime AUCUN fichier sur le disque. Le nettoyage produit
# un manifeste (data/manifest_clean.csv) qui est la seule source de verite du
# pipeline d'entrainement. Toute exclusion est reversible et auditables : la
# colonne `exclusion_reason` dit pourquoi chaque image ecartee l'a ete.
#
# Laisser le chargeur de donnees decider seul de ce qui entre dans le modele
# reviendrait a ne jamais savoir sur quoi on a reellement entraine.
# ==============================================================================

source("R/common/bootstrap.R")

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(ggplot2)
})

log_step("03 -- Nettoyage des donnees")

if (!file.exists(PATH_MANIFEST_RAW)) {
  stop("Manifeste brut absent. Executez d'abord R/02_audit_data.R", call. = FALSE)
}

m <- fread(PATH_MANIFEST_RAW)
n_start <- nrow(m)
log_info("Lignes en entree : ", n_start)

m[, keep := TRUE]
m[, exclusion_reason := NA_character_]

#' Applique une regle d'exclusion et journalise son effet.
#' L'ordre des regles compte : une image n'est comptee que pour la PREMIERE
#' regle qui la rejette, ce qui rend le tableau de bilan additif.
apply_rule <- function(dt, condition, reason) {
  hit <- dt$keep & condition & !is.na(condition)
  dt[hit, `:=`(keep = FALSE, exclusion_reason = reason)]
  log_info(sprintf("  %-46s -> %5d ecartee(s)", reason, sum(hit)))
  invisible(dt)
}

# ------------------------------------------------------------------------------
# REGLE 1 (repond a P2) -- fichiers illisibles
# ------------------------------------------------------------------------------
# Une image que magick ne sait pas decoder ne sera pas non plus decodable par
# le pipeline TensorFlow. On l'ecarte au lieu de laisser TF l'ignorer en
# silence : ainsi le nombre d'images d'entrainement est celui qu'on croit.
apply_rule(m, m$ok == FALSE, "illisible / corrompue")

# ------------------------------------------------------------------------------
# REGLE 2 (repond a P8) -- images trop petites
# ------------------------------------------------------------------------------
apply_rule(m, m$min_side < MIN_SIDE_PX,
           sprintf("cote < %d px", MIN_SIDE_PX))

# ------------------------------------------------------------------------------
# REGLE 3 (repond a P7) -- ratios d'aspect aberrants
# ------------------------------------------------------------------------------
# Au-dela de 4:1, le redimensionnement en 224x224 rend le sujet meconnaissable.
# Le seuil est un choix, pas une verite : il est expose ici pour pouvoir etre
# discute et modifie.
AR_MAX <- 4
apply_rule(m, m$aspect_ratio > AR_MAX | m$aspect_ratio < 1 / AR_MAX,
           sprintf("ratio d'aspect hors [1/%d, %d]", AR_MAX, AR_MAX))

# ------------------------------------------------------------------------------
# REGLE 4 (repond a P5) -- doublons exacts
# ------------------------------------------------------------------------------
# On garde la premiere occurrence de chaque MD5 et on ecarte les suivantes.
# Indispensable AVANT le split : c'est le seul moyen d'empecher qu'une meme
# image se retrouve des deux cotes de la frontiere train / test.
apply_rule(m, m$is_duplicate == TRUE, "doublon exact (MD5)")

# ------------------------------------------------------------------------------
# REGLE 5 (repond a P3) -- images animees / multipage
# ------------------------------------------------------------------------------
apply_rule(m, m$n_frames > 1L, "image multi-frames (GIF anime / TIFF)")

# ------------------------------------------------------------------------------
# NORMALISATION 1 (repond a P1) -- separation synset / nom de race
# ------------------------------------------------------------------------------
# Le nom de dossier "n02085620-Chihuahua" porte deux informations distinctes :
#   - le synset ImageNet, identifiant stable, cle de jointure ;
#   - le nom de race, destine a l'affichage.
# On les separe, et on genere un libelle propre pour l'interface.
m[, `:=`(
  synset      = synset_from_dirname(dirname),
  breed_raw   = breed_from_dirname(dirname),
  breed_label = pretty_breed(breed_from_dirname(dirname))
)]

# ------------------------------------------------------------------------------
# NORMALISATION 2 (repond a P3) -- espace colorimetrique
# ------------------------------------------------------------------------------
# On ne reecrit pas les fichiers : la conversion en sRGB 3 canaux est faite a la
# volee, au moment du decodage, par image_to_tensor() (R/common/utils_image.R).
# Ce choix garantit que la MEME conversion s'applique a l'entrainement, a
# l'evaluation, dans l'API et dans la WebApp. On se contente de tracer ici les
# images concernees.
m[, needs_colorspace_fix := ok == TRUE & !colorspace %in% c("sRGB", "RGB")]
log_info(sprintf("  %-46s -> %5d image(s) converties a la volee",
                 "conversion sRGB 3 canaux", sum(m$needs_colorspace_fix, na.rm = TRUE)))

# ------------------------------------------------------------------------------
# REGLE 6 (repond a P6) -- classes sous-representees
# ------------------------------------------------------------------------------
# Une classe qui ne compte pas assez d'exemples survivants ne peut ni etre
# apprise, ni etre evaluee de facon fiable. Sur Stanford Dogs cette regle ne
# devrait retirer aucune classe (min ~148 images) ; elle protege le pipeline
# si on lui donne un autre jeu de donnees.
per_class <- m[keep == TRUE, .N, by = dirname]
small_classes <- per_class[N < MIN_PER_CLASS, dirname]
if (length(small_classes)) {
  apply_rule(m, m$dirname %in% small_classes,
             sprintf("classe sous le seuil de %d images", MIN_PER_CLASS))
  log_warn("Classes retirees : ", paste(small_classes, collapse = ", "))
} else {
  log_info(sprintf("  %-46s -> %5d ecartee(s)",
                   sprintf("classe sous le seuil de %d images", MIN_PER_CLASS), 0))
}

clean <- m[keep == TRUE]
log_info("Lignes conservees : ", nrow(clean), " / ", n_start,
         sprintf(" (%.2f %%)", 100 * nrow(clean) / n_start))

# ------------------------------------------------------------------------------
# SPLIT STRATIFIE (repond a P11) -- train 70 / validation 15 / test 15
# ------------------------------------------------------------------------------
# Trois proprietes exigees du split :
#   1. STRATIFIE : la repartition 70/15/15 est appliquee classe par classe, donc
#      chaque race est representee dans les memes proportions partout.
#   2. UN JEU DE TEST existe, et n'est ouvert qu'une fois, en 06_evaluate_model.R.
#      La validation sert a piloter l'entrainement, le test a rapporter le
#      resultat. Confondre les deux roles rend la metrique finale optimiste.
#   3. REPRODUCTIBLE : l'assignation est ecrite dans le manifeste. Rejouer le
#      pipeline ne redistribue pas les images.
set.seed(SEED)

assign_split <- function(n) {
  # Effectifs cibles, arrondis de facon a ce que la somme fasse exactement n.
  n_train <- floor(n * SPLIT_TRAIN)
  n_val   <- floor(n * SPLIT_VAL)
  n_test  <- n - n_train - n_val
  sample(rep(c("train", "val", "test"), times = c(n_train, n_val, n_test)))
}

clean[, split := assign_split(.N), by = dirname]

split_tab <- clean[, .N, by = split][order(-N)]
print(split_tab)

# Verification explicite : chaque classe doit exister dans les trois sous-ensembles.
coverage <- dcast(clean[, .N, by = .(dirname, split)], dirname ~ split,
                  value.var = "N", fill = 0L)
missing_any <- coverage[train == 0 | val == 0 | test == 0]
if (nrow(missing_any)) {
  log_warn(nrow(missing_any), " classe(s) absente(s) d'au moins un sous-ensemble.")
  print(missing_any)
} else {
  log_info("Les ", uniqueN(clean$dirname), " classes sont presentes dans train, val et test.")
}

# Controle anti-fuite : aucun MD5 ne doit apparaitre dans deux sous-ensembles.
leak <- clean[, .(n_splits = uniqueN(split)), by = md5][n_splits > 1L]
if (nrow(leak)) {
  log_warn(nrow(leak), " empreinte(s) MD5 presentes dans plusieurs splits !")
} else {
  log_info("Controle anti-fuite : aucune empreinte MD5 partagee entre splits.")
}

# ------------------------------------------------------------------------------
# INDEX DES CLASSES -- source unique de verite
# ------------------------------------------------------------------------------
# L'ordre alphabetique des noms de dossiers est celui que suit le chargeur de
# donnees. En le figeant ici, dans un fichier produit par le pipeline lui-meme,
# on supprime tout risque de liste de classes maintenue a la main et
# desynchronisee de l'ordre reel des sorties du modele : un decalage d'un seul
# indice rendrait toutes les predictions fausses, sans lever la moindre
# erreur.
class_dirs  <- sort(unique(clean$dirname))
class_index <- data.table(
  index       = seq_along(class_dirs) - 1L,          # base 0, comme Keras
  dirname     = class_dirs,
  synset      = synset_from_dirname(class_dirs),
  breed_raw   = breed_from_dirname(class_dirs),
  breed_label = pretty_breed(breed_from_dirname(class_dirs))
)

clean[, label := match(dirname, class_dirs) - 1L]

fwrite(class_index, file.path(DIR_DATA, "class_index.csv"))
log_info("Index des classes ecrit : ", file.path(DIR_DATA, "class_index.csv"),
         " (", nrow(class_index), " classes)")

# ------------------------------------------------------------------------------
# POIDS DE CLASSE (repond a P6)
# ------------------------------------------------------------------------------
# Poids inversement proportionnels a l'effectif, normalises pour que leur
# moyenne vaille 1 (sinon on modifie implicitement le taux d'apprentissage).
train_counts <- clean[split == "train", .N, by = label][order(label)]
train_counts[, weight := (nrow(clean[split == "train"]) / (nrow(train_counts) * N))]
train_counts[, weight := weight / mean(weight)]
fwrite(train_counts, file.path(DIR_DATA, "class_weights.csv"))
log_info(sprintf("Poids de classe : min %.3f / max %.3f",
                 min(train_counts$weight), max(train_counts$weight)))

# ------------------------------------------------------------------------------
# SAUVEGARDE
# ------------------------------------------------------------------------------
keep_cols <- c("path", "filename", "dirname", "synset", "breed_raw", "breed_label",
               "label", "split", "width", "height", "aspect_ratio", "min_side",
               "colorspace", "format", "size_byte", "md5", "needs_colorspace_fix")
fwrite(clean[, ..keep_cols], PATH_MANIFEST)
log_info("Manifeste nettoye ecrit : ", PATH_MANIFEST)

# Le detail des exclusions est conserve : un nettoyage qu'on ne peut pas
# inspecter apres coup n'est pas un nettoyage, c'est une perte de donnees.
excluded <- m[keep == FALSE, .(path, dirname, exclusion_reason, width, height, error)]
fwrite(excluded, file.path(DIR_REPORTS, "excluded_files.csv"))

# ------------------------------------------------------------------------------
# RAPPORT DE NETTOYAGE
# ------------------------------------------------------------------------------
bilan <- m[keep == FALSE, .N, by = exclusion_reason][order(-N)]

report <- c(
  "# Rapport de nettoyage",
  "",
  sprintf("_Genere le %s par `R/03_clean_data.R`._", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "## Bilan",
  "",
  "| Indicateur | Valeur |",
  "|---|---|",
  sprintf("| Images en entree | %d |", n_start),
  sprintf("| Images conservees | %d (%.2f %%) |", nrow(clean), 100 * nrow(clean) / n_start),
  sprintf("| Images ecartees | %d |", n_start - nrow(clean)),
  sprintf("| Classes conservees | %d |", nrow(class_index)),
  "",
  "## Motifs d'exclusion",
  "",
  "| Motif | Images |",
  "|---|---|",
  if (nrow(bilan)) sprintf("| %s | %d |", bilan$exclusion_reason, bilan$N) else "| aucun | 0 |",
  "",
  "## Repartition apres split stratifie",
  "",
  "| Sous-ensemble | Images | Part |",
  "|---|---|---|",
  sprintf("| %s | %d | %.1f %% |", split_tab$split, split_tab$N,
          100 * split_tab$N / nrow(clean)),
  "",
  "## Normalisations appliquees a la volee",
  "",
  sprintf("- Conversion sRGB 3 canaux : %d image(s) concernees.",
          sum(m$needs_colorspace_fix, na.rm = TRUE)),
  sprintf("- Redimensionnement force en %dx%d, ratio d'aspect non preserve.",
          IMG_WIDTH, IMG_HEIGHT),
  "- Mise a l'echelle des pixels : realisee DANS le modele (`layer_rescaling`),",
  "  pas dans le code appelant. Voir README, section \"Modelisation\".",
  "",
  "## Fichiers produits",
  "",
  "- `data/manifest_clean.csv` -- une ligne par image conservee, avec son split",
  "- `data/class_index.csv` -- correspondance indice <-> race (source de verite)",
  "- `data/class_weights.csv` -- poids de classe pour l'entrainement",
  "- `reports/excluded_files.csv` -- detail de chaque exclusion")

writeLines(report, file.path(DIR_REPORTS, "cleaning_report.md"), useBytes = TRUE)
log_info("Rapport de nettoyage ecrit : ", file.path(DIR_REPORTS, "cleaning_report.md"))

# --- Figure de controle -------------------------------------------------------
p <- ggplot(clean[, .N, by = .(breed_label, split)],
            aes(x = stats::reorder(breed_label, N), y = N, fill = split)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c(train = "#2c7fb8", val = "#7fcdbb", test = "#edf8b1")) +
  labs(title = "Repartition train / validation / test apres nettoyage",
       subtitle = "Split stratifie par race : les proportions sont identiques partout",
       x = NULL, y = "Images", fill = NULL) +
  theme_minimal(base_size = 7)
ggsave(file.path(DIR_REPORTS, "cleaning_split.png"), p, width = 7, height = 12, dpi = 150)

log_info("Etape 03 terminee. Enchainez avec R/05_train_model.R")
