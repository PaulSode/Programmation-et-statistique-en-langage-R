# ==============================================================================
# 02_audit_data.R -- AUDIT DU JEU DE DONNEES
#
#   >>> Ce script repond a la question : "Quels problemes le jeu de donnees
#       pose-t-il ?" Il ne modifie rien. Il mesure, puis ecrit un rapport
#       reproductible dans reports/audit_report.md.
#
# Rien ici ne suppose que le jeu est propre : chaque fichier est ouvert, mesure
# et compare aux attentes. Sans cette etape, les images corrompues sont ecartees
# silencieusement par TensorFlow, le desequilibre des classes reste invisible,
# et les doublons se repartissent entre entrainement et validation sans que rien
# ne le signale.
# ==============================================================================

source("R/common/bootstrap.R")

suppressPackageStartupMessages({
  library(data.table)
  library(digest)
  library(ggplot2)
})

log_step("02 -- Audit du jeu de donnees")

stopifnot(dir.exists(DIR_IMAGES))

# ------------------------------------------------------------------------------
# 1. Construction du manifeste brut
# ------------------------------------------------------------------------------
log_info("Recensement des fichiers...")

files <- list.files(DIR_IMAGES, recursive = TRUE, full.names = TRUE, all.files = TRUE)
files <- files[!dir.exists(files)]

manifest <- data.table(
  path      = normalizePath(files, winslash = "/"),
  dirname   = basename(dirname(files)),
  filename  = basename(files),
  size_byte = file.info(files)$size
)

manifest[, `:=`(
  synset    = synset_from_dirname(dirname),
  breed_raw = breed_from_dirname(dirname),
  ext       = tolower(tools::file_ext(filename))
)]

log_info("Fichiers recenses : ", nrow(manifest))

# ------------------------------------------------------------------------------
# 2. Inspection image par image (format reel, dimensions, espace colorimetrique)
# ------------------------------------------------------------------------------
log_info("Inspection des images (compter quelques minutes)...")

pb   <- utils::txtProgressBar(min = 0, max = nrow(manifest), style = 3)
insp <- vector("list", nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  insp[[i]] <- inspect_image(manifest$path[i])
  if (i %% 250L == 0L || i == nrow(manifest)) utils::setTxtProgressBar(pb, i)
}
close(pb)

manifest <- cbind(manifest, rbindlist(insp, fill = TRUE))

manifest[, aspect_ratio := ifelse(!is.na(height) & height > 0, width / height, NA_real_)]
manifest[, min_side     := pmin(width, height)]

# ------------------------------------------------------------------------------
# 3. Empreinte MD5 -> detection des doublons exacts
# ------------------------------------------------------------------------------
log_info("Calcul des empreintes MD5...")
manifest[, md5 := vapply(path, function(p)
  tryCatch(digest::digest(file = p, algo = "md5"), error = function(e) NA_character_),
  character(1L))]

manifest[, dup_group := .N,           by = md5]
manifest[, dup_rank  := seq_len(.N),  by = md5]
manifest[is.na(md5), `:=`(dup_group = 1L, dup_rank = 1L)]
# Est marquee doublon toute occurrence APRES la premiere d'un meme MD5.
manifest[, is_duplicate := dup_group > 1L & dup_rank > 1L]

# ------------------------------------------------------------------------------
# 4. Diagnostics
# ------------------------------------------------------------------------------
n_total <- nrow(manifest)
ok_m    <- manifest[ok == TRUE]
# `diags` et non `diag` : ce dernier masquerait base::diag().
diags   <- list()

# --- P1. Nomenclature : le nom de dossier melange synset ImageNet et race ------
diags$P1_nomenclature <- list(
  titre = "Noms de classes illisibles (synset ImageNet colle au nom de race)",
  detail = sprintf(
    paste0("Les %d dossiers sont nommes 'nXXXXXXXX-race' (ex. '%s'). Utilises tels ",
           "quels comme labels, ils produisent des sorties opaques pour un ",
           "utilisateur final. Il faut de plus ",
           "conserver le synset separement : c'est la cle de jointure vers la ",
           "traduction francaise et vers les metadonnees ImageNet."),
    uniqueN(manifest$dirname), manifest$dirname[1]),
  n = uniqueN(manifest$dirname)
)

# --- P2. Fichiers illisibles / corrompus --------------------------------------
bad <- manifest[ok == FALSE]
diags$P2_corrompus <- list(
  titre  = "Fichiers illisibles ou corrompus",
  detail = sprintf(
    paste0("%d fichier(s) sur %d n'ont pas pu etre decodes. Motifs rencontres : %s. ",
           "TensorFlow les ecarte silencieusement a l'entrainement : on croit donc ",
           "entrainer sur N images alors qu'on en utilise moins. Cote service, une ",
           "telle image doit produire une erreur 400 explicite, pas un 500."),
    nrow(bad), n_total,
    if (nrow(bad)) paste(unique(bad$error), collapse = " | ") else "aucun"),
  n = nrow(bad)
)

# --- P3. Espace colorimetrique non RGB ----------------------------------------
non_rgb <- ok_m[!colorspace %in% c("sRGB", "RGB")]
diags$P3_colorspace <- list(
  titre  = "Images hors sRGB (niveaux de gris, CMYK, palette indexee)",
  detail = sprintf(
    paste0("%d image(s) concernees. Espaces rencontres : %s. Un decodeur qui ne force ",
           "pas explicitement 3 canaux renvoie un tableau de forme (H, W, 1) ou ",
           "(H, W, 4), incompatible avec l'entree (224, 224, 3) du modele."),
    nrow(non_rgb),
    if (nrow(non_rgb)) paste(sort(unique(non_rgb$colorspace)), collapse = ", ") else "aucun"),
  n = nrow(non_rgb)
)

# --- P4. Extension mensongere -------------------------------------------------
ext_mismatch <- ok_m[!(toupper(format) %in% c("JPEG", "JPG")) & ext %in% c("jpg", "jpeg")]
diags$P4_extension <- list(
  titre  = "Extension .jpg alors que le contenu reel est d'un autre format",
  detail = sprintf(
    "%d fichier(s). Formats reels rencontres : %s. Le filtrage par extension seule est donc peu fiable sur ce jeu de donnees.",
    nrow(ext_mismatch),
    if (nrow(ext_mismatch)) paste(sort(unique(ext_mismatch$format)), collapse = ", ") else "aucun"),
  n = nrow(ext_mismatch)
)

# --- P5. Doublons exacts ------------------------------------------------------
dups <- manifest[is_duplicate == TRUE]
diags$P5_doublons <- list(
  titre  = "Doublons exacts (meme empreinte MD5)",
  detail = sprintf(
    paste0("%d fichier(s) redondants. Avec un split aleatoire, une image dupliquee peut ",
           "se retrouver simultanement en entrainement et en validation : le modele est ",
           "alors evalue sur une image qu'il a memorisee, et la precision de validation ",
           "est gonflee sans que rien ne le signale."),
    nrow(dups)),
  n = nrow(dups)
)

# --- P6. Desequilibre des classes ---------------------------------------------
per_class <- ok_m[, .N, by = .(dirname, breed_raw)][order(N)]
diags$P6_desequilibre <- list(
  titre  = "Desequilibre entre classes et faible effectif absolu",
  detail = sprintf(
    paste0("Effectifs par race : min = %d (%s), max = %d (%s), mediane = %.0f, ",
           "ratio max/min = %.2f. Le desequilibre reste modere, mais l'effectif ABSOLU ",
           "est faible : ~%.0f images par race. Entrainer un CNN de zero sur si peu ",
           "d'exemples sur-apprendrait immediatement -- c'est ce qui justifie le ",
           "transfer learning."),
    per_class$N[1], pretty_breed(per_class$breed_raw[1]),
    per_class$N[nrow(per_class)], pretty_breed(per_class$breed_raw[nrow(per_class)]),
    median(per_class$N), max(per_class$N) / min(per_class$N), mean(per_class$N)),
  n = nrow(per_class)
)

# --- P7. Heterogeneite des resolutions et des ratios --------------------------
diags$P7_resolutions <- list(
  titre  = "Resolutions et ratios d'aspect tres heterogenes",
  detail = sprintf(
    paste0("Largeur de %d a %d px, hauteur de %d a %d px. Ratio d'aspect de %.2f a %.2f ",
           "(mediane %.2f). Le redimensionnement force en %dx%d ecrase donc fortement ",
           "les images les plus allongees. C'est un choix assume (identique cote ",
           "entrainement et cote service), pas un oubli."),
    min(ok_m$width), max(ok_m$width), min(ok_m$height), max(ok_m$height),
    min(ok_m$aspect_ratio, na.rm = TRUE), max(ok_m$aspect_ratio, na.rm = TRUE),
    median(ok_m$aspect_ratio, na.rm = TRUE), IMG_WIDTH, IMG_HEIGHT),
  n = nrow(ok_m)
)

# --- P8. Images trop petites --------------------------------------------------
tiny <- ok_m[min_side < MIN_SIDE_PX]
diags$P8_minuscules <- list(
  titre  = sprintf("Images de moins de %d px de cote", MIN_SIDE_PX),
  detail = sprintf(
    "%d image(s). Agrandies en %dx%d, elles n'apportent que du flou d'interpolation et bruitent le gradient.",
    nrow(tiny), IMG_WIDTH, IMG_HEIGHT),
  n = nrow(tiny)
)

# --- P9. Classes visuellement confusables -------------------------------------
# La liste vit dans config.R : 06_evaluate_model.R la confronte ensuite aux
# confusions reellement observees sur le jeu de test.
confusables <- CONFUSABLE_GROUPS

diags$P9_confusion <- list(
  titre  = "Groupes de races visuellement quasi identiques",
  detail = sprintf(
    paste0("%d races reparties en %d groupes ambigus (huskies / malamutes / chiens ",
           "esquimaux, caniches distingues par la TAILLE et non par l'apparence, ",
           "schnauzers idem, Pembroke / Cardigan...). Une partie de l'erreur residuelle ",
           "est irreductible depuis une seule photo : elle vient de la definition des ",
           "classes, pas du modele. C'est pourquoi la precision top-3 est une metrique ",
           "plus honnete que la top-1 sur ce probleme, et pourquoi l'API renvoie 3 ",
           "candidats plutot qu'un seul."),
    length(unique(unlist(confusables))), length(confusables)),
  n = length(unique(unlist(confusables)))
)

# --- P10. Fuite de donnees via le pre-entrainement ImageNet --------------------
diags$P10_fuite <- list(
  titre  = "Fuite de donnees : Stanford Dogs est un sous-ensemble d'ImageNet",
  detail = paste0(
    "Les 120 races de Stanford Dogs proviennent d'ImageNet, et MobileNetV2 est ",
    "pre-entraine sur ImageNet-1k, qui contient exactement ces 120 classes de chiens. ",
    "Le reseau de base a donc potentiellement deja vu, pendant son pre-entrainement, ",
    "les images memes de notre jeu de validation. La precision de validation est en ",
    "consequence optimiste et ne mesure pas la generalisation a de nouvelles photos. ",
    "Ce probleme ne se corrige pas par du nettoyage : il se documente, et se compense ",
    "en evaluant en plus sur des photos externes au jeu (voir 06_evaluate_model.R, ",
    "section 'jeu de controle externe')."),
  n = NA_integer_
)

# --- P11. Protocole d'evaluation incomplet ------------------------------------
diags$P11_protocole <- list(
  titre  = "Un protocole d'evaluation naif serait biaise",
  detail = paste0(
    "Un simple split 80/20 train/validation, dont on lirait la precision de validation ",
    "comme resultat final, serait biaise par selection : ce meme jeu sert a decider du ",
    "nombre d'epochs, donc a choisir un modele. Par ailleurs un decoupage qui melange ",
    "tous les chemins puis coupe en un bloc n'est pas stratifie : les proportions par ",
    "classe varient librement. 03_clean_data.R applique donc un split stratifie ",
    "70/15/15, avec un jeu de test jamais regarde avant l'evaluation finale."),
  n = NA_integer_
)

# ------------------------------------------------------------------------------
# 5. Figures
# ------------------------------------------------------------------------------
log_info("Generation des figures...")

p_bal <- ggplot(per_class, aes(x = stats::reorder(pretty_breed(breed_raw), N), y = N)) +
  geom_col(fill = "#2c7fb8") +
  coord_flip() +
  labs(title = "Nombre d'images par race",
       subtitle = sprintf("%d races -- ratio max/min = %.2f",
                          nrow(per_class), max(per_class$N) / min(per_class$N)),
       x = NULL, y = "Images") +
  theme_minimal(base_size = 7)
ggsave(file.path(DIR_REPORTS, "audit_class_balance.png"), p_bal,
       width = 7, height = 12, dpi = 150)

p_dim <- ggplot(ok_m, aes(x = width, y = height)) +
  geom_point(alpha = 0.08, size = 0.5, colour = "#2c7fb8") +
  geom_hline(yintercept = IMG_HEIGHT, linetype = 2, colour = "red") +
  geom_vline(xintercept = IMG_WIDTH,  linetype = 2, colour = "red") +
  scale_x_log10() + scale_y_log10() +
  labs(title = "Dispersion des resolutions d'origine",
       subtitle = sprintf("Les pointilles rouges marquent la cible %dx%d", IMG_WIDTH, IMG_HEIGHT),
       x = "Largeur (px, echelle log)", y = "Hauteur (px, echelle log)") +
  theme_minimal(base_size = 10)
ggsave(file.path(DIR_REPORTS, "audit_resolutions.png"), p_dim,
       width = 7, height = 6, dpi = 150)

p_ar <- ggplot(ok_m, aes(x = aspect_ratio)) +
  geom_histogram(bins = 80, fill = "#2c7fb8") +
  geom_vline(xintercept = 1, linetype = 2, colour = "red") +
  labs(title = "Distribution des ratios d'aspect",
       subtitle = "La cible 224x224 impose un ratio de 1 : tout ecart est une deformation",
       x = "Largeur / Hauteur", y = "Images") +
  theme_minimal(base_size = 10)
ggsave(file.path(DIR_REPORTS, "audit_aspect_ratio.png"), p_ar,
       width = 7, height = 4, dpi = 150)

# ------------------------------------------------------------------------------
# 6. Rapport Markdown
# ------------------------------------------------------------------------------
lines <- c(
  "# Rapport d'audit -- Stanford Dogs",
  "",
  sprintf("_Genere le %s par `R/02_audit_data.R`._", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "## Vue d'ensemble",
  "",
  sprintf("- Fichiers recenses : **%d**", n_total),
  sprintf("- Fichiers decodables : **%d** (%.2f %%)", sum(manifest$ok), 100 * mean(manifest$ok)),
  sprintf("- Races (dossiers) : **%d**", uniqueN(manifest$dirname)),
  sprintf("- Images par race : min %d / mediane %.0f / max %d",
          min(per_class$N), median(per_class$N), max(per_class$N)),
  sprintf("- Poids total : **%.1f Go**", sum(manifest$size_byte, na.rm = TRUE) / 1024^3),
  "",
  "## Problemes identifies",
  ""
)

for (nm in names(diags)) {
  d <- diags[[nm]]
  lines <- c(lines,
    sprintf("### %s -- %s", sub("_.*$", "", nm), d$titre),
    "",
    if (is.na(d$n)) "**Probleme structurel (non chiffrable par comptage)**"
    else sprintf("**Occurrences : %s**", format(d$n, big.mark = " ")),
    "",
    d$detail,
    "")
}

lines <- c(lines,
  "## Figures",
  "",
  "- `audit_class_balance.png` -- effectif par race",
  "- `audit_resolutions.png` -- dispersion des resolutions d'origine",
  "- `audit_aspect_ratio.png` -- distribution des ratios d'aspect",
  "",
  "## Suite",
  "",
  "Les corrections apportees a chacun de ces points sont implementees dans",
  "`R/03_clean_data.R` et recapitulees dans le README, section \"Nettoyage\".")

writeLines(lines, file.path(DIR_REPORTS, "audit_report.md"), useBytes = TRUE)

# ------------------------------------------------------------------------------
# 7. Sauvegarde du manifeste brut
# ------------------------------------------------------------------------------
fwrite(manifest, PATH_MANIFEST_RAW)
log_info("Manifeste brut ecrit  : ", PATH_MANIFEST_RAW)
log_info("Rapport d'audit ecrit : ", file.path(DIR_REPORTS, "audit_report.md"))

# --- Resume console -----------------------------------------------------------
cat("\n---------------- SYNTHESE DE L'AUDIT ----------------\n")
for (nm in names(diags)) {
  d <- diags[[nm]]
  cat(sprintf("%-5s %8s  %s\n", sub("_.*$", "", nm),
              if (is.na(d$n)) "-" else format(d$n, big.mark = " "), d$titre))
}
cat("-----------------------------------------------------\n")
log_info("Etape 02 terminee. Enchainez avec R/03_clean_data.R")
