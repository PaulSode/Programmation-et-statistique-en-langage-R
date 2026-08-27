# ==============================================================================
# 98_smoke_test.R -- Verification rapide de l'installation
#
# A executer AVANT de lancer le pipeline complet : ce script valide en quelques
# secondes que l'environnement R, le backend Python, le decodage d'images et
# (si le modele existe) le service d'inference fonctionnent.
#
#     Rscript R/98_smoke_test.R
#     Rscript R/98_smoke_test.R --api http://127.0.0.1:5000
#
# Chaque test affiche PASS, FAIL ou SKIP. Le script sort en code 1 si un test
# echoue, ce qui le rend utilisable en integration continue.
# ==============================================================================

if (!file.exists("R/common/config.R")) {
  stop("Executez ce script depuis la racine du projet.", call. = FALSE)
}

results <- list()

check <- function(label, expr, skip_if = FALSE, skip_reason = "") {
  # skip_if peut dependre d'une variable qu'un test precedent n'a pas pu
  # definir : on traite une erreur d'evaluation comme un motif de saut.
  skip <- tryCatch(isTRUE(skip_if), error = function(e) TRUE)
  if (skip) {
    cat(sprintf("SKIP  %-52s %s\n", label, skip_reason))
    results[[label]] <<- NA
    return(invisible(NA))
  }
  out <- tryCatch(expr, error = function(e) e)
  if (inherits(out, "error")) {
    cat(sprintf("FAIL  %-52s %s\n", label, conditionMessage(out)))
    results[[label]] <<- FALSE
  } else if (isTRUE(out) || is.character(out)) {
    cat(sprintf("PASS  %-52s %s\n", label, if (is.character(out)) out else ""))
    results[[label]] <<- TRUE
  } else {
    cat(sprintf("FAIL  %-52s resultat inattendu\n", label))
    results[[label]] <<- FALSE
  }
  invisible(results[[label]])
}

cat("\n=== Verification de l'environnement ===\n\n")

# --- 1. Paquets R -------------------------------------------------------------
required <- c("keras3", "tensorflow", "tfdatasets", "reticulate", "magick",
              "digest", "jsonlite", "data.table", "ggplot2", "scales",
              "plumber", "shiny", "bslib", "httr2", "curl", "base64enc")
for (pkg in required) {
  check(sprintf("paquet '%s'", pkg), {
    if (!requireNamespace(pkg, quietly = TRUE))
      stop("absent -- lancez R/00_install_dependencies.R")
    as.character(utils::packageVersion(pkg))
  })
}

# --- 2. Configuration ---------------------------------------------------------
check("chargement de config.R", {
  source("R/common/config.R")
  source("R/common/utils_image.R")
  sprintf("racine : %s", PROJECT_ROOT)
})

# --- 3. Backend Python --------------------------------------------------------
check("backend TensorFlow", {
  tf <- reticulate::import("tensorflow", delay_load = FALSE)
  sprintf("TensorFlow %s, %d GPU",
          tf$`__version__`, length(tf$config$list_physical_devices("GPU")))
})

# --- 4. Decodage et pretraitement d'image -------------------------------------
# Une image synthetique suffit : on teste la chaine, pas le contenu.
tmp_png <- file.path(tempdir(), "smoke_test.png")
check("creation d'une image de test", {
  img <- magick::image_blank(320, 180, color = "steelblue")
  magick::image_write(img, tmp_png, format = "png")
  sprintf("%s (320x180)", basename(tmp_png))
})

check("inspect_image()", {
  info <- inspect_image(tmp_png)
  if (!isTRUE(info$ok)) stop(info$error)
  if (info$width != 320L || info$height != 180L)
    stop(sprintf("dimensions lues %dx%d", info$width, info$height))
  sprintf("%dx%d %s", info$width, info$height, info$colorspace)
})

check("image_to_tensor() -- forme et plage", {
  x <- image_to_tensor(tmp_png)
  expected <- c(1L, IMG_HEIGHT, IMG_WIDTH, 3L)
  if (!identical(dim(x), expected))
    stop(sprintf("forme %s au lieu de %s",
                 paste(dim(x), collapse = "x"), paste(expected, collapse = "x")))
  # Les valeurs doivent rester en 0-255 : la mise a l'echelle appartient au
  # modele, pas au code appelant. Un tenseur dans [0, 1] ici signalerait que la
  # normalisation a fuite hors du modele.
  if (max(x) <= 1.0) stop("valeurs dans [0,1] : la normalisation a fuite hors du modele")
  if (min(x) < 0 || max(x) > 255) stop("valeurs hors de [0, 255]")
  sprintf("%s, min %.0f max %.0f", paste(dim(x), collapse = "x"), min(x), max(x))
})

check("image_to_tensor() depuis des octets bruts", {
  raw_bytes <- readBin(tmp_png, "raw", file.info(tmp_png)$size)
  x <- image_to_tensor(raw_bytes)
  if (!identical(dim(x), c(1L, IMG_HEIGHT, IMG_WIDTH, 3L))) stop("forme incorrecte")
  "identique au chemin de fichier"
})

# --- 4 bis. Disposition des axes et des canaux, au pixel pres -----------------
# Test de non-regression sur un piege reel : magick definit une methode
# as.integer.bitmap() qui transpose silencieusement le tableau en (H, W, C).
# Une implementation qui l'ignore melange axes et canaux sans lever d'erreur --
# le modele recoit du bruit et repond quand meme. Seule une verification au
# pixel pres le detecte, d'ou ce test.
check("image_to_tensor() -- axes et canaux au pixel pres", {
  W <- 4L; H <- 2L
  cols <- c("red", "green", "blue", "white", "black", "yellow", "cyan", "magenta")
  expected <- matrix(c(255,0,0,  0,128,0,  0,0,255,  255,255,255,
                       0,0,0,    255,255,0, 0,255,255, 255,0,255),
                     nrow = 8L, byrow = TRUE)

  im <- magick::image_blank(W, H, "white")
  for (y in 0:(H - 1L)) for (x in 0:(W - 1L)) {
    im <- magick::image_composite(im, magick::image_blank(1, 1, cols[y * W + x + 1L]),
                                  offset = sprintf("+%d+%d", x, y))
  }
  f <- file.path(tempdir(), "layout_probe.png")
  magick::image_write(im, f, format = "png")
  on.exit(unlink(f), add = TRUE)

  # Redimensionnement desactive : on veut comparer pixel a pixel.
  arr <- image_to_tensor(f, target_size = c(H, W))[1, , , ]

  bad <- 0L
  for (y in seq_len(H)) for (x in seq_len(W)) {
    if (!all(arr[y, x, ] == expected[(y - 1L) * W + x, ])) bad <- bad + 1L
  }
  if (bad > 0L) {
    stop(sprintf("%d/%d pixels incorrects : les axes ou les canaux sont permutes",
                 bad, H * W))
  }
  sprintf("%d/%d pixels exacts (ordre RGB et disposition (H,W,C) verifies)",
          H * W, H * W)
})

# --- 5. Donnees ---------------------------------------------------------------
check("table de traduction francaise", {
  if (!file.exists(PATH_BREEDS_FR)) stop("data/breeds_fr.csv absent")
  b <- data.table::fread(PATH_BREEDS_FR, encoding = "UTF-8")
  if (nrow(b) != 120L) stop(sprintf("%d lignes au lieu de 120", nrow(b)))
  sprintf("%d races traduites", nrow(b))
})

check("manifeste nettoye", {
  if (!file.exists(PATH_MANIFEST)) stop("absent")
  m <- data.table::fread(PATH_MANIFEST)
  sprintf("%d images, %d classes, splits : %s", nrow(m), data.table::uniqueN(m$label),
          paste(sort(unique(m$split)), collapse = "/"))
}, skip_if = !file.exists(PATH_MANIFEST),
   skip_reason = "pipeline non execute (R/03_clean_data.R)")

# --- 6. Service d'inference ---------------------------------------------------
model_exists <- file.exists(PATH_MODEL)

check("chargement du service", {
  source("R/common/predict_service.R")
  if (!load_service()) stop("load_service() a echoue")
  st <- service_status()
  sprintf("%d classes", st$n_classes)
}, skip_if = !model_exists, skip_reason = "models/model.keras absent")

check("prediction de bout en bout", {
  res <- classify_dog_breed(tmp_png, top_k = 3L)
  if (length(res$predictions) != 3L) stop("top_k non respecte")
  conf <- vapply(res$predictions, function(p) p$confidence, numeric(1))
  if (is.unsorted(rev(conf))) stop("predictions non triees par confiance")
  if (any(conf < 0 | conf > 1)) stop("confiances hors de [0, 1]")
  sprintf("top-1 : %s (%.1f %%), %.0f ms",
          res$top_prediction$class_label, 100 * conf[1], res$inference_ms)
}, skip_if = !model_exists, skip_reason = "models/model.keras absent")

# --- 7. API (si une adresse est fournie) --------------------------------------
args <- commandArgs(trailingOnly = TRUE)
api_url <- {
  i <- which(args == "--api")
  if (length(i) && length(args) > i[1]) args[i[1] + 1L] else NULL
}

check("API : GET /api/health", {
  r <- httr2::request(api_url) |>
    httr2::req_url_path("/api/health") |>
    httr2::req_timeout(10) |>
    httr2::req_error(is_error = function(x) FALSE) |>
    httr2::req_perform()
  b <- httr2::resp_body_json(r)
  sprintf("HTTP %d, ready = %s, %s classes",
          httr2::resp_status(r), b$ready, b$n_classes)
}, skip_if = is.null(api_url), skip_reason = "passez --api <url> pour tester")

check("API : POST /api/predict", {
  r <- httr2::request(api_url) |>
    httr2::req_url_path("/api/predict") |>
    httr2::req_body_multipart(image = curl::form_file(tmp_png)) |>
    httr2::req_timeout(60) |>
    httr2::req_error(is_error = function(x) FALSE) |>
    httr2::req_perform()
  b <- httr2::resp_body_json(r)
  if (httr2::resp_status(r) != 200L)
    stop(sprintf("HTTP %d : %s", httr2::resp_status(r), b$error))
  sprintf("HTTP 200, top-1 : %s (%.1f %%)",
          b$top_prediction$class_label, 100 * b$top_prediction$confidence)
}, skip_if = is.null(api_url), skip_reason = "passez --api <url> pour tester")

# --- Bilan --------------------------------------------------------------------
unlink(tmp_png)

vals <- unlist(results)
n_pass <- sum(vals %in% TRUE)
n_fail <- sum(vals %in% FALSE)
n_skip <- sum(is.na(vals))

cat(sprintf("\n%d PASS / %d FAIL / %d SKIP\n\n", n_pass, n_fail, n_skip))

if (n_fail > 0L) {
  cat("Des tests ont echoue. Voir R/00_install_dependencies.R pour l'installation.\n")
  quit(status = 1L)
}
cat("Environnement operationnel.\n")
