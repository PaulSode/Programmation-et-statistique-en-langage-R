# ==============================================================================
# app/app.R -- WebApp Shiny de classification de races de chiens
#
# Interface web du projet : depot d'image, prediction, et consultation des
# rapports produits par le pipeline. Tout tient dans un seul fichier R :
# interface, logique et rendu.
#
# Deux modes de prediction, choisis dans la barre laterale :
#
#   "API"    la WebApp envoie l'image a l'API plumber en HTTP, exactement comme
#            le ferait n'importe quel client. C'est le mode qui valide le
#            deploiement de bout en bout.
#   "local"  la WebApp charge le modele dans son propre processus. Utile pour
#            demontrer l'application sans avoir a lancer deux services.
#
# Lancement :
#     Rscript app/run_app.R
#   ou, en session R depuis la racine du projet :
#     shiny::runApp("app", port = 3838)
# ==============================================================================

# --- Amorcage -----------------------------------------------------------------
.root <- Sys.getenv("DOGCLF_ROOT", unset = "")
if (!nzchar(.root)) {
  .root <- if (file.exists("R/common/config.R")) normalizePath(".", winslash = "/")
           else normalizePath("..", winslash = "/")
}
source(file.path(.root, "R", "common", "config.R"))
source(file.path(.root, "R", "common", "utils_image.R"))
source(file.path(.root, "R", "common", "predict_service.R"))

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(ggplot2)
  library(data.table)
  library(jsonlite)
  library(base64enc)
  library(httr2)
  library(curl)     # curl::form_file(), pour l'envoi multipart vers l'API
  library(scales)
})

# ------------------------------------------------------------------------------
# Client API
# ------------------------------------------------------------------------------

#' Appelle POST /api/predict sur l'API plumber.
predict_via_api <- function(path, base_url, top_k = TOP_K) {
  resp <- httr2::request(base_url) |>
    httr2::req_url_path("/api/predict") |>
    httr2::req_url_query(type = "dog_breed", top_k = top_k) |>
    httr2::req_body_multipart(image = curl::form_file(path)) |>
    httr2::req_timeout(60) |>
    httr2::req_error(is_error = function(r) FALSE) |>
    httr2::req_perform()

  body <- httr2::resp_body_json(resp, simplifyVector = FALSE)

  if (httr2::resp_status(resp) >= 400L) {
    stop(sprintf("API %d : %s%s", httr2::resp_status(resp),
                 body$error %||% "erreur inconnue",
                 if (!is.null(body$detail)) paste0(" -- ", body$detail) else ""),
         call. = FALSE)
  }
  body
}

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0L) b else a

#' Verifie que l'API repond.
api_health <- function(base_url) {
  tryCatch({
    resp <- httr2::request(base_url) |>
      httr2::req_url_path("/api/health") |>
      httr2::req_timeout(5) |>
      httr2::req_error(is_error = function(r) FALSE) |>
      httr2::req_perform()
    httr2::resp_body_json(resp, simplifyVector = FALSE)
  }, error = function(e) NULL)
}

#' Normalise la reponse (API ou locale) en un data.table pour l'affichage.
as_prediction_table <- function(result) {
  rbindlist(lapply(result$predictions, function(p) data.table(
    race_fr    = p$class_name_fr %||% p$class_label %||% NA_character_,
    race       = p$class_label   %||% p$class_name  %||% NA_character_,
    synset     = p$class_id      %||% NA_character_,
    confiance  = as.numeric(p$confidence %||% NA_real_)
  )))
}

# ------------------------------------------------------------------------------
# Chargement des rapports (onglet Documentation)
# ------------------------------------------------------------------------------
read_report <- function(name, titre_defaut) {
  f <- file.path(DIR_REPORTS, name)
  if (!file.exists(f)) {
    return(sprintf(paste0("### %s\n\n_Rapport non genere._\n\nExecutez le script ",
                          "correspondant du pipeline pour le produire."), titre_defaut))
  }
  paste(readLines(f, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
}

# ------------------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------------------
# bslib::font_google() telecharge la police au moment ou l'interface est
# construite. Sans acces reseau -- conteneur isole, poste hors ligne --, l'appel
# echoue et emporte toute l'application avec lui. On retombe donc sur une pile
# de polices systeme, qui ne change que l'apparence.
.app_font <- tryCatch(
  bslib::font_google("Inter"),
  error = function(e) {
    message("Police Google indisponible (", conditionMessage(e),
            ") : utilisation des polices systeme.")
    bslib::font_collection("system-ui", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif")
  }
)

ui <- bslib::page_navbar(
  title = "Classification de races de chiens",
  theme = bslib::bs_theme(version = 5, preset = "flatly", base_font = .app_font),
  fillable = FALSE,

  # ---------------------------------------------------------------- Prediction
  bslib::nav_panel(
    "Prediction",
    bslib::layout_sidebar(
      sidebar = bslib::sidebar(
        width = 340,

        fileInput("image", "Image a analyser",
                  accept = c("image/jpeg", "image/png", "image/webp",
                             ".jpg", ".jpeg", ".png", ".webp"),
                  buttonLabel = "Parcourir", placeholder = "Aucun fichier"),

        sliderInput("top_k", "Nombre de candidats", min = 1, max = 10,
                    value = TOP_K, step = 1),

        radioButtons("backend", "Moteur d'inference",
                     choices = c("API REST (plumber)" = "api",
                                 "Modele local (en processus)" = "local"),
                     selected = "api"),

        conditionalPanel(
          "input.backend == 'api'",
          textInput("api_url", "Adresse de l'API", value = API_BASE_URL),
          actionButton("check_api", "Tester la connexion", class = "btn-sm btn-outline-secondary"),
          uiOutput("api_status")
        ),

        hr(),
        actionButton("go", "Analyser", class = "btn-primary w-100", icon = icon("play")),
        div(class = "form-text mt-2",
            sprintf("Taille maximale : %d Mo. Formats : JPEG, PNG, WebP.", MAX_UPLOAD_MB))
      ),

      bslib::layout_columns(
        col_widths = c(5, 7),
        bslib::card(
          bslib::card_header("Image"),
          uiOutput("preview"),
          bslib::card_footer(textOutput("file_meta"))
        ),
        bslib::card(
          bslib::card_header("Resultat"),
          uiOutput("verdict"),
          plotOutput("bars", height = "260px"),
          tableOutput("table"),
          bslib::card_footer(textOutput("timing"))
        )
      )
    )
  ),

  # ------------------------------------------------------------------- Modele
  bslib::nav_panel(
    "Modele",
    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(
        bslib::card_header("Metadonnees du modele deploye"),
        verbatimTextOutput("metadata")
      ),
      bslib::card(
        bslib::card_header("Chaine de traitement"),
        htmlOutput("pipeline_desc")
      )
    ),
    bslib::card(
      bslib::card_header("Races reconnues"),
      tableOutput("breeds")
    )
  ),

  # ------------------------------------------------------------ Documentation
  bslib::nav_panel(
    "Donnees et methode",
    bslib::navset_tab(
      bslib::nav_panel("Problemes rencontres",
        bslib::card(bslib::card_body(uiOutput("doc_audit")))),
      bslib::nav_panel("Nettoyage",
        bslib::card(bslib::card_body(uiOutput("doc_cleaning")))),
      bslib::nav_panel("Evaluation",
        bslib::card(bslib::card_body(uiOutput("doc_eval"))))
    )
  ),

  bslib::nav_spacer(),
  bslib::nav_item(tags$span(class = "navbar-text small",
                            "MobileNetV2 - transfer learning - 120 races"))
)

# ------------------------------------------------------------------------------
# Serveur
# ------------------------------------------------------------------------------
server <- function(input, output, session) {

  result   <- reactiveVal(NULL)
  err_msg  <- reactiveVal(NULL)
  elapsed  <- reactiveVal(NULL)

  # --- Apercu de l'image ------------------------------------------------------
  output$preview <- renderUI({
    f <- input$image
    if (is.null(f)) {
      return(div(class = "text-muted text-center p-5",
                 icon("image", class = "fa-3x mb-3"), br(),
                 "Selectionnez une photo de chien."))
    }
    # ignore.case plutot que le drapeau inline (?i) : le moteur de regex par
    # defaut de R (TRE) ne le reconnait pas, il faudrait perl = TRUE.
    mime <- if (grepl("[.]png$",  f$name, ignore.case = TRUE)) "image/png"
            else if (grepl("[.]webp$", f$name, ignore.case = TRUE)) "image/webp"
            else "image/jpeg"
    tags$img(
      src = paste0("data:", mime, ";base64,", base64enc::base64encode(f$datapath)),
      style = "max-width:100%; border-radius:6px;"
    )
  })

  output$file_meta <- renderText({
    f <- input$image
    if (is.null(f)) return("")
    info <- inspect_image(f$datapath)
    if (!isTRUE(info$ok)) return(sprintf("%s - illisible (%s)", f$name, info$error))
    sprintf("%s - %d x %d px - %s - %.0f Ko",
            f$name, info$width, info$height, info$colorspace, f$size / 1024)
  })

  # --- Test de connexion a l'API ----------------------------------------------
  api_state <- reactiveVal(NULL)

  observeEvent(input$check_api, {
    h <- api_health(input$api_url)
    api_state(h)
  })

  output$api_status <- renderUI({
    h <- api_state()
    if (is.null(h)) return(NULL)
    if (isTRUE(h$ready)) {
      div(class = "alert alert-success py-2 px-3 mt-2 mb-0 small",
          sprintf("API joignable - %s classes chargees.", h$n_classes))
    } else {
      div(class = "alert alert-warning py-2 px-3 mt-2 mb-0 small",
          "API joignable mais modele non charge (mode degrade).")
    }
  })

  # --- Analyse ----------------------------------------------------------------
  observeEvent(input$go, {
    f <- input$image
    err_msg(NULL); result(NULL); elapsed(NULL)

    if (is.null(f)) {
      err_msg("Aucune image selectionnee.")
      return()
    }
    if (f$size / 1024^2 > MAX_UPLOAD_MB) {
      err_msg(sprintf("Image trop volumineuse (%.1f Mo, limite %d Mo).",
                      f$size / 1024^2, MAX_UPLOAD_MB))
      return()
    }

    t0 <- Sys.time()
    withProgress(message = "Analyse en cours", value = 0.4, {
      out <- tryCatch({
        if (identical(input$backend, "api")) {
          predict_via_api(f$datapath, input$api_url, top_k = input$top_k)
        } else {
          if (!service_ready() && !load_service()) {
            stop("Modele local indisponible : ", PATH_MODEL, " est absent. ",
                 "Entrainez-le avec R/05_train_model.R.", call. = FALSE)
          }
          classify_dog_breed(f$datapath, top_k = input$top_k)
        }
      }, error = function(e) e)

      if (inherits(out, "error")) err_msg(conditionMessage(out)) else result(out)
    })
    elapsed(as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000)
  })

  # --- Verdict ----------------------------------------------------------------
  output$verdict <- renderUI({
    if (!is.null(err_msg())) {
      return(div(class = "alert alert-danger", strong("Echec : "), err_msg()))
    }
    r <- result()
    if (is.null(r)) {
      return(div(class = "text-muted p-4", "Aucun resultat pour l'instant."))
    }

    dt   <- as_prediction_table(r)
    best <- dt[1]

    # Un avertissement explicite quand le modele hesite. Le score de confiance
    # est calibre et verifie en 06_evaluate_model.R : il est donc legitime de
    # s'en servir pour nuancer la reponse plutot que d'afficher le top-1 seul.
    ton <- if (best$confiance >= 0.70) "success"
           else if (best$confiance >= 0.40) "warning" else "danger"

    tagList(
      div(class = paste0("alert alert-", ton, " mb-3"),
          h4(class = "alert-heading mb-1", best$race_fr),
          div(class = "small text-muted", best$race),
          div(class = "mt-2", sprintf("Confiance : %.1f %%", 100 * best$confiance)),
          if (best$confiance < 0.40)
            div(class = "small mt-2",
                "Confiance faible : la photo est peut-etre ambigue, ou la race ",
                "ne fait pas partie des 120 apprises.")
          else if (best$confiance < 0.70)
            div(class = "small mt-2",
                "Confiance moderee : consultez les autres candidats ci-dessous.")
      )
    )
  })

  # --- Barres -----------------------------------------------------------------
  output$bars <- renderPlot({
    r <- result(); req(r)
    dt <- as_prediction_table(r)
    dt[, race_fr := factor(race_fr, levels = rev(race_fr))]

    ggplot(dt, aes(x = race_fr, y = confiance)) +
      geom_col(fill = "#2c7fb8", width = 0.65) +
      geom_text(aes(label = sprintf("%.1f %%", 100 * confiance)),
                hjust = -0.15, size = 3.6) +
      coord_flip() +
      scale_y_continuous(labels = scales::percent, limits = c(0, 1.18)) +
      labs(x = NULL, y = NULL) +
      theme_minimal(base_size = 12) +
      theme(panel.grid.major.y = element_blank())
  })

  # --- Tableau ----------------------------------------------------------------
  output$table <- renderTable({
    r <- result(); req(r)
    dt <- as_prediction_table(r)
    data.frame(
      Rang       = seq_len(nrow(dt)),
      Race       = dt$race_fr,
      `Nom source` = dt$race,
      Synset     = ifelse(is.na(dt$synset), "-", dt$synset),
      Confiance  = sprintf("%.2f %%", 100 * dt$confiance),
      check.names = FALSE
    )
  }, striped = TRUE, hover = TRUE, width = "100%")

  output$timing <- renderText({
    e <- elapsed(); r <- result()
    if (is.null(e) || is.null(r)) return("")
    infer <- r$meta$inference_ms %||% r$inference_ms %||% NA
    sprintf("Aller-retour : %.0f ms%s | moteur : %s", e,
            if (!is.na(infer)) sprintf(" (inference : %.0f ms)", infer) else "",
            if (identical(input$backend, "api")) "API REST" else "local")
  })

  # --- Onglet Modele ----------------------------------------------------------
  output$metadata <- renderText({
    meta <- if (identical(input$backend, "api")) {
      h <- api_health(input$api_url)
      if (!is.null(h)) h$metadata else NULL
    } else {
      service_status()$metadata
    }
    if (is.null(meta) && file.exists(PATH_METADATA)) {
      meta <- jsonlite::read_json(PATH_METADATA)
    }
    if (is.null(meta)) return("Metadonnees indisponibles (modele non entraine).")
    jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE)
  })

  output$pipeline_desc <- renderUI({
    HTML('
<pre style="line-height:1.5; font-size:0.85rem;">
image envoyee (JPEG / PNG / WebP, taille libre)
      |
      v  decodage, conversion sRGB 3 canaux, redimensionnement 224x224
tableau 1 x 224 x 224 x 3, valeurs 0-255
      |
      v  ENTREE DU MODELE
[ rescaling  x/127.5 - 1 ]      normalisation integree au modele
      |
[ MobileNetV2 sans tete  ]      poids ImageNet, dernieres couches affinees
      |
[ GlobalAveragePooling2D ]      7 x 7 x 1280  ->  1280
      |
[ Dropout 0.2            ]
      |
[ Dense 120, softmax     ]
      |
      v
distribution de probabilite  ->  top-3  ->  synset + nom francais
</pre>
<p class="small text-muted">La normalisation des pixels fait partie du graphe
du modele. Un client n\'a donc rien a normaliser lui-meme, et ne peut pas se
tromper de convention.</p>')
  })

  output$breeds <- renderTable({
    if (file.exists(PATH_BREEDS_FR)) {
      b <- fread(PATH_BREEDS_FR, encoding = "UTF-8")
      data.frame(`Nom source` = b$breed_label, `Nom francais` = b$breed_fr,
                 check.names = FALSE)
    } else {
      data.frame(Message = "Table des races indisponible.")
    }
  }, striped = TRUE, width = "100%")

  # --- Onglet Documentation ---------------------------------------------------
  # shiny::markdown() convertit le Markdown en HTML sans dependance
  # supplementaire : les rapports produits par le pipeline sont donc lisibles
  # directement dans l'application, sans etre recopies a la main.
  output$doc_audit    <- renderUI(
    shiny::markdown(read_report("audit_report.md", "Problemes rencontres")))
  output$doc_cleaning <- renderUI(
    shiny::markdown(read_report("cleaning_report.md", "Nettoyage")))
  output$doc_eval     <- renderUI(
    shiny::markdown(read_report("evaluation_report.md", "Evaluation")))

  # --- Sorties des onglets non affiches au demarrage --------------------------
  # Par defaut, Shiny suspend une sortie tant qu'elle reste invisible et ne la
  # calcule qu'a l'affichage. Dans une barre de navigation bslib, ce reveil ne
  # se produit pas de facon fiable : la sortie reste alors indefiniment en
  # "recalculating", sans erreur ni message -- l'onglet parait simplement vide.
  #
  # Ces sorties sont peu couteuses (trois fichiers Markdown, un CSV, du HTML
  # statique) : on les calcule donc systematiquement.
  for (.id in c("metadata", "pipeline_desc", "breeds",
                "doc_audit", "doc_cleaning", "doc_eval")) {
    outputOptions(output, .id, suspendWhenHidden = FALSE)
  }
}

shinyApp(ui, server)
