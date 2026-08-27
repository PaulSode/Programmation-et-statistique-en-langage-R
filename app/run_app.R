# ==============================================================================
# app/run_app.R -- Lanceur de la WebApp Shiny
#
# Usage :
#     Rscript app/run_app.R
#     Rscript app/run_app.R --port 3838 --host 0.0.0.0
#
# La WebApp et l'API sont deux processus independants. Pour la chaine complete,
# lancez d'abord l'API :
#     Rscript api/run_api.R      (terminal 1)
#     Rscript app/run_app.R      (terminal 2)
#
# La WebApp fonctionne aussi seule, en basculant le moteur d'inference sur
# "Modele local" dans la barre laterale.
# ==============================================================================

.script <- {
  a <- commandArgs(trailingOnly = FALSE)
  h <- grep("^--file=", a, value = TRUE)
  if (length(h)) sub("^--file=", "", h[1]) else "app/run_app.R"
}
root <- normalizePath(file.path(dirname(.script), ".."), winslash = "/", mustWork = TRUE)
Sys.setenv(DOGCLF_ROOT = root)
setwd(root)

source(file.path(root, "R", "common", "config.R"))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- which(args == flag)
  if (length(i) && length(args) > i[1]) args[i[1] + 1L] else default
}
host <- get_arg("--host", "127.0.0.1")
port <- as.integer(get_arg("--port", Sys.getenv("DOGCLF_APP_PORT", "3838")))

# Taille maximale des fichiers acceptes par Shiny, alignee sur celle de l'API.
options(shiny.maxRequestSize = MAX_UPLOAD_MB * 1024^2)

cat(sprintf("
+---------------------------------------------------------------+
|  WebApp de classification de races de chiens (Shiny)          |
+---------------------------------------------------------------+
|  Interface   http://%s:%d                            |
|  API cible   %-46s |
+---------------------------------------------------------------+

", host, port, API_BASE_URL))

if (!file.exists(PATH_MODEL)) {
  log_warn("Modele absent : ", PATH_MODEL,
           " -- le mode 'Modele local' sera indisponible.")
}

shiny::runApp(file.path(root, "app"), host = host, port = port, launch.browser = FALSE)
