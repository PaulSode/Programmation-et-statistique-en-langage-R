# ==============================================================================
# 05_train_model.R -- MODELISATION
#
#   >>> Ce script repond a la question : "Quelle est la modelisation, du
#       preprocessing a la prediction ?"
#
# Transfer learning a partir de MobileNetV2 pre-entraine sur ImageNet, en deux
# phases : tete de classification seule, puis fine-tuning des dernieres couches.
#
# Architecture du modele produit (le pretraitement est DANS le graphe) :
#
#   image brute 224x224x3, valeurs 0-255
#        v
#   [data_augmentation]      actif a l'entrainement, inerte en inference
#        v
#   [rescaling 1/127.5 - 1]  -> plage [-1, 1] attendue par MobileNetV2
#        v
#   [MobileNetV2 sans tete]  poids ImageNet ; gele en phase 1, partiellement
#        v                   degele en phase 2
#   [GlobalAveragePooling2D] 7x7x1280 -> 1280
#        v
#   [Dropout 0.2]
#        v
#   [Dense 120, softmax]     -> distribution de probabilite sur les races
#
# Pourquoi la mise a l'echelle est une COUCHE et non une etape externe :
# des que la normalisation vit dans le code appelant, elle doit etre repetee a
# l'identique partout ou le modele est utilise -- entrainement, evaluation, API,
# WebApp. La moindre divergence (diviser par 255 d'un cote, centrer sur [-1, 1]
# de l'autre) fait recevoir au modele une distribution qu'il n'a jamais vue, et
# ne provoque aucune erreur : seulement des predictions fausses.
# En integrant la normalisation au graphe, le fichier .keras devient
# autosuffisant -- quiconque le charge obtient forcement le bon pretraitement.
# ==============================================================================

source("R/common/bootstrap.R")
source("R/04_build_datasets.R")

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(jsonlite)
  library(ggplot2)
  library(data.table)
})

log_step("05 -- Entrainement du modele")

# Reproductibilite : graine unique pour R, NumPy et TensorFlow.
keras3::set_random_seed(SEED)

log_info("TensorFlow : ", tensorflow::tf$`__version__`)
log_info("GPU disponibles : ", length(tensorflow::tf$config$list_physical_devices("GPU")))

# ------------------------------------------------------------------------------
# 1. Donnees
# ------------------------------------------------------------------------------
# cache_train = FALSE par defaut : mettre les ~14 000 images d'entrainement en
# cache RAM demande environ 8 Go. Passez a TRUE si la machine le permet.
data <- build_datasets(cache_train = as.logical(Sys.getenv("DOGCLF_CACHE_TRAIN", "FALSE")))
n_classes <- data$n_classes

# ------------------------------------------------------------------------------
# 2. Reseau de base pre-entraine
# ------------------------------------------------------------------------------
log_info("Chargement de MobileNetV2 (poids ImageNet)...")

base_model <- keras3::application_mobilenet_v2(
  input_shape = c(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS),
  include_top = FALSE,          # on jette la tete ImageNet a 1000 classes
  weights     = "imagenet"
)
base_model$trainable <- FALSE   # phase 1 : extracteur de caracteristiques fige

log_info("Parametres du reseau de base : ",
         format(base_model$count_params(), big.mark = " "))

# ------------------------------------------------------------------------------
# 3. Assemblage
# ------------------------------------------------------------------------------
augment <- build_data_augmentation()

inputs  <- keras3::keras_input(shape = c(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS),
                               name = "image")
x <- augment(inputs)

# Normalisation attendue par MobileNetV2 : x / 127.5 - 1, qui envoie la plage
# [0, 255] sur [-1, 1].
x <- keras3::layer_rescaling(x, scale = 1 / 127.5, offset = -1,
                             name = "mobilenetv2_preprocessing")

# training = FALSE fige les statistiques des couches BatchNormalization du
# reseau de base. Sans cela, elles continueraient a se mettre a jour meme avec
# trainable = FALSE, et les features derivereraient des le premier batch.
x <- base_model(x, training = FALSE)

x <- keras3::layer_global_average_pooling_2d(x, name = "embedding")
x <- keras3::layer_dropout(x, rate = DROPOUT_RATE, name = "dropout")
outputs <- keras3::layer_dense(x, units = n_classes, activation = "softmax",
                               name = "breed_probabilities")

model <- keras3::keras_model(inputs, outputs, name = "dog_breed_classifier")

# ------------------------------------------------------------------------------
# 4. Compilation
# ------------------------------------------------------------------------------
# sparse_categorical_crossentropy : les labels sont des entiers, pas du one-hot.
# On suit aussi la precision top-3, qui est la metrique reellement pertinente
# ici (voir probleme P9 de l'audit : plusieurs races sont indiscernables sur
# une seule photo) et qui correspond a ce que l'API renvoie.
compile_model <- function(m, lr) {
  compile(
    m,
    optimizer = keras3::optimizer_adam(learning_rate = lr),
    loss      = "sparse_categorical_crossentropy",
    metrics   = list(
      keras3::metric_sparse_categorical_accuracy(name = "accuracy"),
      keras3::metric_sparse_top_k_categorical_accuracy(k = 3L, name = "top3_accuracy")
    )
  )
}
compile_model(model, LR_HEAD)

summary(model)

# ------------------------------------------------------------------------------
# 5. Callbacks
# ------------------------------------------------------------------------------
# Lancer un nombre d'epochs fixe en aveugle gaspille du temps ou s'arrete trop
# tot. Ces callbacks stoppent l'entrainement quand il cesse de progresser et
# restaurent les MEILLEURS poids, au lieu de conserver ceux de la derniere
# epoch -- souvent moins bons.
make_callbacks <- function(tag) list(
  keras3::callback_early_stopping(
    monitor = "val_accuracy", patience = 4L, mode = "max",
    restore_best_weights = TRUE, verbose = 1L
  ),
  keras3::callback_reduce_lr_on_plateau(
    monitor = "val_loss", factor = 0.5, patience = 2L, min_lr = 1e-7, verbose = 1L
  ),
  keras3::callback_model_checkpoint(
    filepath = file.path(DIR_MODELS, sprintf("checkpoint_%s.keras", tag)),
    monitor = "val_accuracy", mode = "max", save_best_only = TRUE, verbose = 0L
  ),
  keras3::callback_csv_logger(
    filename = file.path(DIR_REPORTS, sprintf("training_log_%s.csv", tag)),
    append = FALSE
  )
)

# ------------------------------------------------------------------------------
# 5 bis. Poids de classe (optionnel)
# ------------------------------------------------------------------------------
# Le desequilibre mesure a l'audit est modere (ratio max/min ~ 1.7). Ponderer
# les classes dans ce regime ajoute surtout de la variance au gradient sans
# gain de precision. On laisse donc l'option desactivee par defaut, mais on
# expose le mecanisme.
class_weight <- NULL
if (identical(Sys.getenv("DOGCLF_CLASS_WEIGHTS"), "1")) {
  cw <- fread(file.path(DIR_DATA, "class_weights.csv"))
  class_weight <- stats::setNames(as.list(cw$weight), as.character(cw$label))
  log_info("Ponderation des classes activee.")
}

# ------------------------------------------------------------------------------
# 6. PHASE 1 -- entrainement de la tete seule
# ------------------------------------------------------------------------------
# Seuls les poids de la couche Dense finale sont appris. Le reseau de base,
# gele, ne sert que d'extracteur de caracteristiques : c'est ce qui rend
# l'entrainement possible avec ~170 images par race, la ou un CNN entraine de
# zero en exigerait des milliers.
log_step("Phase 1 -- tete de classification (base gelee)")

t0 <- Sys.time()
history_head <- fit(
  model,
  data$train,
  validation_data = data$val,
  epochs          = EPOCHS_HEAD,
  callbacks       = make_callbacks("head"),
  class_weight    = class_weight,
  verbose         = 1L
)
t_head <- difftime(Sys.time(), t0, units = "mins")
log_info(sprintf("Phase 1 terminee en %.1f min", as.numeric(t_head)))

# ------------------------------------------------------------------------------
# 7. PHASE 2 -- fine-tuning
# ------------------------------------------------------------------------------
# On degele les dernieres couches du reseau de base pour qu'elles se
# specialisent sur les textures de pelage et les silhouettes canines, que les
# features ImageNet generiques ne separent qu'imparfaitement.
#
# Deux precautions indispensables :
#   1. taux d'apprentissage divise par 100. Avec le LR de la phase 1, les
#      gradients issus d'une tete encore imparfaite detruiraient les poids
#      pre-entraines des la premiere iteration.
#   2. les couches BatchNormalization restent gelees. Le batch de 32 est trop
#      petit pour reestimer des statistiques fiables, et les degeler fait
#      typiquement chuter la precision de validation de plusieurs points.
log_step("Phase 2 -- fine-tuning des dernieres couches")

base_model$trainable <- TRUE
n_layers <- length(base_model$layers)
freeze_until <- n_layers - UNFREEZE_LAST_N

layer_class_name <- function(l) {
  tryCatch(as.character(l$`__class__`$`__name__`), error = function(e) "")
}

for (i in seq_len(n_layers)) {
  layer <- base_model$layers[[i]]
  is_batchnorm <- identical(layer_class_name(layer), "BatchNormalization")
  layer$trainable <- (i > freeze_until) && !is_batchnorm
}

n_trainable <- sum(vapply(base_model$layers, function(l) isTRUE(l$trainable), logical(1)))
log_info(sprintf("Couches degelees dans le reseau de base : %d / %d",
                 n_trainable, n_layers))

compile_model(model, LR_FINETUNE)   # recompilation obligatoire apres degel

# Nombre d'epochs reellement effectuees en phase 1 : l'arret anticipe a pu la
# ecourter. Sert uniquement a decaler la numerotation dans les courbes.
ep0 <- length(history_head$metrics$loss)

# On ne passe deliberement PAS initial_epoch. Dans keras3, cet argument est
# indexe a partir de 1 : fit(initial_epoch = 3, epochs = 5) execute les epochs
# 3, 4 ET 5, soit trois epochs -- la ou l'API Python de Keras, indexee a 0, n'en
# executerait que deux. Compter dessus pour reprendre la numerotation apres la
# phase 1 ferait donc tourner une epoch de trop, et un decalage d'un cran ici
# coute une heure de calcul.
#
# `epochs` compte simplement les epochs de CETTE phase. Le decalage de
# numerotation dans les courbes est applique nous-memes (argument `offset` de
# to_dt, plus bas).
t0 <- Sys.time()
history_ft <- fit(
  model,
  data$train,
  validation_data  = data$val,
  epochs           = EPOCHS_FINETUNE,
  callbacks        = make_callbacks("finetune"),
  class_weight     = class_weight,
  verbose          = 1L
)
t_ft <- difftime(Sys.time(), t0, units = "mins")
log_info(sprintf("Phase 2 terminee en %.1f min", as.numeric(t_ft)))

# ------------------------------------------------------------------------------
# 8. Sauvegarde des artefacts
# ------------------------------------------------------------------------------
log_step("Sauvegarde")

# Le modele et ses tables de reference sont ecrits par ce meme script : aucun
# artefact servi en production n'existe sans le code qui l'a produit.
keras3::save_model(model, PATH_MODEL, overwrite = TRUE)
log_info("Modele ecrit : ", PATH_MODEL)

# class_names.json : l'ordre des noms EST l'ordre des sorties du modele. Ce
# fichier est ecrit par le meme script que le modele, donc les deux ne peuvent
# pas diverger -- une liste maintenue separement finirait par se decaler, et un
# decalage d'un seul indice rendrait toutes les predictions fausses en silence.
jsonlite::write_json(data$class_names, PATH_CLASS_NAMES,
                     auto_unbox = TRUE, pretty = TRUE)
log_info("Noms de classes ecrits : ", PATH_CLASS_NAMES)

# Copie de l'index complet a cote du modele : l'API et la WebApp n'ont ainsi
# besoin que du dossier models/ pour fonctionner, sans acces a data/.
fwrite(data$class_index, file.path(DIR_MODELS, "class_index.csv"))

# Metadonnees : tout ce qu'un consommateur du modele doit savoir pour
# l'utiliser correctement, embarque avec lui.
metrics_ft  <- history_ft$metrics
final_epoch <- ep0 + length(metrics_ft$loss)

metadata <- list(
  model_name       = "dog_breed_classifier",
  created_at       = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  framework        = list(
    r          = as.character(getRversion()),
    keras      = tryCatch(as.character(keras3::keras$`__version__`),
                          error = function(e) NA_character_),
    tensorflow = as.character(tensorflow::tf$`__version__`)),
  architecture     = list(
    base            = "MobileNetV2",
    base_weights    = "imagenet",
    head            = "GlobalAveragePooling2D -> Dropout -> Dense(softmax)",
    unfrozen_layers = UNFREEZE_LAST_N
  ),
  input            = list(
    height          = IMG_HEIGHT,
    width           = IMG_WIDTH,
    channels        = N_CHANNELS,
    value_range     = "0-255 (la mise a l'echelle vers [-1,1] est faite par le modele)",
    note            = "Ne PAS diviser par 255 avant d'appeler le modele."
  ),
  output           = list(
    n_classes       = n_classes,
    activation      = "softmax",
    note            = "La sortie est deja une distribution de probabilite. Ne pas reappliquer softmax."
  ),
  training         = list(
    seed              = SEED,
    batch_size        = BATCH_SIZE,
    epochs_head       = ep0,
    epochs_finetune   = length(metrics_ft$loss),
    lr_head           = LR_HEAD,
    lr_finetune       = LR_FINETUNE,
    class_weights     = !is.null(class_weight),
    minutes_total     = round(as.numeric(t_head) + as.numeric(t_ft), 1)
  ),
  data             = list(
    source          = DATA_URL,
    n_train         = data$counts$train,
    n_val           = data$counts$val,
    n_test          = data$counts$test,
    manifest        = basename(PATH_MANIFEST)
  ),
  validation_metrics = list(
    accuracy        = as.numeric(tail(metrics_ft$val_accuracy, 1)),
    top3_accuracy   = as.numeric(tail(metrics_ft$val_top3_accuracy, 1)),
    loss            = as.numeric(tail(metrics_ft$val_loss, 1))
  )
)

jsonlite::write_json(metadata, PATH_METADATA, auto_unbox = TRUE, pretty = TRUE)
log_info("Metadonnees ecrites : ", PATH_METADATA)

# ------------------------------------------------------------------------------
# 9. Courbes d'apprentissage
# ------------------------------------------------------------------------------
to_dt <- function(h, phase, offset = 0L) {
  m <- h$metrics
  data.table(
    epoch    = offset + seq_along(m$loss),
    phase    = phase,
    loss     = as.numeric(m$loss),
    val_loss = as.numeric(m$val_loss),
    acc      = as.numeric(m$accuracy),
    val_acc  = as.numeric(m$val_accuracy)
  )
}

hist_dt <- rbind(
  to_dt(history_head, "1 - tete"),
  to_dt(history_ft,   "2 - fine-tuning", offset = ep0)
)
fwrite(hist_dt, file.path(DIR_REPORTS, "training_history.csv"))

long <- melt(hist_dt, id.vars = c("epoch", "phase"),
             measure.vars = c("acc", "val_acc"),
             variable.name = "serie", value.name = "accuracy")
long[, serie := factor(serie, levels = c("acc", "val_acc"),
                       labels = c("entrainement", "validation"))]

p <- ggplot(long, aes(x = epoch, y = accuracy, colour = serie, linetype = phase)) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.2) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Precision au fil des epochs",
       subtitle = "Phase 1 : base gelee -- Phase 2 : fine-tuning a faible taux d'apprentissage",
       x = "Epoch", y = "Precision", colour = NULL, linetype = NULL) +
  theme_minimal(base_size = 11)
ggsave(file.path(DIR_REPORTS, "training_accuracy.png"), p, width = 8, height = 5, dpi = 150)

log_info(sprintf("Precision de validation finale : %.2f %% (top-3 : %.2f %%)",
                 100 * metadata$validation_metrics$accuracy,
                 100 * metadata$validation_metrics$top3_accuracy))
log_info("Etape 05 terminee. Enchainez avec R/06_evaluate_model.R")
