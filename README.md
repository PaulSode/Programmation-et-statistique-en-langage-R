# Classification de races de chiens — R

Classifieur d'images en **120 races de chiens** (jeu Stanford Dogs), entièrement
en R : audit des données, nettoyage, entraînement par transfer learning,
évaluation, **API REST** et **WebApp**.

**Résultat sur le jeu de test** (3 187 images jamais vues, ouvertes une seule
fois) : **80,0 % top-1** et **95,2 % top-3**. Le hasard donnerait 0,8 %.

```
audit  →  nettoyage  →  datasets  →  entraînement  →  évaluation
                                          ↓
                                    models/model.keras
                                       ↙        ↘
                              API REST          WebApp
                              (plumber)         (Shiny)
```

---

## Démarrage rapide

```bash
Rscript R/00_install_dependencies.R
```

```bash
Rscript -e "keras3::install_keras(backend = 'tensorflow')"
```

> **Espace disque.** TensorFlow et ses dépendances pèsent ~1,5 Go, installés par
> défaut sous `~/.virtualenvs/r-keras` — donc sur `C:` sous Windows. Si le disque
> système est juste, l'installation échoue en cours de route
> (`No space left on device`). Installez alors l'environnement ailleurs :
>
> ```r
> Sys.setenv(WORKON_HOME = "D:/venvs", PIP_CACHE_DIR = "D:/pip-cache")
> keras3::install_keras(backend = "tensorflow", envname = "D:/venvs/r-keras")
> ```
>
> puis, dans les sessions suivantes, indiquez l'interpréteur au projet :
> `Sys.setenv(DOGCLF_PYTHON = "D:/venvs/r-keras/Scripts/python.exe")`
> (variable lue par `R/common/config.R`).

Vérifier l'installation avant de lancer quoi que ce soit de long :

```bash
Rscript R/98_smoke_test.R
```

Puis le pipeline complet (téléchargement de 757 Mo, puis entraînement) :

```bash
Rscript R/run_pipeline.R
```

Enfin les deux services :

```bash
Rscript api/run_api.R
```

```bash
Rscript app/run_app.R
```

L'API écoute sur `http://localhost:5000`, la WebApp sur `http://localhost:3838`.
Tant que `models/model.keras` n'existe pas, les deux démarrent en mode dégradé
et le disent explicitement (`503`), plutôt que de servir des valeurs de repli.

---

# 1. Problèmes rencontrés sur le jeu de données

Le jeu est **Stanford Dogs** : 20 580 photos réparties en 120 races, dérivé
d'ImageNet. Le script [`R/02_audit_data.R`](R/02_audit_data.R) mesure chacun des
problèmes ci-dessous et produit `reports/audit_report.md` — il n'y a pas de
« on suppose que les données sont propres ».

### Ce qui a effectivement été mesuré

Audit exécuté sur les 20 580 fichiers. Plusieurs contrôles ne trouvent **rien** :
c'est un résultat, pas une formalité — il fallait le vérifier pour le savoir.

| | Contrôle | Mesuré |
|---|---|---|
| P1 | Noms de classes à décomposer | 120 dossiers |
| P2 | Fichiers illisibles | **0** |
| P3 | Images hors sRGB | **0** |
| P4 | Extension mensongère | **1** (`n02105855_2933.jpg`, en réalité un PNG) |
| P5 | Doublons exacts (MD5) | **89**, répartis sur 43 races |
| P6 | Déséquilibre | 148 à 252 images/race, ratio **1,70**, médiane 160 |
| P7 | Résolutions hétérogènes | 97×100 à 3264×2562 px, ratios 0,39 à 3,63 |
| P8 | Images < 64 px de côté | **0** (le plus petit côté observé est 97 px) |
| P9 | Races indiscernables | 27 races en 10 groupes |
| P10 | Fuite ImageNet | structurel, non chiffrable |
| P11 | Protocole d'évaluation | structurel, non chiffrable |

Sur ce jeu, **seuls P4 et P5 déclenchent une exclusion** — 89 images écartées sur
20 580 (0,43 %). Les autres contrôles restent utiles : ils garantissent que ce
silence est constaté, et ils protègent le pipeline si on lui donne un autre jeu
de données.

### P1 — Noms de classes illisibles

Les dossiers s'appellent `n02085620-Chihuahua` : l'identifiant synset ImageNet
est collé au nom de race. Utilisés tels quels comme labels, ils produisent des
sorties opaques. Le synset doit malgré tout être **conservé séparément** : c'est
la clé de jointure vers la traduction française et les métadonnées ImageNet.

### P2 — Fichiers corrompus ou illisibles

**Mesuré : 0.** Un fichier qui ne se décode pas serait écarté silencieusement
par TensorFlow à l'entraînement — on croirait alors entraîner sur *n* images
alors qu'on en utilise moins. Le contrôle reste en place : côté service, une
image illisible doit produire un `400` explicite, ce que l'API fait.

### P3 — Images hors sRGB

**Mesuré : 0** — les 20 580 fichiers sont déclarés sRGB. Un décodeur qui ne
force pas explicitement 3 canaux renvoie, pour une image en niveaux de gris ou
un PNG avec alpha, un tableau de forme `(H, W, 1)` ou `(H, W, 4)` incompatible
avec l'entrée `(224, 224, 3)`. La conversion reste donc appliquée
systématiquement : elle ne coûte rien ici et rend le décodage robuste à
n'importe quelle image entrante — y compris celles que l'API recevra en
production, qui, elles, ne sont pas contrôlées.

### P4 — Extensions mensongères

**Mesuré : 1.** Le fichier `n02105855_2933.jpg` (berger des Shetland) est en
réalité un **PNG** de 213×189 px. Filtrer ou décoder en se fiant à l'extension
n'est donc pas fiable ; c'est le contenu qui fait foi.

### P5 — Doublons exacts

**Mesuré : 89**, répartis sur 43 races (jusqu'à 7 pour le griffon bruxellois).
C'est le seul défaut réellement présent en nombre.

Avec un split aléatoire, une image dupliquée peut se retrouver **à la fois en
entraînement et en validation** : le modèle est alors évalué sur une image qu'il
a mémorisée, et la précision de validation est gonflée sans que rien ne le
signale. La déduplication a donc lieu **avant** le découpage.

### P6 — Effectif faible et déséquilibre modéré

**Mesuré :** de 148 (`redbone`) à 252 (`Maltese_dog`) images par race, médiane
160, moyenne 171,5, ratio max/min **1,70**. Le déséquilibre reste supportable,
mais l'effectif **absolu** est faible.
Entraîner un CNN de zéro sur si peu d'exemples sur-apprendrait immédiatement —
c'est précisément ce qui justifie le transfer learning.

### P7 — Résolutions et ratios d'aspect très hétérogènes

**Mesuré :** largeurs de 97 à 3264 px, hauteurs de 100 à 2562 px (médiane
500×375), ratios d'aspect de 0,39 à 3,63. Le redimensionnement forcé en 224×224
déforme donc sensiblement les images les plus allongées. C'est un choix assumé —
appliqué **à l'identique** à l'entraînement et au service — et non un oubli.

### P8 — Images minuscules

**Mesuré : 0** — le plus petit côté observé est de 97 px, une seule image passant
sous 100 px. Le seuil de 64 px ne déclenche donc jamais ici. Il reste utile comme
garde-fou : une image plus petite, agrandie en 224×224, n'apporterait que du
flou d'interpolation.

### P9 — Races visuellement indiscernables

Une dizaine de groupes sont ambigus même pour un expert humain sur une seule
photo : husky sibérien / malamute / chien esquimau, caniche toy / nain /
standard (distingués par la **taille réelle**, invisible sur une photo),
schnauzers idem, Norfolk / Norwich terrier, Pembroke / Cardigan, bouviers
suisses…

**Conséquence directe sur la conception** : une partie de l'erreur est
irréductible et vient de la définition des classes, pas du modèle. C'est
pourquoi la métrique suivie est la **précision top-3** autant que la top-1, et
pourquoi l'API renvoie 3 candidats plutôt qu'un seul.

### P10 — Fuite de données via le pré-entraînement ImageNet

**Le problème le plus important, et le moins visible.** Stanford Dogs est un
sous-ensemble d'ImageNet ; MobileNetV2 est pré-entraîné sur ImageNet-1k, qui
contient exactement ces 120 classes de chiens. Le réseau de base a donc
potentiellement **déjà vu, pendant son pré-entraînement, les images mêmes de
notre jeu de validation**.

Toute précision mesurée sur ce jeu est donc optimiste et ne mesure pas la
généralisation à de nouvelles photos. Ce problème ne se corrige pas par du
nettoyage : il se **documente**, et se compense en évaluant sur des photos
externes (voir `data/holdout/` dans [`R/06_evaluate_model.R`](R/06_evaluate_model.R)).

### P11 — Un protocole d'évaluation naïf serait biaisé

Un simple split 80/20 train/validation, dont on lirait la précision de
validation comme résultat final, serait **biaisé par sélection** : ce même jeu
sert à décider du nombre d'epochs, donc à choisir un modèle. Par ailleurs un
découpage qui mélange tous les chemins puis coupe en un bloc n'est **pas
stratifié** — les proportions par classe varient librement.

---

# 2. Comment les données ont été nettoyées

Implémenté dans [`R/03_clean_data.R`](R/03_clean_data.R), qui produit
`reports/cleaning_report.md`.

### Principe : un manifeste, pas des suppressions

**Aucun fichier n'est effacé du disque.** Le nettoyage produit
`data/manifest_clean.csv` — une ligne par image retenue — qui devient la seule
source de vérité du pipeline d'entraînement. Toute exclusion est réversible et
auditable : `reports/excluded_files.csv` dit pourquoi chaque image écartée l'a
été.

Laisser le chargeur de données décider seul de ce qui entre dans le modèle
reviendrait à ne jamais savoir sur quoi on a réellement entraîné.

### Règles d'exclusion

Appliquées dans l'ordre ; une image n'est comptée que pour la **première** règle
qui la rejette, ce qui rend le bilan additif.

| # | Règle | Répond à |
|---|---|---|
| 1 | Image non décodable par ImageMagick | P2 |
| 2 | Plus petit côté < 64 px | P8 |
| 3 | Ratio d'aspect hors de `[1/4, 4]` | P7 |
| 4 | Doublon exact (MD5 déjà vu) — la première occurrence est gardée | P5 |
| 5 | Image multi-frames (GIF animé, TIFF multipage) | P3 |
| 6 | Classe comptant moins de 50 images survivantes | P6 |

### Normalisations — appliquées à la volée, pas en réécrivant les fichiers

- **Séparation synset / nom de race** (P1) : `n02085620-Chihuahua` devient trois
  colonnes — `synset`, `breed_raw`, `breed_label`. Le synset reste disponible
  pour la jointure avec `data/breeds_fr.csv` (traduction française).
- **Conversion sRGB 3 canaux** (P3) et **redimensionnement 224×224** (P7) :
  réalisés au moment du décodage par `image_to_tensor()`
  ([`R/common/utils_image.R`](R/common/utils_image.R)). Une seule
  implémentation, partagée par l'entraînement, l'évaluation, l'API et la WebApp.
- **Mise à l'échelle des pixels** : elle n'est **pas** faite ici. Elle est
  intégrée au modèle. Voir section 3.

### Split stratifié 70 / 15 / 15 (P11)

Trois propriétés exigées :

1. **Stratifié** — la répartition est appliquée race par race, donc chaque race
   est représentée dans les mêmes proportions dans les trois sous-ensembles.
2. **Un jeu de test existe**, et n'est ouvert qu'une seule fois, dans
   [`R/06_evaluate_model.R`](R/06_evaluate_model.R). La validation pilote
   l'entraînement, le test rapporte le résultat — confondre les deux rôles rend
   la métrique finale optimiste.
3. **Reproductible** — l'affectation est écrite dans le manifeste. Rejouer le
   pipeline ne redistribue pas les images.

Deux contrôles automatiques suivent le split :

- chaque classe est bien présente dans `train`, `val` **et** `test` ;
- **aucun MD5 n'apparaît dans deux sous-ensembles** (contrôle anti-fuite).

### Artefacts produits

| Fichier | Contenu |
|---|---|
| `data/manifest_clean.csv` | une ligne par image retenue, avec son split |
| `data/class_index.csv` | correspondance indice ↔ synset ↔ race — **source unique de vérité** |
| `data/class_weights.csv` | poids de classe (optionnels, voir section 3) |
| `reports/excluded_files.csv` | détail de chaque exclusion |
| `reports/cleaning_report.md` | bilan chiffré |

---

# 3. Modélisation, du preprocessing à la prédiction

Implémentée dans [`R/05_train_model.R`](R/05_train_model.R).

### Architecture

```
image brute 224 x 224 x 3, valeurs 0-255
      |
      v
[ data_augmentation      ]   actif à l'entraînement, inerte en inférence
      |                      flip horizontal, rotation ±10 %, zoom ±10 %, contraste ±10 %
      v
[ rescaling  x/127.5 - 1 ]   -> plage [-1, 1] attendue par MobileNetV2
      |
      v
[ MobileNetV2 sans tête  ]   poids ImageNet ; gelé en phase 1,
      |                      30 dernières couches dégelées en phase 2
      v
[ GlobalAveragePooling2D ]   7 x 7 x 1280  ->  1280
      |
      v
[ Dropout 0.2            ]
      |
      v
[ Dense 120, softmax     ]   -> distribution de probabilité sur les races
```

### Le prétraitement est **dans** le modèle

C'est la décision de conception la plus importante du projet.

Dès que la normalisation vit dans le code appelant, elle doit être répétée à
l'identique partout où le modèle est utilisé — entraînement, évaluation, API,
WebApp. La moindre divergence (diviser par 255 d'un côté, centrer sur `[-1, 1]`
de l'autre) fait recevoir au modèle une distribution qu'il n'a jamais vue. Rien
ne plante : les prédictions deviennent simplement fausses, avec un classement
encore plausible.

En faisant de la normalisation une **couche** (`layer_rescaling(scale = 1/127.5,
offset = -1)`), le fichier `.keras` devient autosuffisant : quiconque le charge
obtient forcément le bon prétraitement. Le code appelant n'a plus qu'à fournir
des pixels bruts 0-255, et ne peut plus se tromper de convention.

Le contrat est explicité dans `models/model_metadata.json` et exposé par
`GET /api/metadata`.

### Pourquoi le transfer learning

~171 images par race (P6) : un CNN entraîné de zéro sur-apprendrait
immédiatement. MobileNetV2 apporte des détecteurs de formes, textures et motifs
appris sur 1,2 M d'images. On ne réapprend que la couche de décision.

MobileNetV2 plutôt qu'un réseau plus lourd : ~3,5 M de paramètres et une
inférence de quelques dizaines de millisecondes sur CPU — ce qui rend l'API
utilisable sans GPU, contrainte réelle de déploiement.

### Entraînement en deux phases

**Phase 1 — tête seule** (10 epochs max, `lr = 1e-3`). Base gelée, seuls les
poids de la couche `Dense` finale sont appris.

**Phase 2 — fine-tuning** (8 epochs max, `lr = 1e-5`). Les 30 dernières couches
du réseau de base sont dégelées pour se spécialiser sur les textures de pelage
et les silhouettes canines, que les features ImageNet génériques ne séparent
qu'imparfaitement.

Deux précautions indispensables, souvent omises :

1. **Taux d'apprentissage divisé par 100.** Avec le `lr` de la phase 1, les
   gradients issus d'une tête encore imparfaite détruiraient les poids
   pré-entraînés dès la première itération.
2. **Les couches `BatchNormalization` restent gelées.** Un batch de 32 est trop
   petit pour réestimer des statistiques fiables ; les dégeler fait typiquement
   chuter la précision de validation de plusieurs points. C'est aussi la raison
   du `training = FALSE` sur l'appel au réseau de base.

### Callbacks

- `early_stopping` sur `val_accuracy`, patience 4, **restauration des meilleurs
  poids** (et non ceux de la dernière epoch, souvent moins bons) ;
- `reduce_lr_on_plateau` sur `val_loss`, facteur 0,5 ;
- `model_checkpoint` du meilleur état ;
- `csv_logger` pour la traçabilité.

### Pondération des classes — mesurée, puis écartée

`data/class_weights.csv` est calculé, mais **désactivé par défaut**. Avec un
ratio max/min de ~1,7, pondérer ajoute surtout de la variance au gradient sans
gain de précision. Le mécanisme reste disponible via `DOGCLF_CLASS_WEIGHTS=1`.

### Évaluation ([`R/06_evaluate_model.R`](R/06_evaluate_model.R))

Sur le **jeu de test**, ouvert pour la première fois :

- précision **top-1** et **top-3** (voir P9 sur la pertinence du top-3) ;
- précision / rappel / F1 **par race** ;
- **matrice de confusion** et **paires les plus confondues**, avec confrontation
  chiffrée de l'hypothèse P9 : quelle part de l'erreur tombe réellement dans les
  groupes déclarés ambigus *avant* l'entraînement. Mesuré ici : **20,3 %** des
  erreurs et 11 des 25 confusions les plus fréquentes — l'hypothèse explique une
  part notable de l'erreur, pas la majorité ;
- **courbe de calibration** — l'API annonce un score de confiance à ses clients ;
  il faut donc vérifier que parmi les prédictions annoncées à 90 %, environ 90 %
  sont correctes. Un modèle mal calibré trompe l'utilisateur même quand sa
  précision globale est bonne. La WebApp s'appuie sur ce résultat pour nuancer
  sa réponse quand la confiance est faible ;
- **jeu de contrôle externe** (`data/holdout/<race>/`) — seule mesure honnête de
  généralisation, compte tenu de P10.

### Prédiction ([`R/common/predict_service.R`](R/common/predict_service.R))

```
image (fichier ou octets)
   -> décodage, sRGB 3 canaux, 224x224, valeurs 0-255
   -> predict()                      # sortie déjà softmax : PAS de second softmax
   -> ordre décroissant, top-3
   -> jointure class_index.csv       # synset réel de la race prédite
   -> jointure breeds_fr.csv         # nom français
   -> JSON
```

Trois verrous, chacun destiné à rendre impossible une panne **silencieuse** :

- **Pas de prédictions fictives.** Un modèle indisponible produit une erreur
  explicite (`503`), jamais une valeur de repli mise en forme comme un résultat.
- **Vérification de cohérence au chargement** : le nombre de sorties du modèle
  doit égaler le nombre de noms de classes. Sinon le service refuse de démarrer.
  Un décalage d'un seul indice rendrait toutes les prédictions fausses — en
  silence.
- **Chargement unique**, dans un environnement dédié : latence prévisible.

---

# 4. API REST

[`api/plumber.R`](api/plumber.R) — plumber transforme des fonctions R annotées
en endpoints HTTP.

```bash
Rscript api/run_api.R
```

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/api/predict` | Classifie une image (`multipart/form-data`, champ `image`) |
| `GET` | `/api/health` | Sonde de disponibilité — `200` si prêt, `503` sinon |
| `GET` | `/api/breeds` | Les 120 races, dans l'ordre des sorties du modèle |
| `GET` | `/api/metadata` | Provenance et contrat d'entrée du modèle déployé |
| `GET` | `/__docs__/` | Documentation OpenAPI interactive (générée) |

```bash
curl -X POST "http://localhost:5000/api/predict?top_k=3" -F "image=@photo.jpg"
```

```json
{
  "model_type": "dog_breed_classifier",
  "predictions": [
    {
      "class_index": 56,
      "class_id": "n02099601",
      "class_name": "golden_retriever",
      "class_label": "golden retriever",
      "class_name_fr": "Golden Retriever",
      "synset": "n02099601-golden_retriever",
      "confidence": 0.912344
    }
  ],
  "top_prediction": { "...": "..." },
  "meta": { "filename": "photo.jpg", "size_kb": 184.2, "inference_ms": 41.3, "top_k": 3 }
}
```

Formats d'entrée acceptés : `multipart/form-data` (champ `image`), corps binaire
brut, ou JSON `{"image_base64": "..."}`.

### Codes de réponse

Principe : **une erreur doit toujours être distinguable d'un résultat.**

| Situation | Code |
|---|---|
| Succès | `200` |
| Aucune image, ou image illisible (erreur du client) | `400` |
| Type d'analyse inconnu | `400` |
| Envoi > 5 Mo | `413` |
| `type=binary` — aucun modèle binaire n'est entraîné | `501` |
| Modèle absent ou illisible | `503` |
| Échec d'inférence inattendu | `500` |

CORS : l'origine de la requête est validée contre une liste blanche puis
renvoyée telle quelle — les navigateurs refusent `*` dès qu'il y a des
credentials.

---

# 5. WebApp

[`app/app.R`](app/app.R) — Shiny.

```bash
Rscript app/run_app.R
```

Trois onglets :

- **Prédiction** — dépôt d'image, aperçu, top-*k* avec noms français, barres de
  confiance. Un avertissement explicite s'affiche sous 70 % puis sous 40 % de
  confiance, en s'appuyant sur la courbe de calibration mesurée à l'évaluation.
- **Modèle** — métadonnées du modèle réellement servi, schéma de la chaîne de
  traitement, liste des 120 races.
- **Données et méthode** — affiche directement les rapports d'audit, de
  nettoyage et d'évaluation produits par le pipeline. Ils ne sont pas recopiés à
  la main : la documentation ne peut donc pas diverger du code.

Deux moteurs d'inférence, sélectionnables dans la barre latérale :

- **API REST** — la WebApp envoie l'image en HTTP, comme n'importe quel client.
  C'est ce mode qui valide le déploiement de bout en bout.
- **Local** — le modèle est chargé dans le processus Shiny, pour démontrer
  l'application sans lancer deux services.

---

# 6. Déploiement

```bash
docker compose up --build
```

- API : `http://localhost:5000` — docs sur `/__docs__/`
- WebApp : `http://localhost:3838`

`models/` est monté **en volume lecture seule** plutôt que copié dans l'image :
le modèle change à chaque réentraînement et n'a pas à déclencher la
reconstruction de l'image. Réentraîner puis redémarrer suffit à déployer une
nouvelle version.

Le `HEALTHCHECK` interroge `/api/health`, qui renvoie `503` tant que le modèle
n'est pas chargé : l'orchestrateur ne route pas de trafic vers un conteneur
inapte, et la WebApp attend que l'API soit saine (`depends_on: condition:
service_healthy`).

---

# 7. Structure du projet

```
.
├── R/
│   ├── common/
│   │   ├── config.R             constantes partagées (une seule définition)
│   │   ├── bootstrap.R          amorçage commun
│   │   ├── utils_image.R        décodage / prétraitement — implémentation unique
│   │   └── predict_service.R    service d'inférence
│   ├── 00_install_dependencies.R
│   ├── 01_download_data.R       téléchargement + extraction
│   ├── 02_audit_data.R          problèmes du jeu de données
│   ├── 03_clean_data.R          nettoyage + split stratifié
│   ├── 04_build_datasets.R      pipeline tf.data + augmentation
│   ├── 05_train_model.R         transfer learning + fine-tuning
│   ├── 06_evaluate_model.R      évaluation sur jeu de test
│   ├── 07_predict.R             prédiction en ligne de commande
│   ├── 98_smoke_test.R          vérification de l'environnement
│   └── run_pipeline.R           orchestration
├── api/
│   ├── plumber.R                endpoints REST
│   └── run_api.R
├── app/
│   ├── app.R                    WebApp Shiny
│   └── run_app.R
├── data/
│   ├── breeds_fr.csv            traduction française des 120 races
│   ├── holdout/                 jeu de contrôle externe (voir P10)
│   ├── raw/                     archive et images (non versionné)
│   ├── manifest_clean.csv       produit par 03
│   └── class_index.csv          produit par 03
├── models/                      produits par 05
├── reports/                     rapports et figures
├── Dockerfile, docker-compose.yml
└── README.md
```

## Variables d'environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `DOGCLF_ROOT` | déduite | racine du projet |
| `DOGCLF_PYTHON` | — | interpréteur Python portant TensorFlow |
| `DOGCLF_API_HOST` / `DOGCLF_API_PORT` | `0.0.0.0` / `5000` | écoute de l'API |
| `DOGCLF_API_URL` | `http://127.0.0.1:5000` | API ciblée par la WebApp |
| `DOGCLF_APP_PORT` | `3838` | écoute de la WebApp |
| `DOGCLF_CACHE_TRAIN` | `FALSE` | cache RAM des images d'entraînement (~8 Go) |
| `DOGCLF_CLASS_WEIGHTS` | `0` | active la pondération des classes |

---

## Résultats de l'entraînement

Exécuté de bout en bout sur les 20 580 images, R 4.6.1 / TensorFlow 2.21 /
Keras 3.15, **CPU seul** (Intel i7-1165G7, 4 cœurs — TensorFlow ≥ 2.11 n'a plus
de support GPU natif sous Windows).

| Étape | Durée | Résultat |
|---|---|---|
| Téléchargement + extraction | ~9 min | 120 classes, 20 580 images |
| Audit | ~11 min | 89 doublons, 1 extension mensongère |
| Nettoyage + split | < 1 min | 20 491 images → 14 292 / 3 012 / 3 187 |
| Phase 1 — tête seule | 50 min | val 76,1 % top-1 / 93,8 % top-3 |
| Phase 2 — fine-tuning | 42 min | val 80,0 % top-1 / 95,2 % top-3 |
| **Évaluation sur test** | ~2 min | **80,04 % top-1 / 95,20 % top-3** |

Le fine-tuning apporte **+3,9 points** de top-1 par rapport à la tête seule.
`reduce_lr_on_plateau` s'est déclenché deux fois (1e-3 → 5e-4, puis 1e-5 → 5e-6),
avec un gain net et immédiat à chaque fois.

**Test ≈ validation** (80,04 vs 80,01 ; 95,20 vs 95,15) : le jeu de validation
n'a donc pas été sur-exploité par l'arrêt anticipé, et le protocole de la
section 2 tient.

### Le score de confiance est sur-confiant

La courbe de calibration montre un écart systématique :

| Confiance annoncée | Précision réelle |
|---|---|
| 85,4 % | 73,3 % |
| 98,2 % | 94,0 % |

Le modèle annonce plus de certitude qu'il n'en a. C'est pourquoi la WebApp
avertit sous 70 % puis sous 40 % de confiance plutôt que d'afficher le score
brut sans commentaire.

### Races les mieux et les moins bien reconnues

| Bien reconnues (F1) | | Mal reconnues (F1) | |
|---|---|---|---|
| keeshond | 1,00 | collie | 0,41 |
| Sealyham terrier | 0,98 | Yorkshire terrier | 0,42 |
| bluetick | 0,98 | chien esquimau | 0,47 |
| lycaon | 0,98 | whippet | 0,52 |

Le Yorkshire terrier a une **précision de 1,00 mais un rappel de 0,27** : quand
le modèle l'annonce il a raison, mais il le rate trois fois sur quatre — 16 de
ses images partent en « silky terrier ». C'est le type de diagnostic qu'une
précision globale seule masquerait complètement.

---

## État de vérification

Exécuté sur R 4.6.1 / TensorFlow 2.21 / Keras 3.15 (Windows 11) :

| Vérification | Résultat |
|---|---|
| `parse()` de tous les fichiers R | 0 erreur de syntaxe |
| `R/98_smoke_test.R --api …` | **28 PASS / 0 FAIL** |
| `R/02_audit_data.R` + `R/03_clean_data.R` sur jeu synthétique | **17 PASS / 0 FAIL** — les défauts injectés (corrompus, doublons, ratio aberrant, PNG déguisé) sont tous détectés et écartés |
| API : health, breeds, metadata, predict, 400/413/501/503, CORS, OpenAPI | conformes |
| WebApp : upload → API → prédiction, mode local, 3 onglets | fonctionnels |
| Cohérence moteur API vs moteur local | **prédictions identiques** |

Le pipeline a désormais tourné **intégralement sur les vraies données**, du
téléchargement à l'évaluation.

Deux bugs ont été trouvés en le lançant, tous deux silencieux :

1. **`download.file()` et le timeout de 60 s.** La valeur par défaut de R rend
   tout téléchargement de 757 Mo impossible ; il s'interrompait à 519 Mo.
   Corrigé par `curl::multi_download(resume = TRUE)`, avec reprise et
   vérification de la taille finale.
2. **`initial_epoch` est indexé en base 1 dans keras3.**
   `fit(initial_epoch = 3, epochs = 5)` exécute les epochs 3, 4 *et* 5 — trois
   epochs, là où l'API Python en ferait deux. La phase 2 aurait tourné 18 epochs
   au lieu de 8, soit environ une heure de calcul perdue sans le moindre
   message. L'argument n'est plus utilisé.

### Un piège vérifié par test

`magick` définit une méthode `as.integer.bitmap()` qui **transpose
silencieusement** le tableau de `(canaux, largeur, hauteur)` vers
`(hauteur, largeur, canaux)`. Une implémentation qui l'ignore mélange axes et
canaux : le modèle reçoit du bruit et répond quand même, sans la moindre erreur.

`image_to_tensor()` contourne la méthode via `unclass()`, et
[`R/98_smoke_test.R`](R/98_smoke_test.R) vérifie la disposition **au pixel
près** sur une mire de 8 couleurs connues. C'est le seul type de test qui
détecte ce genre de défaut.

## Notes techniques

- R ne réimplémente pas TensorFlow : `keras3` le pilote via `reticulate`. Un
  backend Python est donc nécessaire (`keras3::install_keras()`), y compris en
  production — d'où l'environnement virtuel dans le `Dockerfile`.
- `magick` (ImageMagick) assure tout le traitement d'image.
- `data.table` porte le manifeste : 20 580 lignes avec jointures fréquentes.
- Les indices de classes sont en **base 0** partout (convention Keras), et
  convertis en base 1 uniquement au moment d'indexer un vecteur R.
