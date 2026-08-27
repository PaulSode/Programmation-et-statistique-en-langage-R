# Dossier `models/`

Contenu produit par [`R/05_train_model.R`](../R/05_train_model.R). Rien ici ne
doit être écrit à la main.

| Fichier | Rôle |
|---|---|
| `model.keras` | Modèle entraîné. **Le prétraitement des pixels est intégré au graphe.** |
| `class_names.json` | Noms des classes, dans l'ordre exact des sorties du modèle |
| `class_index.csv` | Correspondance indice / synset ImageNet / nom de race |
| `model_metadata.json` | Provenance, hyperparamètres, contrat d'entrée, métriques |
| `checkpoint_*.keras` | Meilleur état de chaque phase d'entraînement |

Le dossier est vide tant que l'entraînement n'a pas été lancé. L'API et la
WebApp démarrent quand même, en mode dégradé : `GET /api/health` renvoie `503`
et `POST /api/predict` explique que le modèle est absent.

## Contrat d'entrée

- **Forme** : `(N, 224, 224, 3)`
- **Valeurs** : `0-255`, en virgule flottante
- **Ne PAS diviser par 255**, ne pas normaliser en amont

La mise à l'échelle `x/127.5 - 1` est une **couche du modèle**
(`layer_rescaling`). C'est délibéré : dès que la normalisation vit dans le code
appelant, elle doit être répétée à l'identique dans l'entraînement,
l'évaluation, l'API et la WebApp. La moindre divergence fait recevoir au modèle
une distribution qu'il n'a jamais vue — sans lever d'erreur, en produisant
simplement des prédictions fausses au classement plausible.

En intégrant la normalisation au graphe, le fichier `.keras` devient
autosuffisant : quiconque le charge obtient forcément le bon prétraitement.

## Contrat de sortie

- **Forme** : `(N, 120)`
- La sortie est **déjà une distribution de probabilité** (couche `softmax`).
  **Ne pas réappliquer de softmax** : cela laisserait le classement intact mais
  tasserait toutes les confiances autour de 1/120, rendant les pourcentages
  affichés dénués de sens.

## Cohérence indices / noms

`class_names.json` est écrit par le même script que `model.keras` : les deux ne
peuvent pas diverger. Le service vérifie en plus, au chargement, que le nombre
de sorties du modèle égale le nombre de noms, et refuse de démarrer sinon.

Sans ce verrou, une liste de classes décalée d'un seul indice rendrait toutes
les prédictions fausses — silencieusement.
