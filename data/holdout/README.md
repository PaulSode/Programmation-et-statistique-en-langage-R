# Jeu de controle externe

Deposez ici des photos de chiens **qui ne proviennent pas de Stanford Dogs**,
organisees en sous-dossiers portant le **nom de race tel qu'affiche** par le modele (colonne `breed_label` de `data/class_index.csv`).

```
data/holdout/
├── golden retriever/
│   ├── photo1.jpg
│   └── photo2.jpg
└── beagle/
    └── photo3.jpg
```

`R/06_evaluate_model.R` detecte automatiquement ce dossier et y mesure la
precision top-1.

## Pourquoi c'est necessaire

Stanford Dogs est un sous-ensemble d'ImageNet, et MobileNetV2 est pre-entraine
sur ImageNet — qui contient ces memes 120 classes de chiens. Le reseau de base a
donc potentiellement deja vu, pendant son pre-entrainement, les images du jeu de
test. La precision mesuree sur ce jeu est optimiste.

Des photos externes (smartphone, web, jeu prive) sont la seule mesure honnete de
generalisation. Voir README, probleme **P10**.
