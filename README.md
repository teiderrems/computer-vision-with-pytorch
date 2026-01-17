# Computer Vision with PyTorch - Travaux Pratiques

Ce dépôt contient plusieurs notebooks et fichiers d'accompagnement pour des travaux pratiques d'initiation à la vision par ordinateur avec PyTorch (classification et convolution). Le contenu est organisé en séances (`td1` à `td4`) couvrant : préparation des données, perceptron multicouche, filtrage par convolution et premier CNN.

**Structure du dépôt (actuelle)**
- Les notebooks Jupyter sont placés à la racine du dépôt (exemples : `01_pb_without_CNN_correction.ipynb`, `02_convolution_filter.ipynb`, `03_convolution_image.ipynb`, `04_first_CNN.ipynb`, `05_data_augmentation.ipynb`, `06_classifiy_cats_dogs.ipynb`, `08_kp_regression_resnet.ipynb`, `09_YOLO_data_format.ipynb`, ainsi que plusieurs fichiers `Copy of ...`).
- `run_notebooks.sh` : script utilitaire pour lister/exécuter les notebooks (`list` / `execute`).
- Fichiers auxiliaires : `requirements.txt`, `README.md`, `LICENSE`, etc.

**Notebooks principaux**
- `01_pb_without_CNN_correction.ipynb` : pipeline MLP pour Fashion-MNIST (préparation, entraînement, évaluation, visualisations).
- `02_convolution_filter.ipynb` : exploration de la convolution 2D avec NumPy puis PyTorch.
- `03_convolution_image.ipynb` : application de filtres et visualisations sur images.
- `04_first_CNN.ipynb` : implémentation d'un premier réseau convolutionnel simple.
- `05_data_augmentation.ipynb`, `06_classifiy_cats_dogs.ipynb`, `08_kp_regression_resnet.ipynb`, `09_YOLO_data_format.ipynb` : autres exercices et exemples.

**Liste complète des notebooks (dans le dépôt)**
- `01_pb_without_CNN_correction.ipynb` : MLP pour Fashion-MNIST — préparation des données, entraînement et évaluation (sans convolution).
- `02_convolution_filter.ipynb` : Théorie et implémentation de filtres de convolution 2D (NumPy et PyTorch).
- `03_convolution_image.ipynb` : Application de filtres sur images avec visualisations d'exemples.
- `04_first_CNN.ipynb` : Premier réseau convolutionnel simple — architecture, entraînement et poids d'exemple.
- `05_data_augmentation.ipynb` : Techniques d'augmentation (flips, rotations, color jitter, random crop) et effets sur les performances.
- `06_classifiy_cats_dogs.ipynb` : Exemple de classification d'images (Cats vs Dogs) — pipeline de dataset, dataloader, modèle, entraînement.
- `08_kp_regression_resnet.ipynb` : Régression de points clés (keypoint) en s'appuyant sur un backbone ResNet.
- `09_YOLO_data_format.ipynb` : Format de données attendu par YOLO — annotation et conversion au format YOLO.
- `Copy of 07_transfer_learning_VGG.ipynb` : Notebook de transfert d'apprentissage avec VGG (copie de travail).
- `Copy of 10_YOLO_prepare.ipynb` : Préparation des datasets pour entraînement YOLO (copie de travail).
- `Copy of 11_YOLO_train.ipynb` : Exemples d'entraînement YOLO et scripts d'initialisation (copie de travail).
- `Copy of 12_conv_transpose.ipynb` : Exploration des convolutions transposées et opérations d'upsampling (copie de travail).
- `Copy of 13_seg_dataset.ipynb` : Préparation de datasets pour segmentation sémantique (copie de travail).
- `Copy of 14_custom_unet.ipynb` : Implémentation d'un UNet personnalisé pour segmentation (copie de travail).
- `Copy of 15_segmentation_models_pytorch.ipynb` : Utilisation de la bibliothèque `segmentation_models_pytorch` pour architectures/entrainement (copie de travail).

> Remarque : les fichiers marqués "Copy of ..." sont des duplicatas/versions de travail. Si vous souhaitez, nous pouvons les renommer ou les organiser dans un sous-dossier `work/` pour clarifier la structure.
**Dépendances (suggestions)**
- Python 3.8+ recommandé.
- Bibliothèques Python : `torch`, `torchvision`, `numpy`, `matplotlib`, `tqdm`, `scikit-learn`, `torchsummary`.

Exemple rapide pour préparer un environnement et installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision matplotlib numpy tqdm scikit-learn torchsummary
```

Remarque : installez la version de `torch` adaptée à votre GPU / CUDA si nécessaire en suivant les instructions officielles (https://pytorch.org).

**Utilisation des notebooks**
- Ouvrir un notebook avec `jupyter lab` ou `jupyter notebook` depuis la racine du dépôt.
- Les notebooks incluent des cellules pour télécharger les données (`torchvision.datasets.FashionMNIST`) si elles ne sont pas déjà présentes.
- Pour exécuter les notebooks interactivement, activez l'environnement virtuel et lancez :

```bash
source .venv/bin/activate
jupyter lab
```

- Pour l'exécution automatisée ou lister les notebooks, utilisez le script `run_notebooks.sh` (rendre exécutable avec `chmod +x run_notebooks.sh`). Le script recherche récursivement les fichiers `*.ipynb` en ignorant les répertoires `.ipynb_checkpoints` et les environnements virtuels communs (`.venv`, `venv`, `env`).

```bash
# lister les notebooks trouvés
./run_notebooks.sh list

# exécuter tous les notebooks (timeout illimité par défaut)
./run_notebooks.sh execute

# ajuster le timeout d'exécution (en secondes), par exemple 600 pour 10 minutes
NBEXEC_TIMEOUT=600 ./run_notebooks.sh execute
```

- L'exécution utilise `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=...` pour exécuter chaque notebook en place.
**Emplacement des données et des poids**
- Les jeux de données et poids ne sont pas inclus par défaut dans le dépôt actuel. Les notebooks contiennent des cellules pour télécharger automatiquement les datasets via `torchvision.datasets` (ex. FashionMNIST).
- Si vous souhaitez fournir des poids localement, placez-les sous `model_weights/` à la racine (ex. `model_weights/04_simple_cnn_model.pth`) et adaptez les chemins dans les notebooks.

Pour charger des poids dans un modèle PyTorch, exemple :

```python
# map_location='cpu' si pas de GPU
model.load_state_dict(torch.load('model_weights/04_simple_cnn_model.pth', map_location=device))
```

**Conseils et points d'attention**
- Vérifiez le `device` (CPU / CUDA / MPS) avant d'exécuter les notebooks : les notebooks détectent dynamiquement `torch.device` mais adaptez si besoin.
- Certains notebooks affichent des visualisations interactives ; exécutez-les dans un environnement Jupyter avec interface graphique.
- Sauvegardes de modèles incluses pour réexécution sans entraînement complet.

**Licence & Contribution**
- Ce dépôt est conçu pour un usage pédagogique. Ajoutez une licence si vous prévoyez une diffusion publique (par ex. `MIT`).
- Pour contribuer : fork -> modification -> pull request. Pour questions, ouvrez une issue.

