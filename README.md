# Computer Vision with PyTorch - Travaux Pratiques

Ce dépôt contient plusieurs notebooks et fichiers d'accompagnement pour des travaux pratiques d'initiation à la vision par ordinateur avec PyTorch (classification et convolution). Le contenu est organisé en séances (`td1` à `td4`) couvrant : préparation des données, perceptron multicouche, filtrage par convolution et premier CNN.

**Structure du dépôt**
- `td1/` : TP « Classification Fashion-MNIST sans Convolution » (`01_pb_without_CNN.ipynb`) et jeu de données partiel + poids modèle.
- `td2/` : TP « Comprendre la Convolution » (`02_convolution_filter.ipynb`).
- `td3/` : TP sur l'application des convolutions aux images (`03_convolution_image.ipynb`) et images exemples.
- `td4/` : TP « Premier CNN » (`04_first_CNN.ipynb`) et poids du modèle simple.

- `td*/data/FashionMNIST/raw/` : fichiers bruts du dataset Fashion-MNIST (idx3 / idx1).
- `td*/model_weights/` : poids modèles sauvegardés (ex. `01_fmnist_model_no_cnn.pth`, `04_simple_cnn_model.pth`).

**Notebooks principaux**
- `td1/01_pb_without_CNN.ipynb` : pipeline complet d'un MLP pour Fashion-MNIST (préparation, entraînement, évaluation, visualisations). 
- `td2/02_convolution_filter.ipynb` : exploration pas-à-pas de la convolution 2D avec NumPy puis PyTorch.
- `td3/03_convolution_image.ipynb` : exemples d'application de filtres et visualisations sur images.
- `td4/04_first_CNN.ipynb` : implémentation d'un premier réseau convolutionnel simple et expérimentation.

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
- Pour exécuter les notebooks, activez l'environnement virtuel et lancez :

```bash
source .venv/bin/activate
jupyter lab
```

**Emplacement des données et des poids**
- Jeux de données téléchargés : `td1/data/FashionMNIST/raw/` et `td4/data/FashionMNIST/raw/`.
- Poids des modèles : `td1/model_weights/01_fmnist_model_no_cnn.pth`, `td4/model_weights/04_simple_cnn_model.pth`.

Pour charger des poids dans un modèle PyTorch, exemple :

```python
# map_location='cpu' si pas de GPU
model.load_state_dict(torch.load('td1/model_weights/01_fmnist_model_no_cnn.pth', map_location=device))
```

**Conseils et points d'attention**
- Vérifiez le `device` (CPU / CUDA / MPS) avant d'exécuter les notebooks : les notebooks détectent dynamiquement `torch.device` mais adaptez si besoin.
- Certains notebooks affichent des visualisations interactives ; exécutez-les dans un environnement Jupyter avec interface graphique.
- Sauvegardes de modèles incluses pour réexécution sans entraînement complet.

**Licence & Contribution**
- Ce dépôt est conçu pour un usage pédagogique. Ajoutez une licence si vous prévoyez une diffusion publique (par ex. `MIT`).
- Pour contribuer : fork -> modification -> pull request. Pour questions, ouvrez une issue.

