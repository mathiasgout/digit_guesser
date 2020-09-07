# Digits Guesser

Un programme Python classifiant les chiffres dessinés à la main par les utilisateurs à l'aide d'un réseau de neuronne à convolution.
<br>Le modèle est entrainé avec le jeu de données MNIST.

<p align="center">
  <img src="https://raw.githubusercontent.com/mathiasgout/digits_guesser/master/images/README_gif.gif">
</p>

## Instructions

- `main.py` permet d'ouvrir l'application paint.
- `train_model.py` permet d'entrainer son propre CNN. Le modèle est sauvegardé dans le dossier `models/`.

## Requirements

Les packages suivants sont nécéssaires :

- python>=3.5
- tensorflow
- tkinter
- numpy
- pillow

### Installation locale

Il possible d'installer les packages en utilisant `pip` :
```
$ pip install -r requirements.txt
```
