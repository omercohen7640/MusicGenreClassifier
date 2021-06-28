# MusicGenreClassifier
<h1 align="center">
  <br>
Technion EE 046211 - Deep Learning
  <br>
  <img src="https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nn_gumgum.gif" height="200">
</h1>
  <p align="center">
    <a href="https://taldatech.github.io">Omer Cohen</a> •
    <a href="https://github.com/royg27">Jonathan Nir Shalit</a>
  </p>

Our Project for the Technion's EE 046211 course "Deep Learning"
* Animation by <a href="https://medium.com/@gumgumadvertisingblog">GumGum</a>.

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/ee046211-deep-learning"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <a href="https://nbviewer.jupyter.org/github/taldatech/ee046211-deep-learning/tree/main/"><img src="https://jupyter.org/assets/main-logo.svg" alt="Open In NBViewer"/></a>
    <a href="https://mybinder.org/v2/gh/taldatech/ee046211-deep-learning/main"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a>

</h4>


- [OurProject](#OurProject)
  * [Agenda](#agenda)
  * [MusicGenreClassifier](#MusicGenreClassifier)
  * [Dataset](#Dataset)
    + [-----](#-----)
  * [Data augmentation](#installation-instructions)
    + [-----](#-----)
  * [1D-Classifier](#installation-instructions)
    + [Failer](#Failer)
  * [2D-Classifier](#installation-instructions)
    + [Feature extraction](#Feature extraction)
    + [Ensaemble](#Ensemble)
  * [Results](#Results)






## Agenda
As our final project in Deep Learning course we have been asked to choose a problem and to solve it using neural network and deep learning techniques. We chose to implement DL algorithm that classify music track's genre.

## MusicGenreClassifier
 The algorithm's input is a 30 seconds long music track, and the output is one of the following genres: Blues, Rock, Classic, Reggae, Disco, Country, Hip-Hop, Metal, Jazz and Pop. Throughout our work we experimented several approches to solve this problem both via the data and the architecture

## Dataset
We used the widely used [GTZAN](http://marsyas.info/downloads/datasets.html) dataset. the dataset includes 10 classes of music genres. Each class contains 100 tracks of 30 seconds. that is, we faced a low-amount-of-data problem.

### -----

## Data augmentation
To enlarge our dataset, we used data augmentations. To execute those augmentation easily, we used Librosa package.
We have used the following augmentations:

<img src="/img/data_aug.png">
### -----

## 1D-Classifier
Our first trial  to improve model's performance was to work with the raw data and to use 1D convnet. We tried 2 architetures that yileded same performances:

first:

<img src="/img/1dconvnetver1.png">

second:

<img src="/img/1dconvnetver2.png">

We tested our model on 10-classes dataset and we got poor 10% accuracy performance (random prediction). We tried to use chopped sub-tracks of different lengths and still the performance didn’t improve.
At this point we concluded:
1.	Working with the raw data (without any pre-processing) is more difficult and requires more sophisticated architectures. 
2.	Working with 2D data allows us to use known computer-vision architectures  and techniques.

## 2D-Classifier
### Feature extraction
Working with 2D input means transforming the data into time-frequency space of mel-spectrogram. We used Librosa tools to transform the data. Here is an ilustration for the transform:

<img src="/img/original_track.png">

<img src="/img/original_track_db.png">

<img src="/img/mel.png">



We used resnet18 architecture with dropout.We used Optuna to tune our hyper parameters. That model achieved 62.4% accuracy on the dataset.




### Ensemble
someting....


## Result
here we right things about 

