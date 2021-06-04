import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import sklearn

import librosa
import librosa.display
import IPython.display as ipd
import warnings
#warnings.filterwarnings('ignore')
import os
import torchaudio
import torchaudio.transforms as transforms
import torch

def load_gtza_from_torch():
    train_transforms = transforms.Tensor()
    test_transforms = transforms.Tensor()

    trainset = torchaudio.datasets.GTZAN(root="./datasets/train", download=True,subset="training")
    validset = torchaudio.datasets.GTZAN(root="./datasets/validation", download=True, subset="validation")
    testset = torchaudio.datasets.GTZAN(root="./datasets/test", download=True, subset="testing")






if __name__ == '__main__':
    load_gtza_from_torch()