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
import model

def load_gtza_from_torch():
    train_transforms = transforms.Tensor()
    test_transforms = transforms.Tensor()
    batch_size = 64
    num_workers=2

    trainset = torchaudio.datasets.GTZAN(root="./datasets/train", download=True,subset="training")
    validset = torchaudio.datasets.GTZAN(root="./datasets/validation", download=True, subset="validation")
    testset = torchaudio.datasets.GTZAN(root="./datasets/test", download=True, subset="testing")

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    return train_loader, validation_loader, test_loader, trainset


def check_output_size():
    #train_loader, _, _ = load_gtza_from_torch()
    #train_loader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, _, _, trainset = load_gtza_from_torch()
    dummy_model = model.Music1DCNN().to(device)
    dummy_model.eval()
    waveform,_,_ = trainset[25]
    waveform = waveform.unsqueeze(0)
    waveform = waveform.to(device)
    modulelist = list(dummy_model.conv_layer.modules())
    for layer in modulelist[1:]:
        waveform = layer(waveform)

    print("size of feature space is: {}".format(waveform.size()))

if __name__ == '__main__':
    check_output_size()
