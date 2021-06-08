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
import model2
import DataManager
import hparams
import torch.nn as nn
import time
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_warmup as warmup

def check_output_size():
    hp = hparams.HParams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device={}".format(device))
    trainset, _, _ = DataManager.get_dataloader(hp)

    dummy_model = model2.Music1DCNN_ver2()
    with torch.no_grad():
        for data in trainset:
            waveform, label = data
            print(waveform.size())
            print("labebl={}".format(label))
            #print("size of feature space is: {}".format(dummy_model(waveform)))
            modulelist = list(dummy_model.conv_layer.modules())
            for layer in modulelist[1:]:
                waveform = layer(waveform)
                print(waveform.size())
            print("size of feature space is: {}".format(waveform.size()))
            break


def train():
    print("starting training...")
    hp = hparams.HParams()
    train_loader,valid_loader,test_loader = DataManager.get_dataloader(hparams=hp)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    music_classify = model.Music1DCNN()
    lr = hp.learning_rate
    optimizer = torch.optim.Adam(music_classify.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer,last_step=5)
    hp = hparams.HParams()
    number_of_ephocs = hp.num_epochs
    print("starting train loop...")
    for epoch in range(1, number_of_ephocs+1):
        music_classify.train()
        running_loss = 0.0
        epoch_time = time.time()
        total_tracks = 0
        total_correct = 0

        #validloader!!!!!!!!!
        for i,data in enumerate(valid_loader):

            waveform,label = data
            waveform.to(device)
            label.to(device)

            outputs = music_classify(waveform)
            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
        running_loss /= len(valid_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        log = "Epoch: {}  training loss: {:.3f} | train acc: {} | time: {}".format(epoch, running_loss,model_accuracy,epoch_time)
        print(log)
        scheduler.step()
        warmup_scheduler.dampen()


def calculate_accuracy(model, dataloader, device):
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10, 10], int)
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1
    loss /= len(dataloader)
    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix

if __name__ == '__main__':
    check_output_size()
