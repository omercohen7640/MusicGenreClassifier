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
from hparams import hparams
import torch.nn as nn
import time
import torch.optim.lr_scheduler as lr_scheduler
#import pytorch_warmup as warmup
import datamanager_ver2
import model_cnn2d
import optuna
import warmup as my_warmup



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def check_output_size():
    print("device={}".format(device))
    trainset, _, _ = DataManager.get_dataloader(hp)

    dummy_model = model_cnn2d.CNN_2D_V2(hparams)
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
    train_loader, valid_loader,test_loader = DataManager.get_dataloader(hparams=hp)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    music_classify = model2.Music1DCNN_ver2()
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


def load_data():
    train_loader, valid_loader, test_loader = datamanager_ver2.get_dataloader(hparams);
    return train_loader, valid_loader, test_loader
def train_cnn_2d(train_loader, valid_loader, test_loader,music_classify = model_cnn2d.CNN_2D_V2(hparams) ):
    print("starting training...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    lr = hparams.learning_rate
    optimizer = torch.optim.Adam(music_classify.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=hparams.factor, patience=hparams.patience, verbose=True)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer, last_step=5)
    number_of_ephocs = hparams.num_epochs
    print("starting train loop...")
    for epoch in range(1, number_of_ephocs + 1):
        music_classify.train()
        running_loss = 0.0
        epoch_time = time.time()
        total_tracks = 0
        total_correct = 0
        print("lr={}".format(get_lr(optimizer)))

        for i, data in enumerate(train_loader):
            waveform, label = data
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
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        log = "Epoch: {}  training loss: {:.3f} | train acc: {} | time: {}".format(epoch, running_loss, model_accuracy,
                                                                                   epoch_time)
        print(log)
        scheduler.step(metrics=running_loss)
        warmup_scheduler.dampen()




def train_cnn_2d_pre_net(train_loader, valid_loader, test_loader,music_classify = model_cnn2d.CNN_2D_V2(hparams) ):
    print("starting training...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    lr = hparams.learning_rate
    warmup_factor = hparams.warmup_factor
    init_lr = lr / warmup_factor

    optimizer = torch.optim.Adam(music_classify.parameters(), lr=init_lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=hparams.factor, patience=hparams.patience, verbose=True)
    #warmup_scheduler = warmup.UntunedLinearWarmup(optimizer, last_step=5)
    print("begin warmup loop...")
    warmup_epochs = hparams.warmup_epochs
    num_steps = warmup_epochs * len(train_loader)
    warmuper = my_warmup.LinearWarmuper(optimizer=optimizer, steps=num_steps, factor=1e2)
    for w_epoch in range(warmup_epochs + 1):
        print("info: lr={}".format(get_lr(optimizer)))
        for i, data in enumerate(train_loader):
            waveform, label = data
            waveform.to(device)
            label.to(device)
            size = waveform.size()
            inputs = torch.zeros(size[0], 3, size[1], size[2])
            inputs[:, 0, :, :] = waveform
            inputs[:, 1, :, :] = waveform
            inputs[:, 2, :, :] = waveform
            inputs = inputs.to(device)
            outputs = music_classify(inputs)
            outputs = music_classify(inputs)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmuper.step()



    number_of_ephocs = hparams.num_epochs
    print("starting train loop...")
    for epoch in range(1, number_of_ephocs + 1):
        music_classify.train()
        running_loss = 0.0
        epoch_time = time.time()
        total_tracks = 0
        total_correct = 0
        print("info: lr={}".format(get_lr(optimizer)))

        for i, data in enumerate(train_loader):
            waveform, label = data
            waveform.to(device)
            label.to(device)
            size = waveform.size()
            inputs = torch.zeros(size[0], 3, size[1], size[2])
            inputs[:, 0, :, :] = waveform
            inputs[:, 1, :, :] = waveform
            inputs[:, 2, :, :] = waveform
            inputs = inputs.to(device)
            outputs = music_classify(inputs)
            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        valid_accuracy,_,valid_loss = calculate_accuracy(music_classify,valid_loader,device, criterion)
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| valid acc: {} | time: {}".format(epoch, running_loss, model_accuracy, valid_accuracy,
                                                                                   epoch_time)
        print(log)
        scheduler.step(metrics=valid_accuracy)
        #warmup_scheduler.dampen()
    test_accuracy,_,_ =  calculate_accuracy(music_classify,test_loader,device,criterion=criterion)
    return test_accuracy


def calculate_accuracy(model, dataloader, device , criterion):
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10, 10], int)
    runningn_loss = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            size = images.size()
            inputs = torch.zeros(size[0], 3, size[1], size[2])
            inputs[:, 0, :, :] = images
            inputs[:, 1, :, :] = images
            inputs[:, 2, :, :] = images
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            runningn_loss += loss
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1
    runningn_loss /= len(dataloader)
    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix, runningn_loss

def objective(trial):
    music_classify, _ = model_cnn2d.initialize_model(model_name = 'resnet',num_classes=10,feature_extract=False,use_pretrained=False)
    print("loading data...")
    train_loader, valid_loader, test_loader = load_data()
    print("starting training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    warmup_factor = trial.suggest_float("warmup_factor", 1e1, 1e3, log=True)
    init_lr = lr/warmup_factor
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(torch.optim, optimizer_name)(music_classify.parameters(), lr=init_lr)
    sched_factor = trial.suggest_float("sched_factor",0,1)
    patience = trial.suggest_int("patience",3,9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=sched_factor, patience=patience,
                                               verbose=True)
    #warmup section:
    warmup_epochs = trial.suggest_int("warmup_ephocs",2,5)
    num_steps = warmup_epochs*len(train_loader)

    warmuper = my_warmup.LinearWarmuper(optimizer=optimizer,steps=num_steps, factor = warmup_factor)
    for w_epoch in range(warmup_epochs+1):
        total_tracks = 0
        total_correct = 0
        epoch_time = time.time()
        for i, data in enumerate(train_loader):
            running_loss = 0.0
            waveform, label = data
            waveform.to(device)
            label.to(device)
            size = waveform.size()
            inputs = torch.zeros(size[0], 3, size[1], size[2])
            inputs[:, 0, :, :] = waveform
            inputs[:, 1, :, :] = waveform
            inputs[:, 2, :, :] = waveform
            inputs = inputs.to(device)
            outputs = music_classify(inputs)
            outputs = music_classify(inputs)
            loss = criterion(outputs, label)
            running_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmuper.step()
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| time: {}".format(w_epoch, running_loss,
                                                                                                  model_accuracy,
                                                                                                  epoch_time)

    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer, last_step=5)
    ephocs = trial.suggest_int("ephocs",20,45)
    print("starting train loop...")
    for epoch in range(1, ephocs + 1):
        music_classify.train()
        running_loss = 0.0
        epoch_time = time.time()
        total_tracks = 0
        total_correct = 0
        print("info: lr={}".format(get_lr(optimizer)))

        for i, data in enumerate(train_loader):
            waveform, label = data
            waveform.to(device)
            label.to(device)
            size = waveform.size()
            inputs = torch.zeros(size[0], 3, size[1], size[2])
            inputs[:, 0, :, :] = waveform
            inputs[:, 1, :, :] = waveform
            inputs[:, 2, :, :] = waveform
            inputs = inputs.to(device)
            outputs = music_classify(inputs)
            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        valid_accuracy, _, valid_loss = calculate_accuracy(music_classify, valid_loader, device, criterion)
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| valid acc: {} | time: {}".format(epoch, running_loss,
                                                                                                  model_accuracy,
                                                                                                  valid_accuracy,
                                                                                                  epoch_time)
        print(log)
        scheduler.step(metrics=valid_accuracy)
        # warmup_scheduler.dampen()
    test_accuracy, _, _ = calculate_accuracy(music_classify, test_loader, device, criterion=criterion)
    return test_accuracy

def run_parameter_tuning():
    study = optuna.create_study(study_name= "music classifier", direction="maximize",sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=hparams.number_of_trials)
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    tarin_loader, valid_loader, test_loader = load_data()
    feature_extract = False
    model, inpuut_size = model_cnn2d.initialize_model(model_name='resnet', num_classes=10,
                                                      feature_extract=feature_extract, use_pretrained=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_cnn_2d_pre_net(tarin_loader, valid_loader, test_loader, music_classify=model)
