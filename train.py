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
import logging
#import pytorch_warmup as warmup
import datamanager_ver2
import model_cnn2d
import optuna
import warmup as my_warmup
import resnet_dropout
from datetime import datetime as dt
import audio_augmentation_joni
import feature_extraction
import joblib


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


def train_1d(train_loader, valid_loader, test_loader, music_classify):
    print("starting training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    music_classify = music_classify.to(device)
    now = dt.now()
    lr_iterarion = []
    vall_acc_epoch = []
    train_loss_iter = []
    train_acc_epoch = []
    dt_string = now.strftime("%d%m%Y%H%M")
    path = os.path.join(os.getcwd(), "trial_" + dt_string)
    os.mkdir(path)
    criterion = nn.CrossEntropyLoss()
    lr = hparams.learning_rate
    warmup_factor = hparams.warmup_factor
    init_lr = lr / warmup_factor

    optimizer = torch.optim.Adam(music_classify.parameters(), lr=init_lr, weight_decay=hparams.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=hparams.threshold, mode='max',
                                               factor=hparams.factor, patience=hparams.patience, verbose=True)
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer, last_step=5)
    print("begin warmup loop...")
    warmup_epochs = hparams.warmup_epochs
    num_steps = warmup_epochs * len(train_loader)
    warmuper = my_warmup.LinearWarmuper(optimizer=optimizer, steps=num_steps, factor=1e2)
    for w_epoch in range(warmup_epochs + 1):
        total_tracks = 0
        total_correct = 0
        epoch_time = time.time()
        print("info: lr={}".format(get_lr(optimizer)))
        for i, data in enumerate(train_loader):
            running_loss = 0.0
            waveform, label = data
            waveform = waveform.to(device)
            label = label.to(device)
            outputs = music_classify(waveform)
            loss = criterion(outputs, label)
            train_loss_iter.append(loss)
            running_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmuper.step()
            lr_iterarion.append(get_lr(optimizer))
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| time: {}".format(w_epoch, running_loss, model_accuracy,
                                                                                  epoch_time)
        print(log)
    best_valid_acc = 0
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
            waveform = waveform.to(device)
            label = label.to(device)
            outputs = music_classify(waveform)
            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_iterarion.append(get_lr(optimizer))
            running_loss += loss.data.item()
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        valid_accuracy, _, valid_loss = calculate_accuracy_1d(music_classify, valid_loader, device, criterion)
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| valid acc: {} | time: {}".format(epoch, running_loss,
                                                                                                  model_accuracy,
                                                                                                  valid_accuracy,
                                                                                                  epoch_time)
        train_acc_epoch.append(model_accuracy)
        vall_acc_epoch.append(valid_accuracy)
        print(log)
        scheduler.step(metrics=valid_accuracy)
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            path_save = os.path.join(path, "best_model")
            torch.save(music_classify.state_dict(), path_save)
    lr_iterarion = np.array(lr_iterarion)
    vall_acc_epoch = np.array(vall_acc_epoch)
    train_loss_iter = np.array(train_loss_iter)
    train_acc_epoch = np.array(train_acc_epoch)
    np.save(os.path.join(path, 'lr_iteration'), lr_iterarion)
    np.save(os.path.join(path, 'vall_acc_epoch'), vall_acc_epoch)
    np.save(os.path.join(path, 'train_loss_iter'), train_loss_iter)
    np.save(os.path.join(path, 'train_acc_epoch'), train_acc_epoch)
    if test_loader is not None:
        test_accuracy, _, _ = calculate_accuracy_1d(music_classify, test_loader, device, criterion=criterion)
        path_save = os.path.join(path, "last_model")
        torch.save(music_classify.state_dict(), path_save)
        return test_accuracy

def load_data():
    train_loader, valid_loader, test_loader , test_ensemble_loader= datamanager_ver2.get_dataloader(hparams);
    return train_loader, valid_loader, test_loader, test_ensemble_loader


def train_cnn_2d(train_loader, valid_loader, test_loader,music_classify = model_cnn2d.CNN_2D_V2(hparams) ):
    print("starting training...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    lr = hparams.learning_rate
    optimizer = torch.optim.Adam(music_classify.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=hparams.factor, patience=hparams.patience, verbose=True)
    #warmup_scheduler = warmup.UntunedLinearWarmup(optimizer, last_step=5)
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




def train_cnn_2d_pre_net(train_loader, valid_loader, test_loader=None,music_classify = model_cnn2d.CNN_2D_V2(hparams) ):
    print("starting training...")
    now = dt.now()
    lr_iterarion = []
    vall_acc_epoch = []
    train_loss_iter = []
    train_acc_epoch = []
    dt_string = now.strftime("%d%m%Y%H%M")
    path = os.path.join(os.getcwd(),"trial_"+dt_string)
    os.mkdir(path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    music_classify = music_classify.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = hparams.learning_rate
    warmup_factor = hparams.warmup_factor
    init_lr = lr / warmup_factor

    optimizer = torch.optim.Adam(music_classify.parameters(), lr=init_lr, weight_decay=hparams.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,threshold=hparams.threshold, mode='max', factor=hparams.factor, patience=hparams.patience, verbose=True)
    #warmup_scheduler = warmup.UntunedLinearWarmup(optimizer, last_step=5)
    print("begin warmup loop...")
    warmup_epochs = hparams.warmup_epochs
    num_steps = warmup_epochs * len(train_loader)
    warmuper = my_warmup.LinearWarmuper(optimizer=optimizer, steps=num_steps, factor=1e2)
    for w_epoch in range(warmup_epochs + 1):
        total_tracks = 0
        total_correct = 0
        epoch_time = time.time()
        print("info: lr={}".format(get_lr(optimizer)))
        for i, data in enumerate(train_loader):
            running_loss = 0.0
            waveform, label = data
            waveform = waveform.to(device)
            label = label.to(device)
            size = waveform.size()
            inputs = torch.zeros(size[0], 3, size[1], size[2])
            inputs[:, 0, :, :] = waveform
            inputs[:, 1, :, :] = waveform
            inputs[:, 2, :, :] = waveform
            inputs = inputs.to(device)
            outputs = music_classify(inputs)
            loss = criterion(outputs, label)
            train_loss_iter.append(loss)
            running_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmuper.step()
            lr_iterarion.append(get_lr(optimizer))
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| time: {}".format(w_epoch, running_loss,model_accuracy,epoch_time)
        print(log)
    best_valid_acc = 0
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
            waveform = waveform.to(device)
            label = label.to(device)
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
            lr_iterarion.append(get_lr(optimizer))
            running_loss += loss.data.item()
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        valid_accuracy,_,valid_loss = calculate_accuracy(music_classify,valid_loader,device, criterion)
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| valid acc: {} | time: {}".format(epoch, running_loss, model_accuracy, valid_accuracy,epoch_time)
        train_acc_epoch.append(model_accuracy)
        vall_acc_epoch.append(valid_accuracy)
        print(log)
        scheduler.step(metrics=valid_accuracy)
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            path_save = os.path.join(path, "best_model")
            torch.save(music_classify.state_dict(), path_save)
    lr_iterarion = np.array(lr_iterarion)
    vall_acc_epoch = np.array(vall_acc_epoch)
    train_loss_iter = np.array(train_loss_iter)
    train_acc_epoch = np.array(train_acc_epoch)
    np.save(os.path.join(path,'lr_iteration'),lr_iterarion)
    np.save(os.path.join(path,'vall_acc_epoch'),vall_acc_epoch)
    np.save(os.path.join(path,'train_loss_iter'), train_loss_iter)
    np.save(os.path.join(path,'train_acc_epoch'),train_acc_epoch)
    if test_loader is not None:
        test_accuracy,_,_ =  calculate_accuracy(music_classify,test_loader,device,criterion=criterion)
        path_save = os.path.join(path, "last_model")
        torch.save(music_classify.state_dict(), path_save)
        return test_accuracy


def calculate_accuracy_2d(model, dataloader, device , criterion):
    model= model.to(device)
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


def calculate_accuracy_1d(model, dataloader, device , criterion):
    model= model.to(device)
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10, 10], int)
    runningn_loss = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images)
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    music_classify=resnet_dropout.resnet18(num_classes = 10)
    music_classify= music_classify.to(device)
    print("loading data...")
    train_loader,valid_loader,_,_ = datamanager_ver2.get_dataloader(hparams,dataset_size=hparams.optuna_ds)
    test_acc = train_for_optuna(trial=trial,music_classify=music_classify,train_loader=train_loader,valid_loader=valid_loader)
    return test_acc


def train_for_optuna(trial,music_classify, train_loader, valid_loader, test_loader=None):
    now = dt.now()
    dt_string = now.strftime("%d%m%Y%H%M")
    path = os.path.join(os.getcwd(),"trial_"+dt_string)
    os.mkdir(path)


    print("starting training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    warmup_factor = trial.suggest_float("warmup_factor", 1e1, 1e3, log=True)
    init_lr = lr/warmup_factor
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(torch.optim, optimizer_name)(music_classify.parameters(), lr=init_lr, weight_decay=hparams.weight_decay)
    sched_factor = trial.suggest_float("sched_factor",0,1)
    patience = trial.suggest_int("patience",1,6)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=sched_factor, patience=patience,
                                               verbose=True)
    #warmup section:
    warmup_epochs = trial.suggest_int("warmup_ephocs",2,5)
    num_steps = warmup_epochs*len(train_loader)

    warmuper = my_warmup.LinearWarmuper(optimizer=optimizer,steps=num_steps, factor = warmup_factor)
    print("warmup section...")
    for w_epoch in range(warmup_epochs+1):
        print("info: lr={}".format(get_lr(optimizer)))
        total_tracks = 0
        total_correct = 0
        epoch_time = time.time()
        for i, data in enumerate(train_loader):
            running_loss = 0.0
            waveform, label = data
            waveform = waveform.to(device)
            label = label.to(device)
            size = waveform.size()
            inputs = torch.zeros(size[0], 3, size[1], size[2])
            inputs[:, 0, :, :] = waveform
            inputs[:, 1, :, :] = waveform
            inputs[:, 2, :, :] = waveform
            inputs = inputs.to(device)
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
                                                                          model_accuracy,epoch_time)
        print(log)
    best_valid_acc = 0
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer, last_step=5)
    ephocs = trial.suggest_int("ephocs",10,30)
    print("starting train loop...")
    for epoch in range(1, ephocs + 1):
        print("info: lr={}".format(get_lr(optimizer)))
        music_classify.train()
        running_loss = 0.0
        epoch_time = time.time()
        total_tracks = 0
        total_correct = 0

        for i, data in enumerate(train_loader):
            waveform, label = data
            waveform = waveform.to(device)
            label = label.to(device)
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
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            path_save = os.path.join(path,"best_model")
            torch.save(music_classify.state_dict(),path_save)

        print(log)
        scheduler.step(metrics=valid_accuracy)
        # warmup_scheduler.dampen()
    path_save = os.path.join(path, "last_model")
    torch.save(music_classify.state_dict(), path_save)
    return valid_accuracy

def run_parameter_tuning():
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler("foo.log", mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    study = optuna.create_study(study_name= "music classifier", direction="maximize",sampler=optuna.samplers.TPESampler())
    logger.info("Start optimization.")
    study.optimize(objective, n_trials=hparams.number_of_trials)
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.plot_param_importances(study)
    joblib.dump(study, "study.pkl")

def test_ensemble(model,model_expert, test_ensemble_loader,ensamble_method='soft',model_size = None):
    if ensamble_method not in ['soft','hard']:
        print('wrong ensamble method')
        return
    model.eval()  # put in evaluation mode
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([model_size, model_size], int)
    runningn_loss = 0

    with torch.no_grad():
        for data in test_ensemble_loader:
            images, labels = data
            outputs = torch.zeros((1, model_size)).to(device)
            for i in range(hparams.number_of_chunks):
                images = images.to(device)
                size = images.size()
                inputs = torch.zeros(size[0], 3, size[2], size[3])
                inputs[:, 0, :, :] = images[:,i,:,:]
                inputs[:, 1, :, :] = images[:,i,:,:]
                inputs[:, 2, :, :] = images[:,i,:,:]
                inputs = inputs.to(device)
                labels = labels.to(device)
                out = model(inputs)
                if ensamble_method == 'soft':
                    outputs += out
                elif ensamble_method== 'hard':
                    _,mini_pred = torch.max(out,1)
                    outputs[0,mini_pred] += 1
            _, temppred = torch.max(outputs,1)
            if model_expert is not None and temppred in [8,9]:
                outputs = torch.zeros((1, 10)).to(device)
                for i in range(hparams.number_of_chunks):
                    images = images.to(device)
                    size = images.size()
                    inputs = torch.zeros(size[0], 3, size[2], size[3])
                    inputs[:, 0, :, :] = images[:, i, :, :]
                    inputs[:, 1, :, :] = images[:, i, :, :]
                    inputs[:, 2, :, :] = images[:, i, :, :]
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    out = model_expert(inputs)
                    if ensamble_method == 'soft':
                        outputs += out
                    elif ensamble_method == 'hard':
                        _, pred = torch.max(out, 1)
                        outputs[0,8 + pred] += 1
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            runningn_loss += loss
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1
    runningn_loss /= len(test_ensemble_loader)
    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix, runningn_loss


if __name__ == '__main__':
    #run_parameter_tuning()
    #audio_augmentation_joni.main_reduced(option="High")
    #feature_extraction.main()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #train_loader, valid_loader, test_loader,test_ensemble_loader =datamanager_ver2.get_dataloader(hparams,sub_genres=['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae',])
    #model8 = resnet_dropout.resnet18(num_classes = 8)
    #train_cnn_2d_pre_net(train_loader,valid_loader,test_loader,model8)
    #model10.load_state_dict(torch.load('/home/raveh.be@staff.technion.ac.il/trial_270620211046/best_model'))
    #test_acc,test_mat,_ = calculate_accuracy(model10,test_loader,device=device,criterion=criterion)
    #valid_ac, valid_mat,_ = calculate_accuracy(model10,valid_loader,device,criterion)
    #model_acc_en,mat_en,_ = test_ensemble(model10,None, test_ensemble_loader,'hard',model_size= 10)
    #print(model_acc_en)
    #train_cnn_2d_pre_net(train_loader,valid_loader,test_loader,model8)
    #model10 = resnet_dropout.resnet18(num_classes=10)
    #train_cnn_2d_pre_net(train_loader,valid_loader,test_loader,model10)
    #model2 = resnet_dropout.resnet18(num_classes=2)
    #train_extra, valid_extra = datamanager_ver2.get_extra_loaders(hparams)
    #train_cnn_2d_pre_net(train_extra, valid_extra, None, model2)
    #print(model2)
    #model10.load_state_dict(torch.load('/home/raveh.be@staff.technion.ac.il/trial_270620211046/best_model'))
    #model2.load_state_dict(torch.load('/home/raveh.be@staff.technion.ac.il/trial_250620211356/best_model'))
    #model.load_state_dict(torch.load(os.path.join('/home/raveh.be@staff.technion.ac.il/trial_240620211745/best_model')))
    #model_acc_en,mat_en,_ = test_ensemble(model10,model2, test_ensemble_loader,'hard')
    #model_acc, mat_acc, _ = calculate_accuracy(model10, test_loader, device, criterion)
    #print(model_acc)
    model1dver1 = model2.Music1DCNN_ver2()
    train_loader, valid_loader, test_loader = DataManager.get_dataloader(hparams)
    train_1d(train_loader,valid_loader,test_loader,model1dver1)
