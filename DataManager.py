import torchaudio
import torchaudio.transforms as transforms
import os
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import hparams
import torch.nn.functional as F
import random


class GTZANDataset(Dataset):
    def __init__(self,torch_dataset, labels_list,vector_equlizer='padding', output_length=675808):
        x = []
        y = []
        for item in torch_dataset:
            waveform, sr, label = item
            if vector_equlizer =='padding':
                waveform = waveform
                pad = output_length - waveform.size(1)
                x.append(F.pad(input=waveform, pad=(0, pad, 0, 0), mode='constant', value=0))
                y.append(labels_list.index(label))
            elif vector_equlizer == 'cut min':
                x.append(waveform[:,:output_length])
                y.append(labels_list.index(label))
            elif vector_equlizer == 'k sec':
                k = 1
                sec_k = sr * k
                resize_data_factor = 1
                r_list = createRandomSortedList(resize_data_factor, (0 * sr), (26 * sr))
                for i in r_list:
                    y.append(labels_list.index(label))
                    if i < 0:
                        pad = i * (-1)
                        x.append(F.pad(input=waveform[:, :i + sec_k], pad=(pad, 0, 0, 0), mode='constant', value=0))
                    elif i > sr * 25:
                        pad = i + sec_k - waveform.size(1)
                        x.append(F.pad(input=waveform[:, i:], pad=(0, pad, 0, 0), mode='constant', value=0))
                    else:
                        x.append(waveform[:, i:i + sec_k])
        self.x = torch.stack(x)
        self.y = torch.tensor(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def createRandomSortedList(num, start=1, end=100):
    arr = []
    tmp = random.randint(start, end)

    for x in range(num):

        while tmp in arr:
            tmp = random.randint(start, end)

        arr.append(tmp)

    arr.sort()

    return arr



def get_dataloader(hparams):
    trainset,validset,testset = load_gtza_from_torch()

    trainset = GTZANDataset(torch_dataset=trainset, labels_list=hparams.genres,vector_equlizer='k sec')
    validset = GTZANDataset(torch_dataset=validset, labels_list=hparams.genres,vector_equlizer='k sec')
    testset = GTZANDataset(torch_dataset=testset, labels_list=hparams.genres,vector_equlizer='k sec')

    train_loader = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(validset, batch_size=hparams.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(testset, batch_size=hparams.batch_size, shuffle=True, drop_last=False)

    return train_loader, valid_loader, test_loader

def load_gtza_from_torch():

    trainset = torchaudio.datasets.GTZAN(root="./datasets", download=True,subset="training")
    validset = torchaudio.datasets.GTZAN(root="./datasets", download=True, subset="validation")
    testset = torchaudio.datasets.GTZAN(root="./datasets", download=True, subset="testing")
    return trainset, validset, testset

if __name__ == '__main__':
    load_gtza_from_torch()