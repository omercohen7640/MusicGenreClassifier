import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob

class GTZANDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

# Function to get genre index for the give file
def get_label(file_name, hparams):
    genre = file_name.split('.')[0]
    label = hparams.genres.index(genre)
    return label

def load_dataset(set_name, hparams):
    x = []
    y = []

    dataset_path = os.path.join(hparams.feature_path, set_name)
    for root,dirs,files in os.walk(dataset_path):
        for file in files:
            data = np.load(os.path.join(root,file))
            label = get_label(file, hparams)
            x.append(data)
            y.append(label)

    x = np.stack(x)
    y = np.stack(y)

    return x,y
def load_ensemble_test_set(hparams):
    x = []
    y = []
    test_list_path = os.path.join(hparams.dataset_path,'test_list.txt')
    with open(test_list_path) as f:
        lines = f.readlines()
        for line in lines:
            track_name = line.split('/')[1].split('.')[1]
            genre = line.split('/')[0]
            path = os.path.join(hparams.feature_path,'test',genre,'*'+str(track_name)+'*')
            label = get_label(line.split('/')[1], hparams)
            y.append(label)
            x_chunks = []
            for file in glob.glob(path):
                data = np.load(file)
                x_chunks.append(data)
            x_chunks = np.stack(x_chunks)
            x.append(x_chunks)
    x = [i[:8,:,:] for i in x]
    x = np.array(x)
    y = np.stack(y)

    return x,y
def get_dataloader(hparams):
    x_train, y_train = load_dataset('train_aug', hparams)
    x_valid, y_valid = load_dataset('valid', hparams)
    x_test, y_test = load_dataset('test', hparams)
    x_test_ensemble, y_test_ensemble = load_ensemble_test_set(hparams)
    mean = np.mean(x_train)
    std = np.std(x_train)

    x_train = (x_train - mean)/std
    x_valid = (x_valid - mean)/std
    x_test = (x_test - mean)/std
    x_test_ensemble =(x_test_ensemble - mean)/std

    train_set = GTZANDataset(x_train, y_train)
    vaild_set = GTZANDataset(x_valid, y_valid)
    test_set = GTZANDataset(x_test, y_test)
    test_ensemble_set = GTZANDataset(x_test_ensemble,y_test_ensemble)

    train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(vaild_set, batch_size=hparams.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=hparams.batch_size, shuffle=False, drop_last=False)
    test_ensemble_loader = DataLoader(test_ensemble_set, batch_size=1, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader, test_ensemble_loader