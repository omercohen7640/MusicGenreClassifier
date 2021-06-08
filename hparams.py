#import argparse

class HParams(object):
    def __init__(self):
        self.dataset_path = './datasets/'
        #self.feature_path= './dataset/feature_augment'
        self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'blues']

        # Feature Parameters
        #self.sample_rate=22050
        #self.fft_size = 1024
        #self.win_size = 1024
        #self.hop_size = 512
        #self.num_mels = 128
        #self.feature_length
        self.maximum_length = 675_808
        self.minimum_legnth = 660_000

        # Training Parameters
        self.device = 1  # 0: CPU, 1: GPU0, 2: GPU1, ...
        self.batch_size = 8
        self.num_epochs = 5
        self.learning_rate = 1e-2
        self.stopping_rate = 1e-5
        self.weight_decay = 1e-6
        self.momentum = 0.9
        self.factor = 0.2
        self.patience = 5
