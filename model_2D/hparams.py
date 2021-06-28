#import argparse

class HParams(object):
    def __init__(self):
        self.dataset_path = './datasets/'
        self.feature_path= './datasets/feature_augment'
        self.genres_path = './datasets/genres'
        self.genres = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'blues']
        self.genres_expert = ['rock', 'blues']
        # Feature Parameters
        self.sample_rate=22050
        self.fft_size = 1024
        self.win_size = 1024
        self.hop_size = 512
        self.num_mels = 128
        self.feature_length = 1024
        self.maximum_length = 675_808
        self.minimum_legnth = 660_000
        self.number_of_trials = 20
        self.optuna_ds = 0.3
        self.warmup_epochs =4
        self.warmup_factor = 56.87766864314209
        self.aug_number = 8
        self.dropout = 0.2
        self.number_of_chunks = 8
        self.mean_low = 0.17226519
        self.std_low = 0.22243384
        self.mean_high = 0.18000145
        self.std_high = 0.2393243
        self.threshold = 5e-1
        # Training Parameters
        self.device = 1  # 0: CPU, 1: GPU0, 2: GPU1, ...
        self.batch_size = 4
        self.num_epochs = 27
        self.learning_rate = 0.04317577160870588
        self.stopping_rate = 1e-5
        self.weight_decay = 1e-5
        self.momentum = 0.9
        self.factor = 0.15276511276693686
        self.patience = 3

hparams = HParams()