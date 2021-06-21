import torch.nn as nn

#input_size= 661500
class Music1DCNN_ver2(nn.Module):
    """Network Builder"""
    def __init__(self):
        super(Music1DCNN_ver2,self).__init__()
        self.conv_layer = nn.Sequential(
            #use kernrel size of 1
            #batchnorm after RELU
            #put this in forward:


            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),

            #.Conv1d(in_channels=16, out_channels=16, kernel_size=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(16),

            nn.Conv1d(in_channels=16,out_channels=32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),

            #nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(32),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),

            #nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(64),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),

            #nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(128),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            #nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(256),

            #nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(256),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            #nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(512),

            #nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(512),

            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),

            #nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(1024),

        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10),
        )
    def forward(self, x):
        """Perform forward"""

        #conv layer
        x = self.conv_layer(x)

        #flatten
        #x = x.view(x.size(0),-1)
        # output = self.conv(x)
        x = x.max(dim=2)[0]

        #fc layer
        x = self.fc_layer(x)

        return x

