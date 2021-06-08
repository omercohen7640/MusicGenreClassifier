import torch.nn as nn

#input_size= 661500
class Music1DCNN_ver1(nn.Module):
    """Network Builder"""
    def __init__(self):
        super(Music1DCNN_ver1,self).__init__()
        self.conv_layer = nn.Sequential(
            #use kernrel size of 1
            #batchnorm after RELU
            #put this in forward:
            #output = self.conv(x)
            #output2 = output.max(dim=2)[0]

            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=128,stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=128, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2,stride=2),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=64, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=64, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16,out_channels=32, kernel_size=32, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=32, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3 ,stride=1, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(10240, 5120),
            nn.ReLU(inplace=True),
            nn.Linear(5120, 2560),
            nn.ReLU(inplace=True),
            nn.Linear(2560, 1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 320),
            nn.ReLU(inplace=True),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True),
            nn.Linear(160, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, 40),
            nn.ReLU(inplace=True),
            nn.Linear(40, 10),
        )
    def forward(self, x):
        """Perform forward"""

        #conv layer
        x = self.conv_layer(x)

        #flatten
        x = x.view(x.size(0),-1)

        #fc layer
        x = self.fc_layer(x)

        return x

