import torch
from torchsummary import summary
from torch import nn

# change the input to 100*100*100 + 32*32*32 + 8*8*8
# priimid triaining

class Vox_V3(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vox100 = Vox_100()
        self.vox32 = Vox_32()
        self.vox12 = Vox_12()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(16*3*6*6*6, 128),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.4),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x1 = self.vox100(x)
        x2 = self.vox32(x)
        x3 = self.vox12(x)
        x = torch.cat([x1,x2,x3], dim=1)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x

class Vox_12(torch.nn.Module):

    def __init__(self, num_classes=2, input_shape=(12, 12, 12)):

        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1,
                            out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(),
            # torch.nn.Dropout(p=0.2),
            torch.nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool3d(2),
            # torch.nn.Dropout(p=0.3)
            torch.nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3),
            nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool3d(2),
        )
        self.initializetion()

    def initializetion(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, [12, 12, 12], mode='trilinear')
        x = self.body(x)
        return x


class Vox_100(torch.nn.Module):
    def __init__(self, num_classes=2, input_shape=(48, 48, 48)):
                 # weights_path=None,
                 # load_body_weights=True,
                 # load_head_weights=True)
        super().__init__()
        self.preprocess = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1,
                            out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(),
            # torch.nn.Dropout(p=0.2),
            torch.nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(2),
        )
        self.body = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=32,
                            out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(),
            # torch.nn.Dropout(p=0.2),
            torch.nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(2),
            # torch.nn.Dropout(p=0.3)
            torch.nn.Conv3d(in_channels=32, out_channels=16, kernel_size=5),
            nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(2),

            torch.nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(),
        )
        self.initializetion()

    def initializetion(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, [100, 100, 100], mode='trilinear')
        x = self.preprocess(x)
        x = self.body(x)
        return x


class Vox_32(torch.nn.Module):

    def __init__(self, num_classes=2, input_shape=(32, 32, 32)):

        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1,
                            out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(),
            # torch.nn.Dropout(p=0.2),
            torch.nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3),
            nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(2),
            # torch.nn.Dropout(p=0.3)
            torch.nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3),
            nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(2),
        )

        self.initializetion()

    def initializetion(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, [32, 32, 32], mode='trilinear')
        x = self.body(x)
        return x


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Vox_V3(2).to(DEVICE)
    summary(model, (1, 100, 100, 100))
