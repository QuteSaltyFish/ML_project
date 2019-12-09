import torch
from torchsummary import summary
from torch import nn

# change the input to 100*100*100 + 32*32*32 + 8*8*8
# priimid triaining


class Vox_100(torch.nn.Module):

    def __init__(self, num_classes, input_shape=(32, 32, 32)):
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
                            out_channels=32, kernel_size=3, stride=1),
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

        # Trick to accept different input shapes
        x = self.body(torch.autograd.Variable(
            torch.rand((1, 1) + input_shape)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n

        self.head = torch.nn.Sequential(
            torch.nn.Linear(first_fc_in_features, 128),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.4),
            torch.nn.Linear(128, num_classes),
        )

        self.initializetion()

    def initializetion(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, [32, 32, 32], mode='trilinear')
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class Vox_32(torch.nn.Module):

    def __init__(self, num_classes, input_shape=(32, 32, 32)):

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

        # Trick to accept different input shapes
        x = self.body(torch.autograd.Variable(
            torch.rand((1, 1) + input_shape)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n

        self.head = torch.nn.Sequential(
            torch.nn.Linear(first_fc_in_features, 128),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.4),
            torch.nn.Linear(128, num_classes),
        )

        self.initializetion()

    def initializetion(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, [32, 32, 32], mode='trilinear')
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Vox_32(2).to(DEVICE)
    summary(model, (1, 100, 100, 100))
