import torch
from torchsummary import summary
from torch import nn


class VoxNet(torch.nn.Module):

    def __init__(self, num_classes, input_shape=(32, 32, 32)):
                 # weights_path=None,
                 # load_body_weights=True,
                 # load_head_weights=True):
        """
        VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition.
        Modified in order to accept different input shapes.
        Parameters
        ----------
        num_classes: int, optional
            Default: 10
        input_shape: (x, y, z) tuple, optional
            Default: (32, 32, 32)
        weights_path: str or None, optional
            Default: None
        load_body_weights: bool, optional
            Default: True
        load_head_weights: bool, optional
            Default: True
        Notes
        -----
        Weights available at: url to be added
        If you want to finetune with custom classes, set load_head_weights to False.
        Default head weights are pretrained with ModelNet10.
        """
        super(VoxNet, self).__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1,
                            out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(),
            # torch.nn.Dropout(p=0.2),
            torch.nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(64, 32, 3, 2),
            nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(),
            # torch.nn.Dropout(p=0.3)
        )

        # Trick to accept different input shapes
        x = self.body(torch.autograd.Variable(
            torch.rand((16, 1) + input_shape)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n

        self.head = torch.nn.Sequential(
            torch.nn.Linear(first_fc_in_features, 128),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, num_classes)
        )

        self.initializetion()

    def initializetion(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, [32,32,32], mode='trilinear')
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VoxNet(2).to(DEVICE)
    summary(model, (1, 100, 100, 100))
