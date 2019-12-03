import torch
from torchsummary import summary
from torch import nn 
class FC_Net(torch.nn.Module):

    def __init__(self, num_classes, input_shape=(32, 32, 32)):
        
        super(FC_Net, self).__init__()
        self.preprocess = torch.nn.Sequential(
            nn.Linear(100*100*100, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,2),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.preprocess(x)
        
        return x

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FC_Net(2).to(DEVICE)
    summary(model, (1, 100, 100, 100))
