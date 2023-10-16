import torch.nn as nn

class PyTorchNN(nn.Module):

    # constructor
    def __init__(self):
        """
        Assigning Linear Layers to class members variables
        """
        super(PyTorchNN, self).__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=30).double()
        self.layer2 = nn.Linear(in_features=30, out_features=1, bias=False).double()

    # predictor
    def forward(self, x):
        """
        Append Layers
        """
        y = nn.Sequential(self.layer1,
                          nn.Sigmoid().double(),
                          self.layer2)(x)

        return y