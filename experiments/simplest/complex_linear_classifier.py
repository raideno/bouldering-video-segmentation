import torch

from collections import OrderedDict

class ComplexLinearClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, dropout_rate=0.3):
        super(ComplexLinearClassifier, self).__init__()
        
        self.model = torch.nn.Sequential(OrderedDict([
            ("block1", torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_rate)
            )),
            ("block2", torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_rate)
            )),
            ("output", torch.nn.Linear(hidden_size, output_size))
        ]))
    
    def forward(self, x):
        return self.model(x)