import torch

from ._base import BaseClassifier

class MlpClassifier(BaseClassifier):
    def __init__(self, input_size, output_size):
        super(MlpClassifier, self).__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(out_features=output_size)
        )
        
    def forward(self, x):
        return self.network(x)