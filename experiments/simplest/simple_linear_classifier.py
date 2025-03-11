import torch

class SimpleLinearClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearClassifier, self).__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            # torch.nn.Linear(input_size, output_size)
            torch.nn.LazyLinear(out_features=output_size)
        )
        
    def forward(self, x):
        return self.network(x)