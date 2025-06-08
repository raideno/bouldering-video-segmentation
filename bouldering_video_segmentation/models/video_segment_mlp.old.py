import torch

class VideoSegmentMlp(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(out_features=output_size)
        )
        
    def forward(self, x):
        return self.network(x)