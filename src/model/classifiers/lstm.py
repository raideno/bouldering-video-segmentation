import torch

from ._base import BaseClassifier

class LstmClassifier(BaseClassifier):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LstmClassifier, self).__init__()
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=2, end_dim=3)
            
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        
        return output