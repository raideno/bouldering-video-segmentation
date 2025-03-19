import torch

class FullVideoLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        """
        Parameters:
        -----------
        input_size: The feature size for each time step.
        hidden_size: The number of hidden units in the LSTM.
        output_size: The number of classes.
        num_layers: The number of LSTM layers. Default is 1.
        dropout: The dropout probability. Default is 0.0.
        """
        super().__init__()
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=2, end_dim=3)
            
        lstm_out, (final_hidden_states, final_cell_states) = self.lstm(x, None)
        
        output = self.fc(lstm_out)
        
        return output