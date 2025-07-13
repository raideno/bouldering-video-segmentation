import torch
import pytorch_lightning

class Model(pytorch_lightning.LightningModule):
    def __init__(
        self,
        backbone,
        classifier,
        
        learning_rate: float = 0.001
    ):
        """
        Parameters:
        -----------
        input_size: The feature size for each time step.
        hidden_size: The number of hidden units in the LSTM.
        output_size: The number of classes.
        num_layers: The number of LSTM layers. Default is 1.
        dropout: The dropout probability. Default is 0.0.
        learning_rate: The learning rate for the optimizer. Default is 0.001.
        """
        super().__init__()

        self.backbone = backbone
        self.classifier = classifier
        
        self.learning_rate = learning_rate
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.class_correct = {}
        self.class_total = {}
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=2, end_dim=3)
            
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        
        return output
    
    def step(self, batch, batch_idx):
        pass
    
    def training_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        
        # Reshape outputs to [batch_size * seq_length, num_classes]
        batch_size, seq_length, num_classes = outputs.size()
        outputs_reshaped = outputs.reshape(-1, num_classes)
        
        # Get one-hot encoded labels and reshape
        labels_reshaped = labels.argmax(dim=2).reshape(-1)
        
        # Compute loss on reshaped tensors
        mask = (labels_reshaped != -1)  # Ignore padding (-1)
        if mask.sum() > 0:  # Only compute loss if we have valid labels
            loss = self.criterion(outputs_reshaped[mask], labels_reshaped[mask])
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 2)
        _, labels_max = torch.max(labels, 2)
        
        # Calculate accuracy ignoring padding
        valid_positions = (labels_max != -1)
        total = valid_positions.sum().item()
        correct = ((predicted == labels_max) & valid_positions).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)
        
        return {'loss': loss, 'predictions': predicted, 'targets': labels_max, 'accuracy': accuracy}
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        
        # Reshape outputs to [batch_size * seq_length, num_classes]
        batch_size, seq_length, num_classes = outputs.size()
        outputs_reshaped = outputs.reshape(-1, num_classes)
        
        # Get one-hot encoded labels and reshape
        labels_reshaped = labels.argmax(dim=2).reshape(-1)
        
        # Compute loss on reshaped tensors
        mask = (labels_reshaped != -1)  # Ignore padding (-1)
        if mask.sum() > 0:  # Only compute loss if we have valid labels
            loss = self.criterion(outputs_reshaped[mask], labels_reshaped[mask])
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 2)
        _, labels_max = torch.max(labels, 2)
        
        # Calculate accuracy ignoring padding
        valid_positions = (labels_max != -1)
        total = valid_positions.sum().item()
        correct = ((predicted == labels_max) & valid_positions).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        
        # Track per-class accuracy
        for b in range(batch_size):
            for s in range(seq_length):
                label = labels_max[b, s].item()
                pred = predicted[b, s].item()
                
                # Skip padding (-1)
                if label == -1:
                    continue
                    
                if label not in self.class_total:
                    self.class_total[label] = 0
                    self.class_correct[label] = 0
                
                self.class_total[label] += 1
                if pred == label:
                    self.class_correct[label] += 1
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        
        return {'loss': loss, 'predictions': predicted, 'targets': labels_max, 'accuracy': accuracy}
    
    def on_validation_epoch_end(self):
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for class_id in self.class_total:
            if self.class_total[class_id] > 0:
                per_class_accuracy[class_id] = self.class_correct[class_id] / self.class_total[class_id]
            else:
                per_class_accuracy[class_id] = 0.0
                
        # Reset counters for next epoch
        self.class_correct = {}
        self.class_total = {}
        
        # Log per-class accuracy
        for class_id, acc in per_class_accuracy.items():
            self.log(f'val_acc_class_{class_id}', acc)
            
        return per_class_accuracy