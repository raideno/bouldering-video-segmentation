import torch
import pytorch_lightning as pl

class VideoSegmentMlp(pl.LightningModule):
    def __init__(self, output_size: int, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(out_features=output_size)
        )
        
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.class_correct = {}
        self.class_total = {}
        
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)
        
        # Return for optional score metrics
        return {'loss': loss, 'predictions': predicted, 'targets': labels, 'accuracy': accuracy}
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        # Track per-class accuracy
        for i in range(len(labels)):
            label = labels[i].item()
            pred = predicted[i].item()
            
            if label not in self.class_total:
                self.class_total[label] = 0
                self.class_correct[label] = 0
            
            self.class_total[label] += 1
            if pred == label:
                self.class_correct[label] += 1
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        
        return {'loss': loss, 'predictions': predicted, 'targets': labels, 'accuracy': accuracy}
    
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
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)