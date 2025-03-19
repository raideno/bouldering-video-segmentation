import tqdm
import torch

from enum import StrEnum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainerVariant(StrEnum):
    MLP = "MLP"
    LSTM = "LSTM"

class Trainer():
    def __init__(self, model, variant):
        self.model = model
        self.variant = variant
        
        if self.variant not in TrainerVariant:
            raise ValueError("Invalid variant.")
        
    def __train_one_epoch(self, training_dataloader, learning_rate):
        self.model.train()
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        for features, labels in training_dataloader:
            optimizer.zero_grad()
            
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = self.model.forward(features)

            # loss = criterion(outputs, labels)
            # loss.backward()
            
            if self.variant == TrainerVariant.MLP:
                loss = criterion(outputs, labels)
            elif self.variant == TrainerVariant.LSTM:
                # Reshape outputs to [batch_size * seq_length, num_classes]
                batch_size, seq_length, num_classes = outputs.size()
                outputs_reshaped = outputs.reshape(-1, num_classes)
                
                # Reshape labels to [batch_size * seq_length]
                labels_reshaped = labels.argmax(dim=2).reshape(-1)
                
                # Compute loss on reshaped tensors
                mask = (labels_reshaped != -1)  # Ignore padding (-1)
                if mask.sum() > 0:  # Only compute loss if we have valid labels
                    loss = criterion(outputs_reshaped[mask], labels_reshaped[mask])
                else:
                    loss = torch.tensor(0.0, device=device)
            loss.backward()
            
            optimizer.step()

            if self.variant == TrainerVariant.MLP:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            elif self.variant == TrainerVariant.LSTM:
                _, predicted = torch.max(outputs.data, 2)
                _, labels = torch.max(labels, 2)
                # Calculate accuracy considering the temporal dimension
                total += labels.size(0) * labels.size(1)  # batch_size * sequence_length
                correct += (predicted == labels).sum().item()
                
            total_loss += loss.item()
            num_batches += 1
        
        accuracy = correct / total
        
        return total_loss / num_batches, accuracy
            
    def validate(self, validation_loader):
        self.model.eval()
        
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in validation_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = self.model.forward(features)
                # loss = criterion(outputs, labels)
                
                if self.variant == TrainerVariant.MLP:
                    loss = criterion(outputs, labels)
                elif self.variant == TrainerVariant.LSTM:
                    # Reshape outputs to [batch_size * seq_length, num_classes]
                    batch_size, seq_length, num_classes = outputs.size()
                    outputs_reshaped = outputs.reshape(-1, num_classes)
                    
                    # Reshape labels to [batch_size * seq_length]
                    labels_reshaped = labels.argmax(dim=2).reshape(-1)
                    
                    # Compute loss on reshaped tensors
                    mask = (labels_reshaped != -1)  # Ignore padding (-1)
                    if mask.sum() > 0:  # Only compute loss if we have valid labels
                        loss = criterion(outputs_reshaped[mask], labels_reshaped[mask])
                    else:
                        loss = torch.tensor(0.0, device=device)
                
                if self.variant == TrainerVariant.MLP:
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                elif self.variant == TrainerVariant.LSTM:
                    _, predicted = torch.max(outputs.data, 2)
                    _, labels = torch.max(labels, 2)
                    # Calculate accuracy considering the temporal dimension
                    total += labels.size(0) * labels.size(1)  # batch_size * sequence_length
                    correct += (predicted == labels).sum().item()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches, correct / total
    
    def train(self, training_dataloader, validation_loader, title=None):
        history = {
            "training_loss": [],
            "training_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": []
        }
        best_validation_accuracy = 0
        best_training_accuracy = 0
        best_training_loss = float('inf')
        best_validation_loss = float('inf')
        best_epoch = 0
        
        best_model_state_dict = None
        
        with tqdm.tqdm(iterable=range(32), desc=title or "[training]", unit="epoch") as progress_bar:
            for epoch in progress_bar:
                training_loss, training_accuracy = self.__train_one_epoch(training_dataloader, learning_rate=0.001)
                validation_loss, validation_accuracy = self.validate(validation_loader)
                
                # NOTE: store history
                history["training_loss"].append(training_loss)
                history["training_accuracy"].append(training_accuracy)
                history["validation_loss"].append(validation_loss)
                history["validation_accuracy"].append(validation_accuracy)
                
                # if validation_accuracy > best_validation_accuracy:
                #     best_validation_accuracy = validation_accuracy
                #     best_training_accuracy = training_accuracy
                #     best_epoch = epoch
                
                # NOTE: update best losses and accuracies only when validation loss improves
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_training_loss = training_loss
                    
                    best_validation_accuracy = validation_accuracy
                    best_training_accuracy = training_accuracy
                    
                    best_epoch = epoch
                    
                    best_model_state_dict = self.model.state_dict()

                    
                progress_bar.set_postfix({
                    "training-loss": training_loss,
                    "training-accuracy": training_accuracy,
                    "validation-loss": validation_loss,
                    "validation-accuracy": validation_accuracy,
                    "best-validation-accuracy": best_validation_accuracy,
                    "best-training-accuracy": best_training_accuracy
                })
        
        return {
            "history": history,
            "best_training_accuracy": best_training_accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            "best_epoch": best_epoch,
            "best_training_loss": best_training_loss,
            "best_validation_loss": best_validation_loss,
            "best_model_state_dict": best_model_state_dict
        }