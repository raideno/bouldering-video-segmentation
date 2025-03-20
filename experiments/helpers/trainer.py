import tqdm
import torch

from enum import StrEnum
from typing import Callable, Dict, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainerVariant(StrEnum):
    MLP = "MLP"
    LSTM = "LSTM"

class Trainer():
    def __init__(self, model, variant, scores:Dict[str, Callable[[List[int], List[int]], float]]={}):
        self.model = model
        self.scores = scores
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
        
        total_scores_evaluations = {name: 0.0 for name in self.scores.keys()}
        
        for features, labels in training_dataloader:
            optimizer.zero_grad()
            
            features = features.to(device)
            labels = labels.to(device)
            
            # TODO: make sure that if we pass in batch, the model will consider each batch separately
            outputs = self.model.forward(features)
            
            if self.variant == TrainerVariant.MLP:
                loss = criterion(outputs, labels)
            elif self.variant == TrainerVariant.LSTM:
                # NOTE: reshape outputs to [batch_size * seq_length, num_classes]
                batch_size, seq_length, num_classes = outputs.size()
                outputs_reshaped = outputs.reshape(-1, num_classes)
                
                # NOTE: reshape labels to [batch_size * seq_length]
                labels_reshaped = labels.argmax(dim=2).reshape(-1)
                
                # NOTE: compute loss on reshaped tensors
                mask = (labels_reshaped != -1)  # Ignore padding (-1)
                if mask.sum() > 0:  # NOTE: only compute loss if we have valid labels
                    loss = criterion(outputs_reshaped[mask], labels_reshaped[mask])
                else:
                    loss = torch.tensor(0.0, device=device)
            loss.backward()
            
            optimizer.step()

            if self.variant == TrainerVariant.MLP:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                total_scores_evaluations = {name: total_scores_evaluations[name] + score(predicted, labels) for name, score in self.scores.items()}
                
            elif self.variant == TrainerVariant.LSTM:
                _, predicted = torch.max(outputs.data, 2)
                _, labels = torch.max(labels, 2)
                # NOTE: calculate the accuracy considering the temporal dimension
                total += labels.size(0) * labels.size(1)  # batch_size * sequence_length
                correct += (predicted == labels).sum().item()
                
                total_scores_evaluations = {name: total_scores_evaluations[name] + score(predicted, labels) for name, score in self.scores.items()}
                
            total_loss += loss.item()
            num_batches += 1
        
        accuracy = correct / total
        
        scores_evaluations = {name: total_scores_evaluations[name] / num_batches for name in self.scores.keys()}
        
        return total_loss / num_batches, accuracy, scores_evaluations
            
    def validate(self, validation_loader):
        self.model.eval()
        
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        total_scores_evaluations = {name: 0.0 for name in self.scores.keys()}
        
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
                    
                    total_scores_evaluations = {name: total_scores_evaluations[name] + score(predicted, labels) for name, score in self.scores.items()}
                elif self.variant == TrainerVariant.LSTM:
                    _, predicted = torch.max(outputs.data, 2)
                    _, labels = torch.max(labels, 2)
                    # Calculate accuracy considering the temporal dimension
                    total += labels.size(0) * labels.size(1)  # batch_size * sequence_length
                    correct += (predicted == labels).sum().item()
                    
                    total_scores_evaluations = {name: total_scores_evaluations[name] + score(predicted, labels) for name, score in self.scores.items()}
                
                total_loss += loss.item()
                num_batches += 1
                
        scores_evaluations = {name: total_scores_evaluations[name] / num_batches for name in self.scores.keys()}
        
        return total_loss / num_batches, correct / total, scores_evaluations
    
    def train(self, training_dataloader, validation_loader, title=None):
        history = {
            "training_loss": [],
            "training_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "training_scores": [],
            "validation_scores": []
        }
        best_training_accuracy = 0
        best_training_loss = float('inf')
        best_training_scores = None
        
        best_validation_accuracy = 0
        best_validation_loss = float('inf')
        best_validation_scores = None
        
        best_epoch = 0
        
        best_model_state_dict = None
        
        with tqdm.tqdm(iterable=range(32), desc=title or "[training]", unit="epoch") as progress_bar:
            for epoch in progress_bar:
                training_loss, training_accuracy, training_scores = self.__train_one_epoch(training_dataloader, learning_rate=0.001)
                validation_loss, validation_accuracy, validation_scores = self.validate(validation_loader)
                
                # NOTE: store history
                history["training_loss"].append(training_loss)
                history["training_accuracy"].append(training_accuracy)
                history["training_scores"].append(training_scores)
                
                history["validation_loss"].append(validation_loss)
                history["validation_accuracy"].append(validation_accuracy)
                history["validation_scores"].append(validation_scores)
                
                # NOTE: update best losses and accuracies only when validation loss improves
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_training_loss = training_loss
                    
                    best_validation_accuracy = validation_accuracy
                    best_training_accuracy = training_accuracy
                    
                    best_training_scores = training_scores
                    best_validation_scores = validation_scores
                    
                    best_epoch = epoch
                    
                    best_model_state_dict = self.model.state_dict()

                progress_bar.set_postfix({
                    "training-loss": training_loss,
                    "training-accuracy": training_accuracy,
                    **{f"training-{name}": score for name, score in training_scores.items()},
                    # --- --- ---
                    "validation-loss": validation_loss,
                    "validation-accuracy": validation_accuracy,
                    **{f"validation-{name}": score for name, score in validation_scores.items()},
                    # --- --- ---
                    "best-validation-accuracy": best_validation_accuracy,
                    "best-training-accuracy": best_training_accuracy,
                    **{f"best-validation-{name}": score for name, score in best_validation_scores.items()},
                    **{f"best-training-{name}": score for name, score in best_training_scores.items()},
                })
        
        return {
            "history": history,
            # --- --- ---
            "best_epoch": best_epoch,
            # --- --- ---
            "best_training_accuracy": best_training_accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            # --- --- ---
            "best_training_loss": best_training_loss,
            "best_validation_loss": best_validation_loss,
            # --- --- ---
            "best_training_scores": best_training_scores,
            "best_validation_scores": best_validation_scores,
            # --- --- ---
            "best_model_state_dict": best_model_state_dict
        }