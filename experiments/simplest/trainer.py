import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

class Trainer():
    def __init__(self, model):
        self.model = model
        
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
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            num_batches += 1
        
        accuracy = correct / total
        
        return total_loss / num_batches, accuracy
            
    def validate(self, testing_dataloader):
        self.model.eval()
        
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in testing_dataloader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = self.model.forward(features)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches, correct / total
    
    def train(self, training_dataloader, testing_dataloader, title=None):
        history = {
            "training_loss": [],
            "training_accuracy": [],
            "testing_loss": [],
            "testing_accuracy": []
        }
        
        best_validation_accuracy = 0
        best_training_accuracy = 0
        best_epoch = 0
        
        with tqdm.tqdm(iterable=range(32), desc=title or "[training]", unit="epoch") as progress_bar:
            for epoch in progress_bar:
                training_loss, training_accuracy = self.__train_one_epoch(training_dataloader, learning_rate=0.001)
                validation_loss, validation_accuracy = self.validate(testing_dataloader)
                
                # NOTE: store history
                history["training_loss"].append(training_loss)
                history["training_accuracy"].append(training_accuracy)
                history["testing_loss"].append(validation_loss)
                history["testing_accuracy"].append(validation_accuracy)
                
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_training_accuracy = training_accuracy
                    best_epoch = epoch
                    
                progress_bar.set_postfix({
                    "training-loss": training_loss,
                    "training-accuracy": training_accuracy,
                    "validation-loss": validation_loss,
                    "validation-accuracy": validation_accuracy,
                    "best-validation-accuracy": best_validation_accuracy,
                    "best-training-accuracy": best_training_accuracy
                })
                
        return history, best_training_accuracy, best_validation_accuracy, best_epoch