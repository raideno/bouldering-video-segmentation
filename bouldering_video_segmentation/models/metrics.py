class CustomMetricCallback(pl.Callback):
    """
    Callback to compute custom metrics during training or validation.
    """
    def __init__(self, scores: Dict[str, Callable[[List[int], List[int]], float]]={}):
        """
        Parameters:
        -----------
        scores: A dictionary of score functions to compute.
        """
        super().__init__()
        self.scores = scores
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.scores:
            return
            
        # Compute scores
        predictions = outputs['predictions']
        targets = outputs['targets']
        
        for name, score_fn in self.scores.items():
            try:
                score_value = score_fn(predictions, targets)
                trainer.logger.experiment.add_scalar(f'train_{name}', score_value, trainer.global_step)
            except Exception as e:
                print(f"Error calculating {name}: {e}")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.scores:
            return
            
        # Compute scores
        predictions = outputs['predictions']
        targets = outputs['targets']
        
        for name, score_fn in self.scores.items():
            try:
                score_value = score_fn(predictions, targets)
                trainer.logger.experiment.add_scalar(f'val_{name}', score_value, trainer.global_step)
            except Exception as e:
                print(f"Error calculating {name}: {e}")