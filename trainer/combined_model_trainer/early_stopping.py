import os
import torch

class EarlyStopping:
    """
    Early stops the training if monitored metric doesn't improve.
    
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped
        verbose (bool): If True, prints a message for each improvement
        delta (float): Minimum change in the monitored quantity to qualify as an improvement
        path (str): Path to save the checkpoint
        monitor (str): Quantity to monitor ('loss', 'f1', 'recall', etc.)
        mode (str): 'min' for loss-like metrics, 'max' for performance metrics
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', 
                 trace_func=print, monitor='loss', mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.monitor = monitor
        self.mode = mode
        
        # Initialize best_val based on mode
        self.best_val = float('inf') if mode == 'min' else -float('inf')

    def __call__(self, value, model, save_path):
        """
        Check if training should stop based on the monitored metric.
        
        Args:
            value (float): Current value of the metric
            model (torch.nn.Module): Model to save
            save_path (str): Path to save the model checkpoint
        """
        # Determine if improvement based on mode
        if self.mode == 'min':
            is_better = value < self.best_val - self.delta
        else:  # 'max' mode
            is_better = value > self.best_val + self.delta
        
        if self.best_score is None or is_better:
            # Better score found
            self.best_score = value
            self.best_val = value
            self.save_checkpoint(value, model, save_path)
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, value, model, save_path):
        """
        Save model when monitored metric improves.
        
        Args:
            value (float): Metric value
            model (torch.nn.Module): Model to save
            save_path (str): Path to save the checkpoint
        """
        # Determine metric name and improvement direction
        metric_name = {
            'f1': 'F1 score', 
            'recall': 'Recall', 
            'loss': 'Validation loss'
        }.get(self.monitor, self.monitor)
        
        direction = "increased" if self.mode == 'max' else "decreased"
        
        if self.verbose:
            self.trace_func(f'{metric_name} {direction} ({self.best_val:.6f} --> {value:.6f}). Saving model...')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            self.monitor: value
        }, save_path)