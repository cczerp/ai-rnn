"""
Trainer Category
Orchestrates the training loop and learning process.
"""

import numpy as np
from typing import Dict, List, Callable
from ..data.loader import DataLoader
from .loss import LossCalculator
from .optimizer import OptimizerManager


class Trainer:
    """
    Manages the training process for RNN.
    
    Categories of training:
    - Epoch iteration: Loop through dataset multiple times
    - Batch processing: Train on mini-batches
    - Progress tracking: Monitor training metrics
    - Early stopping: Prevent overfitting
    - Checkpointing: Save best models
    """
    
    def __init__(self, model, 
                 loss_calculator: LossCalculator,
                 optimizer: OptimizerManager,
                 data_loader: DataLoader):
        """
        Initialize trainer.
        
        Args:
            model: RNN model to train
            loss_calculator: Loss calculator instance
            optimizer: Optimizer instance
            data_loader: Data loader instance
        """
        self.model = model
        self.loss_calculator = loss_calculator
        self.optimizer = optimizer
        self.data_loader = data_loader
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_params = None
        self.patience_counter = 0
        
    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Train for one epoch.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            
        Returns:
            Average training loss
        """
        epoch_losses = []
        
        # Iterate through batches
        for batch_X, batch_y in self.data_loader.create_batches(X_train, y_train):
            # Forward pass
            predictions = self.model.predict(batch_X)
            
            # Compute loss
            loss = self.loss_calculator.compute_loss(
                predictions, batch_y, 
                self.model.get_parameters()
            )
            epoch_losses.append(loss)
            
            # Backward pass (simplified - actual implementation would do backprop)
            gradient = self.loss_calculator.compute_gradient(predictions, batch_y)
            
            # Update parameters
            gradients = self.model.compute_gradients(batch_X, gradient)
            updated_params = self.optimizer.update_parameters(
                self.model.get_parameters(), 
                gradients
            )
            self.model.set_parameters(updated_params)
        
        return np.mean(epoch_losses)
    
    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            X_val: Validation input data
            y_val: Validation target data
            
        Returns:
            Dictionary of validation metrics
        """
        val_losses = []
        
        for batch_X, batch_y in self.data_loader.create_batches(X_val, y_val):
            predictions = self.model.predict(batch_X)
            loss = self.loss_calculator.compute_loss(predictions, batch_y)
            val_losses.append(loss)
        
        metrics = self.loss_calculator.get_metrics(predictions, batch_y)
        metrics['loss'] = np.mean(val_losses)
        
        return metrics
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100,
              early_stopping_patience: int = 10,
              verbose: int = 1) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data (optional)
            y_val: Validation target data (optional)
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level (0: silent, 1: progress bar, 2: detailed)
            
        Returns:
            Training history dictionary
        """
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self.train_epoch(X_train, y_train)
            self.history['train_loss'].append(train_loss)
            
            # Validate if validation data provided
            if X_val is not None and y_val is not None:
                val_metrics = self.validate(X_val, y_val)
                val_loss = val_metrics['loss']
                self.history['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_params = self.model.get_parameters()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if verbose >= 1:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {train_loss:.4f} - "
                          f"val_loss: {val_loss:.4f}")
                
                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    if verbose >= 1:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose >= 1:
                    print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}")
        
        # Restore best parameters
        if self.best_params is not None:
            self.model.set_parameters(self.best_params)
        
        return self.history
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.history
    
    def reset_history(self):
        """Reset training history."""
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        self.best_loss = float('inf')
        self.best_params = None
        self.patience_counter = 0
