"""
Loss Calculation Category
Defines how the network measures its performance.
"""

import numpy as np
from typing import Tuple


class LossCalculator:
    """
    Computes loss and gradients for RNN training.
    
    Categories of loss computation:
    - Forward loss: Calculate error between predictions and targets
    - Gradient computation: Calculate derivatives for backpropagation
    - Loss types: Support different loss functions
    - Regularization: Add penalties to prevent overfitting
    """
    
    def __init__(self, loss_type: str = 'mse', l2_reg: float = 0.0):
        """
        Initialize loss calculator.
        
        Args:
            loss_type: Type of loss ('mse', 'cross_entropy')
            l2_reg: L2 regularization strength
        """
        self.loss_type = loss_type
        self.l2_reg = l2_reg
        
    def compute_loss(self, predictions: np.ndarray, 
                    targets: np.ndarray, 
                    parameters: dict = None) -> float:
        """
        Compute loss value.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            parameters: Model parameters for regularization
            
        Returns:
            Loss value
        """
        if self.loss_type == 'mse':
            loss = self.mse_loss(predictions, targets)
        elif self.loss_type == 'cross_entropy':
            loss = self.cross_entropy_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Add L2 regularization
        if self.l2_reg > 0 and parameters is not None:
            reg_loss = 0
            for param in parameters.values():
                if isinstance(param, np.ndarray):
                    reg_loss += np.sum(param ** 2)
            loss += 0.5 * self.l2_reg * reg_loss
        
        return loss
    
    def mse_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Mean Squared Error loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            MSE loss value
        """
        return np.mean((predictions - targets) ** 2)
    
    def cross_entropy_loss(self, predictions: np.ndarray, 
                          targets: np.ndarray) -> float:
        """
        Cross-entropy loss.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth targets (one-hot or indices)
            
        Returns:
            Cross-entropy loss value
        """
        # Numerical stability
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        # If targets are one-hot encoded
        if targets.ndim == predictions.ndim:
            return -np.mean(targets * np.log(predictions))
        else:
            # If targets are indices
            batch_size = predictions.shape[0]
            return -np.mean(np.log(predictions[range(batch_size), targets]))
    
    def compute_gradient(self, predictions: np.ndarray, 
                        targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss with respect to predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Gradient array
        """
        if self.loss_type == 'mse':
            return self.mse_gradient(predictions, targets)
        elif self.loss_type == 'cross_entropy':
            return self.cross_entropy_gradient(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def mse_gradient(self, predictions: np.ndarray, 
                    targets: np.ndarray) -> np.ndarray:
        """
        Gradient of MSE loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Gradient of MSE
        """
        return 2 * (predictions - targets) / predictions.size
    
    def cross_entropy_gradient(self, predictions: np.ndarray, 
                              targets: np.ndarray) -> np.ndarray:
        """
        Gradient of cross-entropy loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Gradient of cross-entropy
        """
        return (predictions - targets) / predictions.shape[0]
    
    def get_metrics(self, predictions: np.ndarray, 
                   targets: np.ndarray) -> dict:
        """
        Calculate additional metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        loss = self.compute_loss(predictions, targets)
        
        # Calculate accuracy for classification
        if self.loss_type == 'cross_entropy':
            pred_classes = np.argmax(predictions, axis=-1)
            if targets.ndim == predictions.ndim:
                target_classes = np.argmax(targets, axis=-1)
            else:
                target_classes = targets
            accuracy = np.mean(pred_classes == target_classes)
            
            return {
                'loss': loss,
                'accuracy': accuracy
            }
        else:
            # For regression, calculate R-squared
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            return {
                'loss': loss,
                'r2_score': r2
            }
