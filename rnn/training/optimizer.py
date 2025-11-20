"""
Optimizer Category
Manages parameter updates during training.
"""

import numpy as np
from typing import Dict


class OptimizerManager:
    """
    Manages optimization algorithms for parameter updates.
    
    Categories of optimization:
    - Gradient descent: Basic parameter update
    - Momentum: Accelerate learning with velocity
    - Adaptive learning: Adjust learning rates per parameter
    - Gradient clipping: Prevent gradient explosion
    """
    
    def __init__(self, learning_rate: float = 0.01, 
                 optimizer_type: str = 'sgd',
                 momentum: float = 0.9,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 clip_value: float = None):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Learning rate
            optimizer_type: Type of optimizer ('sgd', 'momentum', 'adam')
            momentum: Momentum coefficient
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
            epsilon: Small constant for numerical stability
            clip_value: Gradient clipping threshold
        """
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value
        
        # Optimizer state
        self.velocity = {}
        self.m = {}  # First moment for Adam
        self.v = {}  # Second moment for Adam
        self.t = 0   # Time step for Adam
        
    def clip_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Clip gradients to prevent explosion.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Clipped gradients
        """
        if self.clip_value is None:
            return gradients
        
        clipped_grads = {}
        for name, grad in gradients.items():
            clipped_grads[name] = np.clip(grad, -self.clip_value, self.clip_value)
        
        return clipped_grads
    
    def update_parameters(self, parameters: Dict[str, np.ndarray],
                         gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Update parameters using gradients.
        
        Args:
            parameters: Current parameters
            gradients: Computed gradients
            
        Returns:
            Updated parameters
        """
        # Clip gradients
        gradients = self.clip_gradients(gradients)
        
        # Apply optimizer
        if self.optimizer_type == 'sgd':
            return self._sgd_update(parameters, gradients)
        elif self.optimizer_type == 'momentum':
            return self._momentum_update(parameters, gradients)
        elif self.optimizer_type == 'adam':
            return self._adam_update(parameters, gradients)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
    
    def _sgd_update(self, parameters: Dict[str, np.ndarray],
                    gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Stochastic Gradient Descent update.
        
        Args:
            parameters: Current parameters
            gradients: Computed gradients
            
        Returns:
            Updated parameters
        """
        updated_params = {}
        for name, param in parameters.items():
            if name in gradients:
                updated_params[name] = param - self.learning_rate * gradients[name]
            else:
                updated_params[name] = param
        
        return updated_params
    
    def _momentum_update(self, parameters: Dict[str, np.ndarray],
                        gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Momentum-based update.
        
        Args:
            parameters: Current parameters
            gradients: Computed gradients
            
        Returns:
            Updated parameters
        """
        updated_params = {}
        
        for name, param in parameters.items():
            if name in gradients:
                # Initialize velocity if needed
                if name not in self.velocity:
                    self.velocity[name] = np.zeros_like(param)
                
                # Update velocity
                self.velocity[name] = (self.momentum * self.velocity[name] - 
                                      self.learning_rate * gradients[name])
                
                # Update parameter
                updated_params[name] = param + self.velocity[name]
            else:
                updated_params[name] = param
        
        return updated_params
    
    def _adam_update(self, parameters: Dict[str, np.ndarray],
                    gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Adam optimizer update.
        
        Args:
            parameters: Current parameters
            gradients: Computed gradients
            
        Returns:
            Updated parameters
        """
        self.t += 1
        updated_params = {}
        
        for name, param in parameters.items():
            if name in gradients:
                # Initialize moments if needed
                if name not in self.m:
                    self.m[name] = np.zeros_like(param)
                    self.v[name] = np.zeros_like(param)
                
                grad = gradients[name]
                
                # Update biased first moment estimate
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected moments
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                updated_params[name] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:
                updated_params[name] = param
        
        return updated_params
    
    def reset_state(self):
        """Reset optimizer state (for new training run)."""
        self.velocity = {}
        self.m = {}
        self.v = {}
        self.t = 0
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.learning_rate
    
    def set_learning_rate(self, learning_rate: float):
        """Set learning rate."""
        self.learning_rate = learning_rate
