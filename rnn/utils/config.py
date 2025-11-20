"""
Configuration Category
Manages hyperparameters and settings for the RNN.
"""

import json
from typing import Dict, Any


class Config:
    """
    Configuration manager for RNN hyperparameters and settings.
    
    Categories of configuration:
    - Model architecture: Network structure parameters
    - Training settings: Learning rate, batch size, etc.
    - Data parameters: Input/output dimensions, preprocessing
    - Optimization: Optimizer settings and regularization
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        # Model architecture
        'input_size': 10,
        'hidden_size': 64,
        'output_size': 10,
        'activation': 'tanh',
        'num_layers': 1,
        
        # Training parameters
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 10,
        
        # Optimizer
        'optimizer_type': 'adam',
        'momentum': 0.9,
        'beta1': 0.9,
        'beta2': 0.999,
        'gradient_clip': 5.0,
        
        # Loss and regularization
        'loss_type': 'mse',
        'l2_reg': 0.0001,
        
        # Data processing
        'sequence_length': 20,
        'vocab_size': None,
        'max_length': 100,
        'shuffle': True,
        'train_ratio': 0.8,
        
        # Model persistence
        'model_dir': './models',
        'save_checkpoints': True,
        'checkpoint_frequency': 10,
        
        # Misc
        'random_seed': 42,
        'verbose': 1
    }
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Dictionary of configuration parameters
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_dict is not None:
            self.update(config_dict)
    
    def update(self, config_dict: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            config_dict: Dictionary of parameters to update
        """
        self.config.update(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def save(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration
        """
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Config instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required parameters
        required = ['input_size', 'hidden_size', 'output_size']
        for key in required:
            if key not in self.config or self.config[key] is None:
                raise ValueError(f"Required parameter '{key}' is missing")
        
        # Check parameter types and ranges
        if self.config['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.config['batch_size'] < 1:
            raise ValueError("batch_size must be at least 1")
        
        if self.config['epochs'] < 1:
            raise ValueError("epochs must be at least 1")
        
        if self.config['hidden_size'] < 1:
            raise ValueError("hidden_size must be at least 1")
        
        return True
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({json.dumps(self.config, indent=2)})"
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting."""
        self.config[key] = value
