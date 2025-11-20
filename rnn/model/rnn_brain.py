"""
RNN Brain - Main orchestrator class
Integrates all RNN components into a cohesive, reusable brain.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .architecture import RNNLayer
from ..data.preprocessor import DataPreprocessor
from ..data.loader import DataLoader
from ..training.loss import LossCalculator
from ..training.optimizer import OptimizerManager
from ..training.trainer import Trainer
from ..utils.persistence import ModelPersistence
from ..utils.config import Config


class RNNBrain:
    """
    Complete RNN system organized into clear categories.
    
    This is a modular "brain" that can be inserted wherever you need
    recurrent neural network capabilities. All components are organized
    by their function in the learning process:
    
    Categories:
    1. Data Processing (preprocessor, data_loader)
    2. Network Architecture (rnn_layer)
    3. Learning Components (loss_calculator, optimizer, trainer)
    4. Persistence (model_persistence)
    5. Configuration (config)
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the RNN Brain with all its components.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        # Configuration
        self.config = config or Config()
        self.config.validate()
        
        # Extract key parameters
        input_size = self.config.get('input_size')
        hidden_size = self.config.get('hidden_size')
        output_size = self.config.get('output_size')
        activation = self.config.get('activation', 'tanh')
        
        # Category 1: Data Processing Components
        self.preprocessor = DataPreprocessor(
            vocab_size=self.config.get('vocab_size'),
            max_length=self.config.get('max_length')
        )
        
        self.data_loader = DataLoader(
            batch_size=self.config.get('batch_size', 32),
            shuffle=self.config.get('shuffle', True),
            sequence_length=self.config.get('sequence_length')
        )
        
        # Category 2: Network Architecture
        self.rnn_layer = RNNLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation
        )
        
        # Category 3: Learning Components
        self.loss_calculator = LossCalculator(
            loss_type=self.config.get('loss_type', 'mse'),
            l2_reg=self.config.get('l2_reg', 0.0)
        )
        
        self.optimizer = OptimizerManager(
            learning_rate=self.config.get('learning_rate', 0.01),
            optimizer_type=self.config.get('optimizer_type', 'adam'),
            momentum=self.config.get('momentum', 0.9),
            clip_value=self.config.get('gradient_clip')
        )
        
        self.trainer = Trainer(
            model=self,
            loss_calculator=self.loss_calculator,
            optimizer=self.optimizer,
            data_loader=self.data_loader
        )
        
        # Category 4: Persistence
        self.persistence = ModelPersistence(
            model_dir=self.config.get('model_dir', './models')
        )
        
        # Initialize state
        self.hidden_state = None
        self.is_trained = False
        
    def preprocess_data(self, data, is_text: bool = False):
        """
        Preprocess input data using the data processing category.
        
        Args:
            data: Raw input data
            is_text: Whether data is text (triggers tokenization)
            
        Returns:
            Preprocessed data ready for network
        """
        if is_text:
            return self.preprocessor.preprocess_sequence(data)
        else:
            return self.preprocessor.normalize(data)
    
    def predict(self, X: np.ndarray, return_sequences: bool = True) -> np.ndarray:
        """
        Make predictions using the network architecture category.
        
        Args:
            X: Input data
            return_sequences: Whether to return all timesteps or just last
            
        Returns:
            Model predictions
        """
        # Ensure correct shape (seq_length, input_size, batch_size)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        outputs, self.hidden_state = self.rnn_layer.forward(
            X, 
            self.hidden_state,
            return_sequences
        )
        
        return outputs
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = None, verbose: int = None) -> Dict:
        """
        Train the RNN brain using the learning components category.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data (optional)
            y_val: Validation target data (optional)
            epochs: Number of epochs (uses config default if None)
            verbose: Verbosity level (uses config default if None)
            
        Returns:
            Training history
        """
        # Use config defaults if not specified
        epochs = epochs or self.config.get('epochs', 100)
        verbose = verbose if verbose is not None else self.config.get('verbose', 1)
        patience = self.config.get('early_stopping_patience', 10)
        
        # Train the model
        history = self.trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            early_stopping_patience=patience,
            verbose=verbose
        )
        
        self.is_trained = True
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        metrics = self.loss_calculator.get_metrics(predictions, y_test)
        return metrics
    
    def save(self, model_name: str, metadata: Dict = None) -> str:
        """
        Save the RNN brain using the persistence category.
        
        Args:
            model_name: Name for the saved model
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model
        """
        if metadata is None:
            metadata = {}
        
        # Add configuration to metadata
        metadata['config'] = self.config.to_dict()
        metadata['is_trained'] = self.is_trained
        
        return self.persistence.save_model(self, model_name, metadata)
    
    def load(self, model_name: str) -> Dict:
        """
        Load the RNN brain using the persistence category.
        
        Args:
            model_name: Name of the saved model
            
        Returns:
            Loaded metadata
        """
        metadata = self.persistence.load_model(self, model_name)
        
        # Restore configuration if available
        if 'config' in metadata:
            self.config.update(metadata['config'])
        
        if 'is_trained' in metadata:
            self.is_trained = metadata['is_trained']
        
        return metadata
    
    def reset_state(self):
        """Reset the hidden state of the RNN."""
        self.hidden_state = None
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get all model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.rnn_layer.get_parameters()
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """
        Set model parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.rnn_layer.set_parameters(params)
    
    def compute_gradients(self, X: np.ndarray, output_gradient: np.ndarray) -> Dict:
        """
        Compute gradients (simplified for interface).
        
        Args:
            X: Input data
            output_gradient: Gradient from loss
            
        Returns:
            Dictionary of parameter gradients
        """
        # This is a simplified interface
        # Actual backpropagation through time would be more complex
        return {
            'W_ih': self.rnn_layer.cell.dW_ih,
            'W_hh': self.rnn_layer.cell.dW_hh,
            'b_h': self.rnn_layer.cell.db_h,
            'W_ho': self.rnn_layer.dW_ho,
            'b_o': self.rnn_layer.db_o
        }
    
    def get_config(self) -> Config:
        """Get configuration object."""
        return self.config
    
    def summary(self) -> str:
        """
        Print a summary of the RNN brain architecture.
        
        Returns:
            Summary string
        """
        summary = []
        summary.append("=" * 60)
        summary.append("RNN Brain Summary")
        summary.append("=" * 60)
        summary.append("\nArchitecture:")
        summary.append(f"  Input Size:  {self.config.get('input_size')}")
        summary.append(f"  Hidden Size: {self.config.get('hidden_size')}")
        summary.append(f"  Output Size: {self.config.get('output_size')}")
        summary.append(f"  Activation:  {self.config.get('activation')}")
        summary.append("\nTraining Configuration:")
        summary.append(f"  Learning Rate: {self.config.get('learning_rate')}")
        summary.append(f"  Optimizer:     {self.config.get('optimizer_type')}")
        summary.append(f"  Loss Type:     {self.config.get('loss_type')}")
        summary.append(f"  Batch Size:    {self.config.get('batch_size')}")
        summary.append(f"  Epochs:        {self.config.get('epochs')}")
        summary.append("\nData Processing:")
        summary.append(f"  Sequence Length: {self.config.get('sequence_length')}")
        summary.append(f"  Vocab Size:      {self.config.get('vocab_size')}")
        summary.append(f"  Max Length:      {self.config.get('max_length')}")
        summary.append("\nStatus:")
        summary.append(f"  Trained: {self.is_trained}")
        summary.append("=" * 60)
        
        return "\n".join(summary)
