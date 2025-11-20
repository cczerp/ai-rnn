"""
RNN Architecture Category
Defines the neural network structure and forward computation.
"""

import numpy as np
from typing import Tuple


class RNNCell:
    """
    Basic RNN cell implementing the recurrent computation.
    
    Categories of RNN cell operations:
    - State initialization: Set initial hidden state
    - Forward computation: Process input and previous state
    - Activation: Apply non-linearity
    - State update: Update hidden state for next timestep
    """
    
    def __init__(self, input_size: int, hidden_size: int, activation: str = 'tanh'):
        """
        Initialize RNN cell parameters.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            activation: Activation function ('tanh' or 'relu')
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_name = activation
        
        # Initialize weights
        self.W_ih = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        
        # Gradients
        self.dW_ih = np.zeros_like(self.W_ih)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)
        
    def activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation_name == 'tanh':
            return np.tanh(x)
        elif self.activation_name == 'relu':
            return np.maximum(0, x)
        return x
    
    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation_name == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_name == 'relu':
            return (x > 0).astype(float)
        return np.ones_like(x)
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through RNN cell.
        
        Args:
            x: Input at current timestep (input_size, batch_size)
            h_prev: Previous hidden state (hidden_size, batch_size)
            
        Returns:
            Tuple of (hidden_state, activation_input)
        """
        # Compute pre-activation
        z = np.dot(self.W_ih, x) + np.dot(self.W_hh, h_prev) + self.b_h
        
        # Apply activation
        h = self.activation(z)
        
        return h, z
    
    def initialize_hidden(self, batch_size: int = 1) -> np.ndarray:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Number of samples in batch
            
        Returns:
            Initial hidden state (hidden_size, batch_size)
        """
        return np.zeros((self.hidden_size, batch_size))


class RNNLayer:
    """
    RNN layer that processes sequences using RNN cells.
    
    Categories of layer operations:
    - Sequence processing: Process entire sequence through time
    - State management: Maintain hidden states across timesteps
    - Output generation: Produce outputs at each timestep
    - Backpropagation through time: Compute gradients across time
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 output_size: int = None, activation: str = 'tanh'):
        """
        Initialize RNN layer.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            output_size: Dimension of output (if None, uses hidden_size)
            activation: Activation function
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        
        # RNN cell
        self.cell = RNNCell(input_size, hidden_size, activation)
        
        # Output layer weights
        self.W_ho = np.random.randn(self.output_size, hidden_size) * 0.01
        self.b_o = np.zeros((self.output_size, 1))
        
        # Output layer gradients
        self.dW_ho = np.zeros_like(self.W_ho)
        self.db_o = np.zeros_like(self.b_o)
        
        # Cache for backward pass
        self.cache = {}
        
    def forward(self, X: np.ndarray, h0: np.ndarray = None, 
                return_sequences: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through entire sequence.
        
        Args:
            X: Input sequence (seq_length, input_size, batch_size)
            h0: Initial hidden state (hidden_size, batch_size)
            return_sequences: Whether to return all outputs or just last
            
        Returns:
            Tuple of (outputs, final_hidden_state)
        """
        seq_length, _, batch_size = X.shape
        
        # Initialize hidden state
        if h0 is None:
            h = self.cell.initialize_hidden(batch_size)
        else:
            h = h0
        
        # Store states for backward pass
        hidden_states = [h]
        outputs = []
        
        # Process sequence
        for t in range(seq_length):
            h, z = self.cell.forward(X[t], h)
            hidden_states.append(h)
            
            # Compute output
            output = np.dot(self.W_ho, h) + self.b_o
            outputs.append(output)
        
        # Cache for backward pass
        self.cache = {
            'X': X,
            'hidden_states': hidden_states,
            'outputs': outputs
        }
        
        outputs = np.array(outputs)
        
        if return_sequences:
            return outputs, h
        else:
            return outputs[-1], h
    
    def get_parameters(self) -> dict:
        """
        Get all layer parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'W_ih': self.cell.W_ih,
            'W_hh': self.cell.W_hh,
            'b_h': self.cell.b_h,
            'W_ho': self.W_ho,
            'b_o': self.b_o
        }
    
    def set_parameters(self, params: dict):
        """
        Set layer parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.cell.W_ih = params['W_ih']
        self.cell.W_hh = params['W_hh']
        self.cell.b_h = params['b_h']
        self.W_ho = params['W_ho']
        self.b_o = params['b_o']
