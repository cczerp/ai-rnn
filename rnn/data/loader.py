"""
Data Loader Category
Handles batch creation and data feeding during training.
"""

import numpy as np
from typing import Tuple, Iterator


class DataLoader:
    """
    Manages data batching and iteration for RNN training.
    
    Categories of data loading:
    - Batching: Create mini-batches from dataset
    - Shuffling: Randomize data order for better learning
    - Sequencing: Maintain temporal relationships in data
    - Iteration: Provide efficient data access
    """
    
    def __init__(self, batch_size: int = 32, shuffle: bool = True, 
                 sequence_length: int = None):
        """
        Initialize the data loader.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            sequence_length: Length of sequences (for sequence data)
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        
    def create_batches(self, X: np.ndarray, y: np.ndarray = None) -> Iterator[Tuple]:
        """
        Create batches from input data.
        
        Args:
            X: Input features
            y: Target labels (optional)
            
        Yields:
            Tuple of (batch_X, batch_y) or just batch_X if y is None
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = X[batch_indices]
            
            if y is not None:
                batch_y = y[batch_indices]
                yield batch_X, batch_y
            else:
                yield batch_X
    
    def create_sequences(self, data: np.ndarray, 
                        target_offset: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for time series prediction.
        
        Args:
            data: Time series data
            target_offset: Offset for target (1 means predict next step)
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        if self.sequence_length is None:
            raise ValueError("sequence_length must be set for sequence creation")
        
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - target_offset + 1):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + target_offset:i + self.sequence_length + target_offset])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray = None, 
                   train_ratio: float = 0.8) -> Tuple:
        """
        Split data into training and validation sets.
        
        Args:
            X: Input features
            y: Target labels (optional)
            train_ratio: Proportion of data for training
            
        Returns:
            Tuple of (X_train, X_val) or (X_train, X_val, y_train, y_val)
        """
        n_samples = len(X)
        split_idx = int(n_samples * train_ratio)
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        
        if y is not None:
            y_train, y_val = y[:split_idx], y[split_idx:]
            return X_train, X_val, y_train, y_val
        
        return X_train, X_val
    
    def get_batch_count(self, n_samples: int) -> int:
        """
        Calculate number of batches for given sample count.
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            Number of batches
        """
        return (n_samples + self.batch_size - 1) // self.batch_size
