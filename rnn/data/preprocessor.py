"""
Data Preprocessor Category
Handles data transformation and preparation for RNN input.
"""

import numpy as np
from typing import Union, List, Tuple


class DataPreprocessor:
    """
    Preprocesses raw data into format suitable for RNN training.
    
    Categories of preprocessing:
    - Normalization: Scale data to appropriate ranges
    - Tokenization: Convert text/sequences to numerical representations
    - Padding: Ensure uniform sequence lengths
    - Encoding: Transform categorical data to numerical format
    """
    
    def __init__(self, vocab_size: int = None, max_length: int = None):
        """
        Initialize the data preprocessor.
        
        Args:
            vocab_size: Size of vocabulary for tokenization
            max_length: Maximum sequence length for padding
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = {}
        self.inverse_vocab = {}
        self.mean = None
        self.std = None
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize numerical data using z-score normalization.
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
        """
        if self.mean is None:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0) + 1e-8
        
        return (data - self.mean) / self.std
    
    def tokenize(self, text: Union[str, List[str]]) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            List of token IDs
        """
        if isinstance(text, str):
            text = list(text)
        
        tokens = []
        for char in text:
            if char not in self.vocab:
                if self.vocab_size and len(self.vocab) >= self.vocab_size:
                    tokens.append(0)  # Unknown token
                else:
                    self.vocab[char] = len(self.vocab) + 1
                    self.inverse_vocab[self.vocab[char]] = char
                    tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab[char])
        
        return tokens
    
    def pad_sequence(self, sequence: List[int], pad_value: int = 0) -> np.ndarray:
        """
        Pad sequence to maximum length.
        
        Args:
            sequence: Input sequence
            pad_value: Value to use for padding
            
        Returns:
            Padded sequence
        """
        if self.max_length is None:
            return np.array(sequence)
        
        if len(sequence) >= self.max_length:
            return np.array(sequence[:self.max_length])
        
        padded = np.full(self.max_length, pad_value)
        padded[:len(sequence)] = sequence
        return padded
    
    def encode_categorical(self, categories: List, num_classes: int = None) -> np.ndarray:
        """
        One-hot encode categorical data.
        
        Args:
            categories: List of category indices
            num_classes: Number of classes
            
        Returns:
            One-hot encoded array
        """
        if num_classes is None:
            num_classes = max(categories) + 1
        
        encoded = np.zeros((len(categories), num_classes))
        encoded[np.arange(len(categories)), categories] = 1
        return encoded
    
    def preprocess_sequence(self, data: Union[str, List, np.ndarray]) -> np.ndarray:
        """
        Complete preprocessing pipeline for sequence data.
        
        Args:
            data: Input data (text, sequence, or array)
            
        Returns:
            Preprocessed data ready for RNN
        """
        if isinstance(data, str):
            tokens = self.tokenize(data)
            return self.pad_sequence(tokens)
        elif isinstance(data, list):
            if all(isinstance(x, str) for x in data):
                tokens = self.tokenize(data)
                return self.pad_sequence(tokens)
            return self.pad_sequence(data)
        else:
            return self.normalize(data)
