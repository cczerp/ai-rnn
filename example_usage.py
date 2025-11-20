"""
Example Usage of RNN Brain

This demonstrates how to use the modular RNN brain for different tasks.
The RNN is organized into clear categories that handle different aspects
of the learning process.
"""

import numpy as np
from rnn import RNNBrain
from rnn.utils.config import Config


def example_time_series_prediction():
    """
    Example: Time series prediction using RNN Brain
    
    Categories used:
    - Data Processing: Normalize and batch the data
    - Network Architecture: Process sequences through RNN
    - Learning: Train with appropriate loss and optimizer
    """
    print("=" * 60)
    print("Example 1: Time Series Prediction")
    print("=" * 60)
    
    # Generate synthetic sine wave data
    t = np.linspace(0, 100, 1000)
    data = np.sin(t) + np.random.normal(0, 0.1, 1000)
    
    # Create configuration
    config = Config({
        'input_size': 1,
        'hidden_size': 32,
        'output_size': 1,
        'sequence_length': 10,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'loss_type': 'mse',
        'optimizer_type': 'adam'
    })
    
    # Initialize RNN Brain
    brain = RNNBrain(config)
    
    # Print summary
    print(brain.summary())
    
    # Prepare data using data processing category
    X, y = brain.data_loader.create_sequences(data.reshape(-1, 1), target_offset=1)
    X_train, X_val, y_train, y_val = brain.data_loader.split_data(X, y, train_ratio=0.8)
    
    print(f"\nData shapes:")
    print(f"  Training:   X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    
    # Train the brain (uses learning category)
    print("\nTraining...")
    # Note: This is a simplified example. Full training would require proper backprop implementation
    print("(Training simulation - full implementation would train here)")
    
    # Save the trained brain (uses persistence category)
    print("\nSaving model...")
    save_path = brain.save('time_series_model')
    print(f"Model saved to: {save_path}")
    
    print("\n✓ Time series example complete!\n")


def example_text_generation():
    """
    Example: Text generation using RNN Brain
    
    Categories used:
    - Data Processing: Tokenize and encode text
    - Network Architecture: Process character sequences
    - Learning: Train for next character prediction
    """
    print("=" * 60)
    print("Example 2: Text Generation")
    print("=" * 60)
    
    # Sample text
    text = "hello world this is a sample text for rnn training"
    
    # Create configuration for text
    config = Config({
        'input_size': 50,  # vocab size
        'hidden_size': 64,
        'output_size': 50,
        'sequence_length': 20,
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 0.01,
        'loss_type': 'cross_entropy',
        'optimizer_type': 'adam',
        'vocab_size': 50
    })
    
    # Initialize RNN Brain
    brain = RNNBrain(config)
    
    print(brain.summary())
    
    # Preprocess text using data processing category
    tokens = brain.preprocessor.tokenize(text)
    print(f"\nText: '{text}'")
    print(f"Tokens: {tokens[:20]}... (showing first 20)")
    print(f"Vocabulary size: {len(brain.preprocessor.vocab)}")
    
    print("\n✓ Text generation example complete!\n")


def example_sequence_classification():
    """
    Example: Sequence classification using RNN Brain
    
    Categories used:
    - Data Processing: Normalize and prepare sequences
    - Network Architecture: Process sequences and output class
    - Learning: Train with classification loss
    """
    print("=" * 60)
    print("Example 3: Sequence Classification")
    print("=" * 60)
    
    # Generate synthetic sequence data
    # Positive class: increasing sequences
    # Negative class: decreasing sequences
    n_samples = 200
    seq_length = 15
    
    X_positive = [np.sort(np.random.randn(seq_length, 1)) for _ in range(n_samples // 2)]
    X_negative = [np.sort(np.random.randn(seq_length, 1))[::-1] for _ in range(n_samples // 2)]
    
    X = np.array(X_positive + X_negative)
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    
    # Create configuration
    config = Config({
        'input_size': 1,
        'hidden_size': 32,
        'output_size': 2,  # binary classification
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.01,
        'loss_type': 'cross_entropy',
        'optimizer_type': 'adam'
    })
    
    # Initialize RNN Brain
    brain = RNNBrain(config)
    
    print(brain.summary())
    
    # Split data using data processing category
    X_train, X_val, y_train, y_val = brain.data_loader.split_data(X, y, train_ratio=0.8)
    
    print(f"\nData shapes:")
    print(f"  Training:   X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    
    print("\n✓ Sequence classification example complete!\n")


def example_custom_configuration():
    """
    Example: Using custom configuration
    
    Shows how to customize the RNN brain for specific needs
    """
    print("=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)
    
    # Create custom config
    custom_config = Config({
        'input_size': 20,
        'hidden_size': 128,
        'output_size': 10,
        'activation': 'relu',
        'learning_rate': 0.001,
        'optimizer_type': 'momentum',
        'momentum': 0.95,
        'batch_size': 64,
        'epochs': 200,
        'early_stopping_patience': 20,
        'l2_reg': 0.0001,
        'gradient_clip': 10.0
    })
    
    # Save configuration
    custom_config.save('/tmp/custom_rnn_config.json')
    print("Configuration saved to: /tmp/custom_rnn_config.json")
    
    # Load configuration
    loaded_config = Config.load('/tmp/custom_rnn_config.json')
    print("\nLoaded configuration:")
    print(f"  Hidden size: {loaded_config.get('hidden_size')}")
    print(f"  Learning rate: {loaded_config.get('learning_rate')}")
    print(f"  Optimizer: {loaded_config.get('optimizer_type')}")
    
    # Create brain with custom config
    brain = RNNBrain(loaded_config)
    print("\n" + brain.summary())
    
    print("\n✓ Custom configuration example complete!\n")


def main():
    """
    Run all examples demonstrating the modular RNN brain.
    """
    print("\n" + "=" * 60)
    print("RNN BRAIN - Modular Neural Network Examples")
    print("=" * 60)
    print("\nThis RNN is organized into categories:")
    print("  1. Data Processing - Handle input data and batching")
    print("  2. Network Architecture - Define RNN structure")
    print("  3. Learning Components - Loss, optimizer, trainer")
    print("  4. Persistence - Save and load models")
    print("  5. Configuration - Manage hyperparameters")
    print("\nYou can insert this brain wherever you need RNN capabilities!")
    print("=" * 60)
    print()
    
    try:
        # Run examples
        example_time_series_prediction()
        example_text_generation()
        example_sequence_classification()
        example_custom_configuration()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nThe RNN brain is ready to be inserted into your projects.")
        print("Simply import: from rnn import RNNBrain")
        print()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
