# ai-rnn
Basic modular RNN brain to insert wherever you need recurrent neural network capabilities.

## Overview

This repository provides a clean, modular implementation of a Recurrent Neural Network (RNN) organized into clear categories. Each category handles a specific aspect of what an RNN needs to learn and operate.

## Architecture - Organized by Category

The RNN is split into **5 main categories** that represent different functional areas:

### 1. **Data Processing** (`rnn/data/`)
Handles everything related to feeding data to the network:
- **DataPreprocessor**: Normalization, tokenization, padding, encoding
- **DataLoader**: Batching, shuffling, sequence creation, train/val splits

### 2. **Network Architecture** (`rnn/model/`)
Defines the neural network structure:
- **RNNCell**: Core recurrent computation unit
- **RNNLayer**: Processes entire sequences through time
- **RNNBrain**: Main orchestrator class that ties everything together

### 3. **Learning Components** (`rnn/training/`)
Manages how the network learns:
- **LossCalculator**: Computes loss and gradients (MSE, Cross-Entropy)
- **OptimizerManager**: Updates parameters (SGD, Momentum, Adam)
- **Trainer**: Orchestrates the training loop with early stopping

### 4. **Persistence** (`rnn/utils/`)
Handles model saving and loading:
- **ModelPersistence**: Save/load models, checkpoints, export/import

### 5. **Configuration** (`rnn/utils/`)
Manages hyperparameters and settings:
- **Config**: Centralized configuration with validation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from rnn import RNNBrain
from rnn.utils.config import Config

# Create configuration
config = Config({
    'input_size': 10,
    'hidden_size': 64,
    'output_size': 10,
    'learning_rate': 0.01,
    'epochs': 100
})

# Initialize the brain
brain = RNNBrain(config)

# Train the model
history = brain.fit(X_train, y_train, X_val, y_val)

# Make predictions
predictions = brain.predict(X_test)

# Save the brain
brain.save('my_rnn_model')

# Load later
brain.load('my_rnn_model')
```

## Usage Examples

See `example_usage.py` for complete examples including:
- Time series prediction
- Text generation
- Sequence classification
- Custom configuration

Run examples:
```bash
python example_usage.py
```

## Features

- **Modular Design**: Each category is independent and can be customized
- **Easy Integration**: Import and use in any project
- **Flexible Configuration**: JSON-based config system
- **Multiple Optimizers**: SGD, Momentum, Adam
- **Multiple Loss Functions**: MSE, Cross-Entropy
- **Data Processing**: Built-in preprocessing and data loading
- **Model Persistence**: Save and load trained models
- **Training Features**: Early stopping, checkpointing, metrics tracking

## Categories Explained

### Why Categories?

The RNN is organized by **function** rather than just code structure. Each category represents what the RNN needs to:

1. **Process Data** - Take raw input and prepare it
2. **Compute** - Run through the network architecture
3. **Learn** - Update itself based on errors
4. **Remember** - Save and restore its knowledge
5. **Configure** - Set its behavior and hyperparameters

This makes it easy to:
- Understand what each part does
- Modify specific functionality
- Insert the brain into different projects
- Extend with new features

## Use Cases

- **Time Series Forecasting**: Stock prices, weather, sensor data
- **Natural Language Processing**: Text generation, sentiment analysis
- **Sequence Classification**: Activity recognition, gesture classification
- **Anomaly Detection**: Pattern recognition in sequential data

## License

See LICENSE file for details.
