# Quick Start Guide

Get started with the RNN Brain in 5 minutes!

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Import and Configure

```python
from rnn import RNNBrain
from rnn.utils.config import Config

# Create configuration
config = Config({
    'input_size': 10,      # Dimension of input features
    'hidden_size': 64,     # Number of hidden units
    'output_size': 10,     # Dimension of output
    'learning_rate': 0.01,
    'epochs': 100
})
```

### 2. Initialize the Brain

```python
# Create RNN brain with all categories initialized
brain = RNNBrain(config)

# View summary
print(brain.summary())
```

### 3. Prepare Your Data

```python
import numpy as np

# For time series
X_train = np.random.randn(100, 20, 10)  # (samples, seq_len, features)
y_train = np.random.randn(100, 20, 10)

# Split into train/validation
X_train, X_val, y_train, y_val = brain.data_loader.split_data(
    X_train, y_train, train_ratio=0.8
)
```

### 4. Train the Brain

```python
# Train the model
history = brain.fit(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    verbose=1
)
```

### 5. Make Predictions

```python
# Make predictions
X_test = np.random.randn(10, 20, 10)
predictions = brain.predict(X_test)
```

### 6. Save and Load

```python
# Save trained brain
brain.save('my_trained_brain')

# Load later
new_brain = RNNBrain(config)
new_brain.load('my_trained_brain')
```

## Complete Example

```python
from rnn import RNNBrain
from rnn.utils.config import Config
import numpy as np

# 1. Configure
config = Config({
    'input_size': 5,
    'hidden_size': 32,
    'output_size': 5,
    'learning_rate': 0.01,
    'batch_size': 16,
    'epochs': 50
})

# 2. Initialize
brain = RNNBrain(config)

# 3. Create dummy data
X_train = np.random.randn(200, 10, 5)  # 200 samples, 10 timesteps, 5 features
y_train = np.random.randn(200, 10, 5)

# 4. Split data
X_train, X_val, y_train, y_val = brain.data_loader.split_data(
    X_train, y_train, train_ratio=0.8
)

# 5. Train
print("Training...")
history = brain.fit(X_train, y_train, X_val, y_val, epochs=50)

# 6. Evaluate
X_test = np.random.randn(50, 10, 5)
y_test = np.random.randn(50, 10, 5)
metrics = brain.evaluate(X_test, y_test)
print(f"Test metrics: {metrics}")

# 7. Save
brain.save('my_model')
print("Model saved!")
```

## Common Use Cases

### Time Series Prediction

```python
# Generate sine wave
t = np.linspace(0, 100, 1000)
data = np.sin(t)

# Create sequences
X, y = brain.data_loader.create_sequences(
    data.reshape(-1, 1), 
    target_offset=1
)
```

### Text Processing

```python
# Tokenize text
text = "hello world"
tokens = brain.preprocessor.tokenize(text)
padded = brain.preprocessor.pad_sequence(tokens)
```

### Classification

```python
config = Config({
    'input_size': 10,
    'hidden_size': 64,
    'output_size': 5,  # 5 classes
    'loss_type': 'cross_entropy'  # For classification
})
```

## Run Examples

See full working examples:

```bash
# Run all examples
python example_usage.py

# Run tests
python test_rnn_brain.py
```

## Next Steps

- Read [CATEGORIES.md](CATEGORIES.md) to understand the architecture
- Check [README.md](README.md) for detailed documentation
- Modify configurations for your specific use case
- Extend categories with custom functionality

## Need Help?

The RNN brain is organized into 5 categories:

1. **Data Processing** - Prepare and batch data
2. **Network Architecture** - Define RNN structure
3. **Learning Components** - Train the network
4. **Persistence** - Save/load models
5. **Configuration** - Manage settings

Each category is independent and can be customized!
