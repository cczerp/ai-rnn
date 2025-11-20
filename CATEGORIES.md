# RNN Categories Explained

This document explains how the RNN is organized into categories based on what a recurrent neural network needs to feed itself and learn.

## Philosophy

Instead of organizing code by technical layers, this RNN is organized by **functional purpose**. Each category represents a key capability that an RNN needs to operate and learn effectively.

## The 5 Categories

### 1. Data Processing Category (`rnn/data/`)

**Purpose**: Transform raw data into a form the RNN can consume

**What the RNN needs**:
- Raw input data needs to be cleaned and normalized
- Text needs to be converted to numbers
- Sequences need to be the right length
- Data needs to be organized into batches for efficient learning

**Components**:
- `DataPreprocessor` - Handles:
  - **Normalization**: Scale numerical data to appropriate ranges
  - **Tokenization**: Convert text to numerical tokens
  - **Padding**: Ensure sequences are uniform length
  - **Encoding**: Transform categorical data to numerical format

- `DataLoader` - Handles:
  - **Batching**: Create mini-batches for efficient training
  - **Shuffling**: Randomize data order for better learning
  - **Sequencing**: Create input-output pairs from time series
  - **Splitting**: Divide data into train/validation sets

**Example**:
```python
preprocessor = DataPreprocessor(vocab_size=1000, max_length=100)
tokens = preprocessor.tokenize("hello world")
padded = preprocessor.pad_sequence(tokens)
```

---

### 2. Network Architecture Category (`rnn/model/`)

**Purpose**: Define the computational structure that processes sequences

**What the RNN needs**:
- A way to maintain memory across time steps
- A mechanism to process sequential input
- A structure to compute outputs at each step
- Parameters that can be updated through learning

**Components**:
- `RNNCell` - The core recurrent unit:
  - **State Initialization**: Start with initial hidden state
  - **Forward Computation**: Process input with previous state
  - **Activation**: Apply non-linear transformation
  - **State Update**: Produce new hidden state for next step

- `RNNLayer` - Processes sequences:
  - **Sequence Processing**: Run cell across all time steps
  - **State Management**: Track hidden states through time
  - **Output Generation**: Produce outputs at each step
  - **Parameter Storage**: Hold weights and biases

- `RNNBrain` - Main orchestrator:
  - Integrates all categories
  - Provides simple interface
  - Manages overall state

**Example**:
```python
rnn_layer = RNNLayer(input_size=10, hidden_size=64, output_size=10)
outputs, final_state = rnn_layer.forward(input_sequence)
```

---

### 3. Learning Components Category (`rnn/training/`)

**Purpose**: Enable the network to improve through experience

**What the RNN needs**:
- A way to measure how wrong its predictions are
- A method to calculate how to adjust parameters
- An algorithm to actually update the parameters
- A process to iterate through training epochs

**Components**:
- `LossCalculator` - Measures performance:
  - **Forward Loss**: Calculate error between predictions and targets
  - **Gradient Computation**: Calculate derivatives for backpropagation
  - **Loss Types**: Support MSE (regression) and Cross-Entropy (classification)
  - **Regularization**: Add penalties to prevent overfitting
  - **Metrics**: Calculate accuracy, R², etc.

- `OptimizerManager` - Updates parameters:
  - **Gradient Descent**: Basic parameter updates (SGD)
  - **Momentum**: Accelerate learning with velocity
  - **Adaptive Learning**: Adjust per-parameter learning rates (Adam)
  - **Gradient Clipping**: Prevent gradient explosion

- `Trainer` - Orchestrates training:
  - **Epoch Iteration**: Loop through dataset multiple times
  - **Batch Processing**: Train on mini-batches
  - **Progress Tracking**: Monitor metrics over time
  - **Early Stopping**: Stop when validation loss stops improving
  - **Checkpointing**: Save best model during training

**Example**:
```python
loss_calc = LossCalculator(loss_type='mse')
optimizer = OptimizerManager(learning_rate=0.01, optimizer_type='adam')
trainer = Trainer(model, loss_calc, optimizer, data_loader)
history = trainer.train(X_train, y_train, X_val, y_val, epochs=100)
```

---

### 4. Persistence Category (`rnn/utils/`)

**Purpose**: Save and restore the network's learned knowledge

**What the RNN needs**:
- A way to save trained parameters to disk
- A method to load parameters back
- Checkpoint saving during training
- Format conversion for different use cases

**Components**:
- `ModelPersistence` - Manages storage:
  - **Model Saving**: Save parameters and metadata to disk
  - **Model Loading**: Restore parameters from disk
  - **Checkpoint Management**: Save periodic training checkpoints
  - **Export/Import**: Convert to/from different formats
  - **Model Listing**: See all saved models
  - **Cleanup**: Delete old models

**Example**:
```python
persistence = ModelPersistence(model_dir='./models')
persistence.save_model(brain, 'my_trained_model')
# Later...
persistence.load_model(brain, 'my_trained_model')
```

---

### 5. Configuration Category (`rnn/utils/`)

**Purpose**: Manage all hyperparameters and settings in one place

**What the RNN needs**:
- A centralized place for all settings
- Default values for common configurations
- Easy way to customize behavior
- Ability to save/load configurations
- Validation to catch invalid settings

**Components**:
- `Config` - Configuration management:
  - **Default Settings**: Reasonable defaults for all parameters
  - **Customization**: Easy override of any setting
  - **Persistence**: Save/load configurations from JSON
  - **Validation**: Ensure parameters are valid
  - **Access**: Dictionary-style and method access

**Configurable Aspects**:
- Model architecture (sizes, activation)
- Training parameters (learning rate, epochs)
- Optimizer settings (type, momentum)
- Loss and regularization
- Data processing (batch size, sequence length)
- Persistence (model directory, checkpointing)

**Example**:
```python
config = Config({
    'input_size': 10,
    'hidden_size': 64,
    'learning_rate': 0.001,
    'epochs': 100,
    'optimizer_type': 'adam'
})
config.save('my_config.json')
loaded_config = Config.load('my_config.json')
```

---

## How Categories Work Together

The RNN brain integrates all categories into a cohesive system:

```python
from rnn import RNNBrain
from rnn.utils.config import Config

# 5. Configuration - Set up all parameters
config = Config({
    'input_size': 10,
    'hidden_size': 64,
    'output_size': 10,
    'learning_rate': 0.01
})

# Initialize brain (automatically sets up all categories)
brain = RNNBrain(config)

# 1. Data Processing - Prepare your data
X_processed = brain.preprocess_data(raw_data)

# 2. Network Architecture - Make predictions
predictions = brain.predict(X_processed)

# 3. Learning Components - Train the network
history = brain.fit(X_train, y_train, X_val, y_val)

# 4. Persistence - Save the trained brain
brain.save('my_brain')
```

## Benefits of This Organization

1. **Clarity**: Each category has a clear purpose
2. **Modularity**: Categories can be modified independently
3. **Reusability**: Take the whole brain or just parts you need
4. **Maintainability**: Easy to find and fix issues
5. **Extensibility**: Add new features within appropriate category
6. **Understanding**: Maps to conceptual model of how RNNs work

## Extending the Categories

Want to add new functionality? Place it in the appropriate category:

- New preprocessing? → Add to `Data Processing`
- New architecture? → Add to `Network Architecture`
- New optimizer? → Add to `Learning Components`
- New save format? → Add to `Persistence`
- New parameter? → Add to `Configuration`

This keeps the codebase organized and maintainable as it grows.
