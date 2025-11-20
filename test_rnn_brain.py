"""
Basic tests for RNN Brain functionality
"""

import numpy as np
from rnn import RNNBrain
from rnn.utils.config import Config


def test_rnn_initialization():
    """Test that RNN brain initializes correctly"""
    config = Config({
        'input_size': 5,
        'hidden_size': 10,
        'output_size': 5
    })
    
    brain = RNNBrain(config)
    assert brain is not None
    assert brain.config.get('input_size') == 5
    assert brain.config.get('hidden_size') == 10
    print("✓ RNN initialization test passed")


def test_data_preprocessing():
    """Test data preprocessing components"""
    config = Config()
    brain = RNNBrain(config)
    
    # Test normalization
    data = np.random.randn(100, 10)
    normalized = brain.preprocessor.normalize(data)
    assert normalized.shape == data.shape
    
    # Test tokenization
    text = "hello world"
    tokens = brain.preprocessor.tokenize(text)
    assert len(tokens) > 0
    
    print("✓ Data preprocessing test passed")


def test_data_loading():
    """Test data loading and batching"""
    config = Config({'batch_size': 16})
    brain = RNNBrain(config)
    
    # Create dummy data
    X = np.random.randn(100, 10)
    y = np.random.randn(100, 5)
    
    # Test batch creation
    batch_count = 0
    for batch_X, batch_y in brain.data_loader.create_batches(X, y):
        assert batch_X.shape[0] <= 16
        assert batch_y.shape[0] <= 16
        batch_count += 1
    
    assert batch_count > 0
    print("✓ Data loading test passed")


def test_rnn_prediction():
    """Test RNN prediction"""
    config = Config({
        'input_size': 5,
        'hidden_size': 10,
        'output_size': 3
    })
    
    brain = RNNBrain(config)
    
    # Create dummy input
    X = np.random.randn(20, 5, 2)  # seq_length=20, input_size=5, batch_size=2
    
    # Test prediction
    predictions = brain.predict(X)
    assert predictions.shape[0] == 20  # sequence length
    assert predictions.shape[1] == 3   # output size
    
    print("✓ RNN prediction test passed")


def test_model_persistence():
    """Test model saving and loading"""
    config = Config({
        'input_size': 5,
        'hidden_size': 10,
        'output_size': 5
    })
    
    # Create and save a model
    brain1 = RNNBrain(config)
    original_params = brain1.get_parameters()
    brain1.save('test_model')
    
    # Create new model and load
    brain2 = RNNBrain(config)
    brain2.load('test_model')
    loaded_params = brain2.get_parameters()
    
    # Verify parameters match
    for key in original_params:
        assert np.allclose(original_params[key], loaded_params[key])
    
    # Cleanup
    brain1.persistence.delete_model('test_model')
    
    print("✓ Model persistence test passed")


def test_configuration():
    """Test configuration management"""
    # Test default config
    config1 = Config()
    assert config1.get('input_size') is not None
    
    # Test custom config
    config2 = Config({
        'input_size': 20,
        'hidden_size': 128,
        'learning_rate': 0.001
    })
    assert config2.get('input_size') == 20
    assert config2.get('hidden_size') == 128
    assert config2.get('learning_rate') == 0.001
    
    # Test config save/load
    config2.save('/tmp/test_config.json')
    config3 = Config.load('/tmp/test_config.json')
    assert config3.get('input_size') == 20
    
    print("✓ Configuration test passed")


def test_loss_calculator():
    """Test loss calculation"""
    from rnn.training.loss import LossCalculator
    
    # Test MSE loss
    loss_calc = LossCalculator(loss_type='mse')
    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([1.0, 2.0, 3.0])
    loss = loss_calc.compute_loss(predictions, targets)
    assert loss == 0.0
    
    # Test with error
    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([2.0, 3.0, 4.0])
    loss = loss_calc.compute_loss(predictions, targets)
    assert loss > 0
    
    print("✓ Loss calculator test passed")


def test_optimizer():
    """Test optimizer functionality"""
    from rnn.training.optimizer import OptimizerManager
    
    optimizer = OptimizerManager(learning_rate=0.01, optimizer_type='sgd')
    
    # Test parameter update
    params = {'W': np.array([[1.0, 2.0], [3.0, 4.0]])}
    grads = {'W': np.array([[0.1, 0.1], [0.1, 0.1]])}
    
    updated = optimizer.update_parameters(params, grads)
    
    # Verify parameters were updated
    assert not np.array_equal(params['W'], updated['W'])
    
    print("✓ Optimizer test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running RNN Brain Tests")
    print("="*60 + "\n")
    
    test_rnn_initialization()
    test_data_preprocessing()
    test_data_loading()
    test_rnn_prediction()
    test_model_persistence()
    test_configuration()
    test_loss_calculator()
    test_optimizer()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")


if __name__ == '__main__':
    run_all_tests()
