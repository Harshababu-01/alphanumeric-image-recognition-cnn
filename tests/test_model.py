import pytest
import numpy as np
from src.train import build_model
from config.config import INPUT_SHAPE
import os

def test_model_architecture():
    """Test that the model initializes with standard shapes."""
    model = build_model()
    # first dimension is batch size (None)
    assert model.input_shape == (None, *INPUT_SHAPE), f"Expected input shape (None, {INPUT_SHAPE}), got {model.input_shape}"
    assert model.output_shape == (None, 10), f"Expected output shape (None, 10), got {model.output_shape}"

def test_preprocess_data():
    """Test data preprocessing and normalization correctness."""
    from src.preprocess import load_and_preprocess_data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    assert x_train.shape[1:] == INPUT_SHAPE, "Train input shape mismatch"
    assert x_test.shape[1:] == INPUT_SHAPE, "Test input shape mismatch"
    assert x_train.dtype == np.float32, "Train data not converted to float32"
    assert x_train.max() <= 1.0 and x_train.min() >= 0.0, "Train data not properly normalized"

def test_config_paths():
    from config.config import BASE_DIR, DATA_RAW_DIR, MODELS_DIR
    assert isinstance(BASE_DIR, str)
    assert 'data' in DATA_RAW_DIR
    assert 'models' in MODELS_DIR
