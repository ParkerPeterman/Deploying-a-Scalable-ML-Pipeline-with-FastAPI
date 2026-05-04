import pytest
import numpy as np
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics

#
@pytest.fixture
def data():
    """ Simple fixture to provide a sample dataframe for testing. """
    df = pd.DataFrame({
        "age": [39, 50, 38],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private"],
        "education": ["Bachelors", "Bachelors", "HS-grad"],
        "salary": ["<=50K", "<=50K", "<=50K"]
    })
    return df

def test_process_data_output_types(data):
    """
    Test if the process_data function returns the expected numpy array types 
    for features and labels.
    """
    cat_features = ["workclass", "education"]
    X, y, encoder, lb = process_data(
        data, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )
    
    assert isinstance(X, np.ndarray)[cite: 4]
    assert isinstance(y, np.ndarray)[cite: 4]
    assert X.shape[0] == data.shape[0][cite: 4]

def test_train_model_algorithm():
    """
    Test if the train_model function returns the expected Random Forest algorithm 
    type as implemented in model.py.
    """
    # Create dummy data
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    
    model = train_model(X_train, y_train)
    
    # Check if the model has the 'predict' method and is a classifier
    assert hasattr(model, "predict")[cite: 4]
    assert "RandomForestClassifier" in str(type(model))[cite: 4]

def test_compute_metrics_values():
    """
    Test if compute_model_metrics returns correct values for a perfect prediction scenario. 
    In a perfect match, Precision, Recall, and F1 should all be 1.0.[cite: 4]
    """
    y_true = np.array([0, 1, 0, 1])
    y_preds = np.array([0, 1, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    
    assert precision == 1.0[cite: 4]
    assert recall == 1.0[cite: 4]
    assert fbeta == 1.0[cite: 4]