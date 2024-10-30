import pytest
import pandas as pd
import numpy as np
import os
from ml import data, model
from sklearn.linear_model import LogisticRegression
# TODO: add necessary import
from ml.model import (
    compute_model_metrics,  # Import the function you want to test
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)



# TODO: implement the first test. Change the function name and input as needed **Done
def test_compute_model_metrics():
    """ This first test is to ensure the compute_model_metrics function is returning the expected types"""
    # Sample true labels and predictions
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    
    # Call the function
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    
    # Check if the return types are correct
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)

# TODO: implement the second test. Change the function name and input as needed
def test_dataset_col_number(test_data_path):
    """
    # Test to make sure there are 15 columns
    """
    # Your code here
    assert test_data_path.shape[1] == 15

# TODO: implement the third test. Change the function name and input as needed

@pytest.fixture(scope="session") #see train_model.py
def test_data_path():
    # Constructing the path to test data
    test_path = os.getcwd()
    test_data_path = os.path.join(test_path, "data", "census.csv")
    df = pd.read_csv(test_data_path) # your code here
    return df

def test_train_model():
    """
   # To check for the correct model type
    """
    # Your code here
    X_train = np.random.rand(100, 25)
    y_train = np.random.randint(2, size=100)

    #Training the model
    model_output = model.train_model(X_train, y_train)

    #Check model type
    assert isinstance(model_output, LogisticRegression)