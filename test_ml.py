import pytest
import pandas as pd
import numpy as np
import os
from ml import data, model
from sklearn.linear_model import LogisticRegression
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_one():

    #Here is my descrption

    """
    # add description for the first test
    """
    # Your code here
    # Return the expected type of result
   
    # Your code here
    X_train = np.random.rand(100, 25)
    y_train = np.random.randint(2, size=100)

    #Training the model
    model_output = model.train_model(X_train, y_train)

    #Check model type
    assert isinstance(model_output, LogisticRegression)


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
