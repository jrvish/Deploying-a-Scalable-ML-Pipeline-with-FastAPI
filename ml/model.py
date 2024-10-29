import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import
from sklearn.linear_model import LogisticRegression

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
   # TODO: implement the function - I ws to use Logistic Regression*DONE
    model = LogisticRegression()

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l1', 'l2'],        # Regularization type
        'solver': ['liblinear']         # Solver for logistic regression
    }

    # Set up the GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='f1', cv=5, verbose=1, n_jobs=-1)

    # Fit the model with hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Returning the best trained model to improve metric results
    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # TODO: implement the function *DONE
    preds = model.predict(X)
    return preds

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # TODO: implement the function *DONE
    with open(path,'wb') as file:
        pickle.dump(model,file) #serializes the oject to be dumped into this file
    

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    # TODO: implement the function *DONE
    with open(path, 'rb') as file: #the rb stands for read binary - to read the binary data from the file
        return pickle.load(file)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # TODO: implement the function *DONE
    # Slice the data
    sliced_data = data[data[column_name] == slice_value] #Only slicing data from "column_name" column

    X_slice, y_slice, _, _ = process_data(
        sliced_data,
        categorical_features = categorical_features,
        label = "salary",
        training = False,
        encoder = encoder,
        lb = lb

        # for input data, use data in column given as "column_name", with the slice_value 
        # use training = False
    )
    preds = inference(model, X_slice) #Gets prediction on X_slice using the inference function
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
