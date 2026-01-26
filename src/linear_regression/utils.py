
import numpy as np

def mse(y_hat, y):
    """
    Compute the Mean Squared Error
    
    :param y_hat: Array of the predicted values
    :param y: Array of the actual values
    """
    m = len(y)
    ms_error = round((1/m) * np.sum((y_hat - y) ** 2), 3)
    return ms_error
    
    