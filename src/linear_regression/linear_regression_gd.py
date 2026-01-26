import numpy as np
from utils import mse

class LinearRegressionGD:
    def __init__(self):
        self.coef_ = None 
        self.intercept_ = None 
    
    def fit(self, X, y, eta=0.1, epochs=100000, tolerance=1e-6):
        """
        Find the optimal values of the model parameters.
        Applied method: Bacth Gradient Descent.
        X: the training dataset features.
        y: the target variable.     
        """
        # initialize theta with values between 0 and 1
        theta = np.random.rand((X.shape[1] + 1))
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]
        
        # make sur X is a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        m = len(X) # m is the number of training instances
        dummy_feature = np.ones((m, 1))
        X = np.concatenate([dummy_feature, X], axis=1)
                
        for epoch in range(epochs):
            gradients = 2/m * X.T @ (X @ theta - y)
            if np.linalg.norm(gradients) < tolerance:
                break 
            else:
                theta = theta - eta * gradients
            
            if epoch % 50 == 0:
                y_hat = X @ theta
                print(f"Epoch {epoch} MSE: {mse(y_hat, y)}")
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]
    
    def predict(self, X):
        """
        Predicts the target variable for X.
        X: Array containing the instances and there features.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1) # make sur X is a 2D array
            
        y_hat = X @ self.coef_ + self.intercept_
        return y_hat