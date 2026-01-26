import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Find the optimal values of the model parameters.
        Applied method: The Normal Equation (closed-form solution).
        X: the training dataset features.
        y: the target variable.     
        """
        m = X.shape[0] # m is the number of training instances
        # add the dummy feature vector, the first column of the X matrix must be equal to 1
        dummy_feature = np.ones((m, 1)) 
        if X.ndim == 1:
            X = X.reshape(-1, 1) # to not get an erro while applying np.concatenate()
        X = np.concatenate([dummy_feature, X], axis=1)
        
        # apply the normal equation
        theta = np.linalg.pinv(X) @ y
        
        # set the model parameters to the optimal values
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