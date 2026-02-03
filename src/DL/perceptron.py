import numpy as np

def heaviside(x):
    return (x>=0).astyp(int)

class Perceptron:
    def __init__(self):
        pass 
    
    def fit(self, X, y):
        num_classes = len(np.unique(y))
        self.W = np.zeros((X.shape[1], num_classes))
        self.b = np.zeros(num_classes)
    
    def predict(self, X):
       Z = X @ self.W + self.b
       S = heaviside(Z)
       return np.argmax(S, axis=1)
       