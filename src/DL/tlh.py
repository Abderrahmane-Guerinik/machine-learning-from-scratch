import numpy as np

def heaviside(x):
    return 1 if x >= 0 else 0

def sgn(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

class ThresholdLogicUnit:
    """ 
    Implements a TLU
    step_function: The step function (heavisider or sgn)
    """
    def __init__(self, step_function="heaviside", n = 1):
        self.w = np.random.rand(n)
        self.b = np.random.rand()
        self.step_function = step_function
    
    def forward(self, x):
        z = np.dot(self.w, x) + self.b
        if self.step_function == "heaviside":
            s = heaviside(z)
        elif self.step_function == "sgn":
            s = sgn(z)
        else:
            raise ValueError("step_function must be 'heaviside' or 'sgn'")
        return s