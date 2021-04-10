import numpy as np
from sklearn.linear_model import BayesianRidge

class BayesianLinearRegression:
    
    def __init__(self, alphaInit=1., lambdaInit=0.2):
        self.alphaInit = alphaInit
        self.lambdaInit = lambdaInit
        self.clf = BayesianRidge(fit_intercept=False)
        self.clf.set_params(
            alpha_init=self.alphaInit, 
            lambda_init=self.lambdaInit
        )
    
    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.clf.predict(X_test.astype(np.float32), return_std=True)