import numpy as np
from BayesianLinearRegression import BayesianLinearRegression
import pandas as pd

class BayesianPolinomialLinearRegression(BayesianLinearRegression):
    
    def __init__(self, alphaInit, lambdaInit, order):
        super().__init__(alphaInit, lambdaInit)
        self.order = order
    
    def fit(self, X_train, y_train):
        X_train = self.updateToPolinomial(X_train)
        super().fit(X_train, y_train)
    
    def predict(self, X_test):
        X_test = self.updateToPolinomial(X_test)
        return super().predict(X_test)
        
    def updateToPolinomial(self, X):
        X_df = pd.DataFrame(X)
        for c in X_df.columns.tolist():
            columnToPol = X_df[c].astype(np.float)
#             columnToPol = X_df[c]
            for i in range(self.order - 1):
                X_df[str(c) + str(i)] = columnToPol.apply(lambda x: pow(x,i+2))

        return X_df.values