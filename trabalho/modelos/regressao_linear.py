from sklearn.linear_model import LinearRegression
import numpy as np


class RegressaoLinear:
    def __init__(self):
        self.modelo = LinearRegression()

    def fit(self, X, y):
        return self.modelo.fit(X, y)

    def predict(self, X):
        means = self.modelo.predict(X)
        vars = np.zeros(means.shape)
        return means, vars
