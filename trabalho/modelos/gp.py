import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary


class GP:
    def __init__(self, kernel):
        self.kernel = kernel
        self.modelo = None

    def fit(self, X, Y):
        self.modelo = gpflow.models.GPR(data=(X, Y), kernel=self.kernel, mean_function=None)
        
        #print_summary(self.kernel)
        #print_summary(self.modelo)

    def predict(self, X):
        return self.modelo.predict_f(X)
