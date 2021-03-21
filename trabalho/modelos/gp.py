import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary
import numpy as np


class GP:
    def __init__(self, kernel, mean_function):
        self.kernel = kernel
        self.mean_function = mean_function
        self.modelo = None

    def fit(self, X, Y):
        self.modelo = gpflow.models.GPR(data=(X, Y), kernel=self.kernel, mean_function=self.mean_function)
        
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(self.modelo.training_loss, self.modelo.trainable_variables, options=dict(maxiter=100), method='BFGS')
        #print(opt_logs)

        #print_summary(self.kernel)
        #print_summary(self.modelo)
        print_summary(self.modelo, fmt="notebook")

    def predict(self, X):
        means, vars = self.modelo.predict_y(X)
        
        return np.array(means), np.array(vars)

    def predict_f(self, X):
        means, vars = self.modelo.predict_f(X)
        
        return np.array(means), np.array(vars)
