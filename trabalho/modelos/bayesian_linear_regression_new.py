import numpy as np


def Φ_vies(X):
    return np.insert(X, 0, np.ones(X.shape[0]), axis=1)

    
class BayesianLinearRegressionNew:
    def __init__(self, m0=None, S0=None, sig2_error=0):
        self.m0 = m0
        self.S0 = S0
        self.sig2_error = sig2_error
        
        self.mu_post = None
        self.sig_post = None

    def phi(self, X, **kwargs):
        return Φ_vies(X)

    def fit(self, X_input, Y):
        Y = Y.flatten()
        X = self.phi(X_input)
        
        N, D = X.shape
        
        if self.m0 is None:
            self.m0 = X.mean(axis=0)
        if self.S0 is None:
            self.S0 = np.identity(D)

        aux_1 = np.linalg.inv(self.S0 @ X.T @ X + np.identity(D) * self.sig2_error)
        aux_2 = self.S0 @ X.T
        aux_3 = Y - X @ self.m0
        aux_4 = aux_2 @ X @ self.S0
        
        self.mu_post = self.m0 + aux_1 @ aux_2 @ aux_3
        self.sig_post = self.S0 - aux_1 @ aux_4

    def predict(self, X):
        X_phi = self.phi(X)
        y_mean = X_phi @ self.mu_post
        
        N, D = X.shape
        std = np.zeros((N, 1))
        
        for i, x in enumerate(X_phi):
            std[i] = 2 * np.sqrt(x @ self.sig_post @ x.T + self.sig2_error)

        return y_mean.reshape((-1, 1)), std