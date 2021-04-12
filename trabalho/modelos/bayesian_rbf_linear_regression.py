import numpy as np
from trabalho.util.phi import phi


class BayesianRBFLinearRegression:

    def __init__(self, m0=None, S0=None, sig2_error=0, mean_rbf_arbitrario=None, lambda_rbf_arbitrario=None):
        self.m0 = m0
        self.S0 = S0
        self.sig2_error = sig2_error
        self.mu_post = None
        self.sig_post = None
        self.mean_rbf = None
        self.lambda_rbf = None
        self.mean_rbf_arbitrario = mean_rbf_arbitrario
        self.lambda_rbf_arbitrario = lambda_rbf_arbitrario

    def phi(self, X):
        return np.array([phi(x, self.mean_rbf, self.lambda_rbf) for x in X])
        
    def fit(self, X_input, Y):
        Y = Y.flatten()
        N, D = X_input.shape
        n_features = D + 1

        if self.mean_rbf is None:
            self.mean_rbf = X_input.mean(axis=0) + X_input.mean(axis=0) * self.mean_rbf_arbitrario

        if self.lambda_rbf is None:
            self.lambda_rbf = np.ones(n_features) * self.lambda_rbf_arbitrario
        
        X = self.phi(X_input)

        if self.m0 is None:
            self.m0 = X.mean(axis=0)
        if self.S0 is None:
            self.S0 = np.identity(n_features)

        aux_1 = np.linalg.inv(np.dot(np.dot(self.S0, X.T), X) + np.identity(n_features) * self.sig2_error)
        aux_2 = np.dot(self.S0, X.T)
        aux_3 = Y - np.dot(X, self.m0)
        aux_4 = np.dot(np.dot(aux_2, X), self.S0)
        self.mu_post = self.m0 + np.dot(np.dot(aux_1, aux_2), aux_3)
        self.sig_post = self.S0 - np.dot(aux_1, aux_4)

    def predict(self, X):
        X_phi = self.phi(X)
        y_mean = X_phi @ self.mu_post
        
        N, D = X.shape
        std = np.zeros((N, 1))
        
        for i, x in enumerate(X_phi):
            std[i] = 2 * np.sqrt(x @ self.sig_post @ x.T + self.sig2_error)

        return y_mean.reshape((-1, 1)), std

    def set_parameters_rbf(self, mean_rbf, lambda_rbf):
        self.mean_rbf = mean_rbf
        self.lambda_rbf = lambda_rbf
