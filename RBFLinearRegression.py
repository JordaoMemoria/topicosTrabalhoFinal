import numpy as np
import utils


class RBFLinearRegression:

    def __init__(self, m0, S0, sig2_error):
        self.m0 = m0
        self.S0 = S0
        self.sig2_error = sig2_error
        self.mu_post = None
        self.sig_post = None
        self.mean_rbf = None
        self.lambda_rbf = None

    def fit(self, X_input, Y):
        if not self.mean_rbf:
            self.mean_rbf = X_input.mean(axis=0) + np.array([x * 0.35 for x in X_input.mean(axis=0)])
            self.lambda_rbf = [0.1] * self.n_features
        X = np.array([utils.phi(x, self.mean_rbf, self.lambda_rbf) for x in X_input])
        n_features = X.shape[1]
        aux_1 = np.linalg.inv(np.dot(np.dot(self.S0, X.T), X) + np.identity(n_features) * self.sig2_error)
        aux_2 = np.dot(self.S0, X.T)
        aux_3 = Y - np.dot(X, self.m0)
        aux_4 = np.dot(np.dot(aux_2, X), self.S0)
        self.mu_post = self.m0 + np.dot(np.dot(aux_1, aux_2), aux_3)
        self.sig_post = self.S0 - np.dot(aux_1, aux_4)

    def predict(self, X):
        X_phi = np.array([utils.phi(x, self.mean_rbf, self.lambda_rbf) for x in X])
        y_mean = X_phi.dot(self.mu_post)
        std = []
        for x in X_phi:
            std.append(2 * np.sqrt(x.dot(self.sig_post).dot(x.T) + self.sig2_error))
        return y_mean, std

    def set_parameters_rbf(self, mean_rbf, lambda_rbf):
        self.mean_rbf = mean_rbf
        self.lambda_rbf = lambda_rbf
