import numpy as np
import pandas as pd

from trabalho.modelos.bayesian_linear_regression_new import BayesianLinearRegressionNew


def Φ_polinomial(X, P):
    X_df = pd.DataFrame(X)
    for c in X_df.columns.tolist():
        columnToPol = X_df[c].astype(np.float64)
        for i in range(int(P) - 1):
            X_df[str(c) + str(i)] = columnToPol.apply(lambda x: pow(x,i+2))

    return X_df.values


class BayesianPolinomialLinearRegressionNew(BayesianLinearRegressionNew):

    def __init__(self, m0=None, S0=None, sig2_error=0, order=2):
        super().__init__(m0, S0, sig2_error)

        self.order = order

    def phi(self, X, **kwargs):
        """
        Φ(x, P) = [1, x^1, x^2, ..., x^P]
        """
        return Φ_polinomial(X, self.order)
