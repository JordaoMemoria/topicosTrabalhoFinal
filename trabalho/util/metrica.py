from sklearn.metrics import mean_squared_error
from scipy import stats


def rmse(a, b):
    return mean_squared_error(a, b, squared=False)


def nlpd(predicao_mean, predicao_std, validacao):
    """
    Basedo em
    https://github.com/learningtitans/bayesian_uncertainty/blob/7b6ccce0f78a/src/bayesian_uncertainty/metrics.py
    """
    EPSILON = 1e-30

    predicao_std = predicao_std.copy()
    predicao_std[predicao_std < EPSILON] = EPSILON
    return -stats.norm(predicao_mean, predicao_std).logpdf(validacao).mean()