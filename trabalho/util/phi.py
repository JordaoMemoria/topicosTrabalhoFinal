import pandas as pd
import numpy as np


def rbf(x, c, l):
    return np.exp(-((x-c)**2)/(2*l))


def phi(x, mean_sample, var_sample):
    phi_x = np.array([1] + [rbf(x[i], mean_sample[i], var_sample[i]) for i in range(len(x))])
    return phi_x
