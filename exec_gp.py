from trabalho.util.metrica import nlpd, rmse, mape
from trabalho.modelos.gaussian_process import GaussianProcess

import torch
import gpytorch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

data_df = pd.read_csv('housesToRentProcessed_train_full.csv.csv', index_col=0)
data_test = pd.read_csv('housesToRentProcessed_test_full.csv', index_col=0)
data = data_df.values
data_test = data_test.values

N = len(data)

X_train = data[:, :-1]
y_train = data[:, -1]

X_test = data_test[:, :-1]
y_test = data_test[:, -1].reshape(-1, 1)

X_scaler = Scaler()
y_scaler = Scaler()

X_train_transform = X_scaler.fit_transform(X_train)
y_train_transform = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_train_transform = y_train_transform.reshape(1, -1)[0]
X_test_transform = X_scaler.fit_transform(X_test)

kernels = [
           gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=X_train.shape[1]),
           gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=X_train.shape[1]),
           gpytorch.kernels.RBFKernel(ard_num_dims=X_train.shape[1])]

rmse_erro = []
nlpd_erro = []
mape_erro = []
for kernel in kernels:
    gp = GaussianProcess(kernel)
    gp.fit(X_train_transform, y_train_transform, training_iter=2000)
    means, vars = gp.predict(X_test_transform)

    means = np.array(means.tolist()).reshape((-1, 1))
    vars = np.array(vars.tolist()).reshape((-1, 1))

    means_dimensao_correta = y_scaler.inverse_transform(means)
    std_errors_modelo = np.sqrt(vars)
    std_errors_scaler = np.sqrt(y_scaler.var_)
    vars_dimensao_correta = (std_errors_modelo * std_errors_scaler) ** 2

    rmse_erro.append(rmse(means_dimensao_correta, y_test))

    nlpd_erro.append(nlpd(means_dimensao_correta, vars_dimensao_correta, y_test))

    mape_erro.append(mape(means_dimensao_correta, y_test))
