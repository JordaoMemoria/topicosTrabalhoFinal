import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcess:
    def __init__(self, kernel):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.kernel = kernel
        self.model = None
        self.optimizer = None

    def fit(self,  train_x, train_y, training_iter=5000):
        train_x = torch.tensor(train_x)
        train_x = train_x.to(self.device)
        train_y = torch.tensor(train_y)
        train_y = train_y.to(self.device)
        self.model = ExactGPModel(train_x*1.0, train_y*1.0, self.likelihood, self.kernel)
        self.model.train()
        self.likelihood.train()
        self.likelihood.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        mll.to(self.device)
        error = []
        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self.model(train_x * 1.0)
            loss = -mll(output, train_y * 1.0)
            loss.backward()
            error.append(loss.item())
            print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.model.likelihood.noise.item()
            ))
            try:
                print('lengthscale Iter: ', str(i + 1))
                print(self.model.covar_module.base_kernel.lengthscale.tolist()[0])
            except:
                pass
            self.optimizer.step()

    def predict(self, x_test):
        x_test = torch.tensor(x_test)
        x_test = x_test.to(self.device)
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x_test * 1.0))
        # means = np.array(observed_pred.mean.to("cpu").tolist())
        means = observed_pred.mean
        # vars = np.array(observed_pred.variance.to("cpu").tolist())
        vars = observed_pred.variance
        return means, vars

    def rmse(self, means_hat, y_test):
        y_test = torch.tensor(y_test)
        y_test = y_test.to(self.device)
        means_hat.to(self.device)
        criterion = torch.nn.MSELoss()
        return torch.sqrt(criterion(y_test, means_hat))