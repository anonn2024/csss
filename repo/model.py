import torch
from torch import nn
from torch import sigmoid
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import set_seeds

SEED = 0
class LinearRegressionModel(nn.Module):
    def __init__(self, x: torch.Tensor or None, z: torch.tensor, tau=2, withXParams=True, std=1.0, seed=0):
        set_seeds(seed)
        super().__init__()
        param_size = x.shape[0] + z.shape[0] if withXParams else z.shape[0]
        # self.weights = nn.Parameter(torch.randn(param_size, dtype=torch.float), requires_grad=True)
        self.weights = nn.Parameter(torch.normal(mean=0.0, std=std, size=(param_size,), dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.tau = tau
        self.withXParams = withXParams

    def forward(self, x: torch.Tensor, z: torch.tensor, tau=None) -> torch.Tensor:
        if tau is None:
            tau = self.tau
        features = torch.concat((x, z), dim=0).type(torch.float) if self.withXParams else z.type(torch.float)
        return sigmoid(tau * (torch.matmul(self.weights, features) + self.bias))

    def set_parameter(self, index, value):
        params = list(self.parameters())
        with torch.no_grad():
            params[0][index].copy_(value)

    def get_parameter(self, index):
        params = list(self.parameters())
        return params[0][index]

    def set_bias(self, value, relative=False):
        if relative:
            value += self.bias.item()
        params = list(self.parameters())
        with torch.no_grad():
            params[1][0].copy_(value)

    def set_bias_from_threshold(self, t):
        sig_inverse = torch.log(t/(1-t))
        self.set_bias(sig_inverse, relative=True)


class LinearClassifier:
    def __init__(self, X_values, b=0.5, penalty=None, C=1):
        self.cls = LogisticRegression(penalty=penalty, solver="saga", C=C)
        self.b = b
        self.X_values = X_values

    def prepare_data(self, X, Z):
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        input = np.concatenate((X, Z), axis=0).T
        input_df = pd.DataFrame(input,
                                columns=[f"x_{x}" for x in range(X.shape[0])] + [f"z_{i}" for i in range(Z.shape[0])])
        return input_df

    def fit(self, X, Z, Y):
        train_input = self.prepare_data(X, Z)
        if len(Y.shape) == 2:
            Y = Y.squeeze()
        self.cls.fit(train_input, Y)

    def __call__(self, X, Z):
        if len(X) == 0:
            return np.array([])
        input = self.prepare_data(X, Z)
        probs = self.cls.predict_proba(input)
        return (probs[:, 1] > self.b).astype(int)

    def set_b(self, b):
        self.b = b


class GridSearchLinearRegressionModel(nn.Module):
    def __init__(self, x0_w, x1_w, bias, tau):
        super().__init__()
        z_w = torch.Tensor([1])
        w_wo_b = torch.Tensor((x0_w, x1_w, z_w))
        self.weights = nn.Parameter(w_wo_b, requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(bias), requires_grad=False)
        self.tau = tau

    def forward(self, x: torch.Tensor, z: torch.tensor) -> torch.Tensor:
        features = torch.concat((x, z), dim=0).type(torch.float)
        return sigmoid(self.tau * (torch.matmul(self.weights, features) + self.bias))
