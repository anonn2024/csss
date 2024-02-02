from data import DataGeneratorZ1D, DataGeneratorZ2D
from trainer import StrategicTrainer
from model import LinearRegressionModel
import torch
from torch import nn
from torch.utils.data import random_split
from torch.optim import SGD

c_u = 0.7
c_s = 0.4
N = 4
n_samples = 10000


train_samples = int(n_samples * 0.7)
val_samples = n_samples - train_samples

DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
X, Z, Y = DG.generate()
X = X.t()
Z = Z.t()
Y = Y.t()
Xtrain, Xval = random_split(X, [train_samples, val_samples])
Ztrain, Zval = random_split(Z, [train_samples, val_samples])
Ytrain, Yval = random_split(Y, [train_samples, val_samples])
Xtrain = Xtrain.dataset.t()
Xval = Xval.dataset.t()
Ztrain = Ztrain.dataset.t()
Zval = Zval.dataset.t()
Ytrain = Ytrain.dataset.t()
Yval = Yval.dataset.t()

population_list = torch.Tensor([*range(N)])

tau = 20
model = LinearRegressionModel(Xtrain, Ztrain)
loss = nn.BCELoss(reduction='none')
opt = SGD(model.parameters(), lr=0.01)
trainer = StrategicTrainer(model, Xtrain, Ztrain, Ytrain, Xval, Zval, Yval, None, None, None, opt, loss, population_list, tau, c_u)

trainer.train(200, early_stop=25)