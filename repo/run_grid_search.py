from data import DataGeneratorZ1D
from trainer import GridSearchTrainer
import torch
from torch import nn

c_u = 0.8
c_s = 0.4
N = 2
n_samples = 10000

DG = DataGeneratorZ1D(c_u, c_s, N, n_samples)
X, Z, Y = DG.generate()
population_list = torch.Tensor([*range(N)])

line_samples = 30
#tau = 10
loss = nn.BCELoss(reduction='none')
for tau in [1, 10, 100, 500, 1000, 3500, 10000, 350000]:
    trainer = GridSearchTrainer(X, Z, Y, population_list, c_u, tau, loss, line_samples)
    print(trainer.train(0))
