import numpy as np
import torch
from torch import sigmoid
from torch.utils.data import DataLoader


class StrategicTester:
    def __init__(self, X, Z, Y, loss, population_list: torch.Tensor, c_u):
        self.X = X
        self.Z = Y
        self.Y = Z
        self.loss = loss
        self.population_list = population_list
        self.c_u = c_u
        self.cp_map = {}

    def conditional_precision(self, Ytrain_pred, Y_train, X_train, x):
        # print(Ypred.shape, Y.shape, X.shape)
        numerator = (Y_train[X_train[x] == 1] * Ytrain_pred[X_train[x] == 1]).sum()
        denominator = Ytrain_pred[X_train[x] == 1].sum()
        if denominator == 0:
            # print("You are racist! An entire group is labled 0")
            return -0.01
        return numerator / denominator

    def calc_all_conditional_precisions(self, Ytrain_pred, Ytrain, Xtrain, Yval, Xval):
        res = torch.zeros_like(Yval).type(torch.float)
        for x in self.population_list:
            x = int(x.item())
            cp = self.conditional_precision(Ytrain_pred, Ytrain, Xtrain, x)
            res[Xval[x] == 1] = cp
            self.cp_map[x] = cp
        return res

    def calc_applications(self, conditional_precisions):
        applications = (conditional_precisions - self.c_u > 0).type(torch.int)
        return applications

    def calc_normalization_term(self, applications):
        return applications.type(torch.float).mean()

    def calc_epoch_loss(self, normalization_term, applications, losses):
        if normalization_term == 0:
            # print("You are pitiful! No one wants to apply")
            return torch.Tensor([float("Inf")])
        return (1 / normalization_term) * ((applications * losses).mean(dtype=torch.float))
