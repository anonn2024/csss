import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


class DataContainer:

    def __init__(self, Xtrain, Xval, Xtest, Ztrain, Zval, Ztest,  Ytrain, Yval, Ytest):
        self.Xtrain = Xtrain
        self.Xval = Xval
        self.Xtest = Xtest
        self.Ztrain = Ztrain
        self.Zval = Zval
        self.Ztest = Ztest
        self.Ytrain = Ytrain
        self.Yval = Yval
        self.Ytest = Ytest
        self.population_size = self.Xtrain.shape[0]

    def get_stats(self, print_out=False):
        res = {'base_rate': {}, 'size': {}}
        self.Ytrain = self.Ytrain.squeeze()
        for x in range(self.population_size):
            Ytrain_x = self.Ytrain[self.Xtrain[x] == 1]
            group_size = Ytrain_x.shape[0]
            group_base_rate = Ytrain_x.sum() / group_size
            if print_out:
                print(f"Base rate for group {x} is: {group_base_rate.item()}")
                print(f"group size is {group_size}")
            res['base_rate'][x] = group_base_rate.item()
            res["size"][x] = group_size
        total_size = self.Ytrain.shape[0]
        total_base_rate = self.Ytrain.sum() / total_size
        res['base_rate']["total"] = total_base_rate.item()
        res["size"]['total'] = total_size
        if print_out:
            print(f"Total base rate is: {total_base_rate.item()}")
        return res


def splid_data(X, Z, Y, n_samples, train_ratio=0.7, seed=42):
    generator = torch.Generator().manual_seed(seed)

    train_samples = int(n_samples * train_ratio)
    val_samples = n_samples - train_samples
    X = X.t()
    Z = Z.t()
    Y = Y.t()
    Xtrain, Xval = random_split(X, [train_samples, val_samples], generator=generator)
    train_indices = Xtrain.indices
    val_indices = Xval.indices
    Xtrain = Xtrain.dataset[train_indices].t()
    Xval = Xval.dataset[val_indices].t()
    Ztrain = Z[train_indices].t()
    Zval = Z[val_indices].t()
    Ytrain = Y[train_indices].t()
    Yval = Y[val_indices].t()

    return DataContainer(Xtrain, Xval, None, Ztrain, Zval, None,  Ytrain, Yval, None)


class GaussianMixtureDistribution:

    def __init__(self, pos_prob, pos_mu, pos_std, neg_mu, neg_std):
        self.pos_prob = pos_prob
        self.pos_mu = pos_mu
        self.pos_std = pos_std
        self.neg_mu = neg_mu
        self.neg_std = neg_std

    def sample(self, n):
        y = np.random.binomial(1, self.pos_prob, n)
        x = (y == 1) * np.random.normal(self.pos_mu, self.pos_std, n) + (y == 0) * np.random.normal(self.neg_mu,
                                                                                                    self.neg_std, n)
        return x, y

    def sample_conditioned_on_y(self, n, y):
        x = (y == 1) * np.random.normal(self.pos_mu, self.pos_std, n) + (y == 0) * np.random.normal(self.neg_mu,
                                                                                                    self.neg_std, n)
        return x


# Generate data with one-dimensional Z
class DataGeneratorZ1D:
    def __init__(self, c_u, c_s, N, n_samples):
        self.c_u = c_u
        self.c_s = c_s
        self.N = N
        self.n_samples = n_samples

    def generate(self, plot=False):
        X_vals = [*range(self.N)]

        x_list, z_list, y_list = [], [], []
        for x in X_vals:
            # create random distribution
            pos_prob = np.random.random()
            pos_mu = np.random.uniform(0.5, 1)
            pos_std = np.random.uniform(0.2, 0.5)
            neg_mu = np.random.uniform(0, 0.5)
            neg_std = np.random.uniform(0.2, 0.5)
            dist = GaussianMixtureDistribution(pos_prob, pos_mu, pos_std, neg_mu, neg_std)
            # sample
            cur_n_samples = self.n_samples // self.N
            cur_X = np.ones(cur_n_samples) * x
            cur_Z, cur_Y = dist.sample(cur_n_samples)
            x_list.append(cur_X)
            z_list.append(cur_Z)
            y_list.append(cur_Y)

        X = np.concatenate(x_list)
        Z = np.concatenate(z_list)
        Y = np.concatenate(y_list)

        if plot:
            data = pd.DataFrame({"x": X, "z": Z, "y": Y, "row": X // 3, "col": X % 3})
            g = sns.displot(data, x="z", hue="y", col="col", row="row", kind="kde", height=2)
            axes = g.axes.flat
            for ax, x in zip(axes, X_vals):
                ax.set_title(f"x={x}")
            for (i, j, k), data in g.facet_data():
                if data.empty:
                    ax = g.facet_axis(i, j)
                    ax.set_title("")
                    ax.set_axis_off()

            plt.show()

        X = torch.from_numpy(X)
        X = torch.nn.functional.one_hot(X.type(torch.long)).T
        Z = torch.unsqueeze(torch.from_numpy(Z), 0)
        Y = torch.unsqueeze(torch.from_numpy(Y), 0)

        return X, Z, Y


# Generate data with two-dimensional Z
class DataGeneratorZ2D:
    def __init__(self, c_u, c_s, N, n_samples):
        self.c_u = c_u
        self.c_s = c_s
        self.N = N
        self.n_samples = n_samples

    def generate(self, plot=False):
        X_vals = [*range(self.N)]

        x_list, z0_list, z1_list, y_list = [], [], [], []
        for x in X_vals:
            # Z0
            pos_prob = np.random.uniform(0.2, 0.6)
            pos_mu = np.random.uniform(0.55, 1)
            pos_std = np.random.uniform(0.2, 0.5)
            neg_mu = np.random.uniform(0, 0.45)
            neg_std = np.random.uniform(0.1, 0.3)
            z0_dist = GaussianMixtureDistribution(pos_prob, pos_mu, pos_std, neg_mu, neg_std)
            # Z1
            pos_prob = np.random.uniform(0.2, 0.6)
            pos_mu = np.random.uniform(0.55, 1)
            pos_std = np.random.uniform(0.2, 0.5)
            neg_mu = np.random.uniform(0, 0.45)
            neg_std = np.random.uniform(0.1, 0.3)
            z1_dist = GaussianMixtureDistribution(pos_prob, pos_mu, pos_std, neg_mu, neg_std)

            # sample
            cur_n_samples = self.n_samples // self.N
            cur_X = np.ones(cur_n_samples) * x
            cur_z0, cur_y = z0_dist.sample(cur_n_samples)
            cur_z1 = z1_dist.sample_conditioned_on_y(cur_n_samples, cur_y)

            x_list.append(cur_X)
            z0_list.append(cur_z0)
            z1_list.append(cur_z1)
            y_list.append(cur_y)

        X = np.concatenate(x_list)
        Z0 = np.concatenate(z0_list)
        Z1 = np.concatenate(z1_list)
        Z = np.concatenate((Z0.reshape(-1, 1), Z1.reshape(-1, 1)), axis=1)
        Y = np.concatenate(y_list)

        if plot:
            data = pd.DataFrame({"x": X, "z0": Z0, "z1": Z1, "y": Y, "row": X // 3, "col": X % 3})
            g = sns.relplot(data, x="z0", y="z1", hue="y", col="col", row="row", height=2, s=10)
            axes = g.axes.flat
            for ax, x in zip(axes, X_vals):
                ax.set_title(f"x={x}")
            for (i, j, k), data in g.facet_data():
                if data.empty:
                    ax = g.facet_axis(i, j)
                    ax.set_title("")
                    ax.set_axis_off()

            plt.show()

        X = torch.from_numpy(X)
        X = torch.nn.functional.one_hot(X.type(torch.long)).T
        Z = torch.from_numpy(Z).T
        Y = torch.unsqueeze(torch.from_numpy(Y), 0)

        return X, Z, Y


def get_train_and_val_datasets(data_generator, train_ratio=0.7, seed=42):
    X, Z, Y = data_generator.generate()
    n_samples = X.shape[1]
    return splid_data(X, Z, Y, n_samples, train_ratio=train_ratio, seed=seed)
