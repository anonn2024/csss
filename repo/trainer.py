import numpy as np
import torch
from torch import sigmoid
from model import GridSearchLinearRegressionModel, LinearRegressionModel
from test import StrategicTester
import matplotlib.pyplot as plt
from utils import *
from data import DataContainer
import fairtorch


class StrategicTrainer:
    def __init__(self, model, data: DataContainer, opt, loss,
                 population_list: torch.Tensor, tau, c_u, c_s, penalty_scalar=6, penalty=False, alt_penalty=False,
                 cust_sig=True, precision_tau=None, app_bias=True, c_u_tolerance=0, fair=False, fairness_param=200, run_name=''):
        self.model = model
        self.data = data
        self.opt = opt
        self.loss = loss
        self.population_list = population_list
        self.c_u = c_u + c_u_tolerance
        self.c_s = c_s
        self.tau = tau
        self.cp_map = {}
        self.penalty = penalty
        self.penalty_scalar = penalty_scalar
        self.penalty_map = {}
        self.alt_penalty = alt_penalty
        self.app_bias = app_bias
        self.cust_sig = cust_sig
        self.precision_tau = precision_tau
        self.metric_manager = MetricManager(output_final_vals_name=run_name)
        self.tester = StrategicTester(self.data.Xval, self.data.Zval, self.data.Yval, self.loss, self.population_list,
                                      c_u)
        self.fair = fair
        self.fairness_map = {"smooth": {},
                             "zero_one": {}}
        self.fairness_param = fairness_param
        self.Xtrain_col = None

    @staticmethod
    def conditional_precision(scores, Y, X, x):
        numerator = (Y[X[x] == 1] * scores[X[x] == 1]).sum()
        denominator = scores[X[x] == 1].sum()
        if denominator == 0:
            return torch.zeros(1)
        return numerator / denominator

    def get_unified_Xtrain_col(self):
        if self.Xtrain_col is not None:
            return self.Xtrain_col

        num_examples = self.data.Xtrain.shape[1]
        res = torch.zeros(num_examples)
        for x in self.population_list:
            x = int(x.item())
            res[self.data.Xtrain[x] == 1] = x

        return res

    def calc_all_conditional_precisions(self, scores, Y, X):
        res = torch.zeros_like(Y).type(torch.float)
        for x in self.population_list:
            x = int(x.item())
            cp = self.conditional_precision(scores, Y, X, x)
            res[X[x] == 1] = cp
            self.cp_map[x] = cp.clone().detach()
        return res

    def calc_application_bias(self, scores, prev_scores, Y, X):
        with torch.no_grad():
            res = torch.zeros_like(Y).type(torch.float)
            for x in self.population_list:
                x = int(x.item())
                sum_ytilde = scores[X[x] == 1].sum()
                bias = ((Y[X[x] == 1] - self.c_u) * (
                        (prev_scores[X[x] == 1] > 0.5).type(torch.int) - scores[X[x] == 1])).sum() / sum_ytilde
                max_score = torch.max(scores[X[x] == 1])
                if max_score < 0.55:
                    res[X[x] == 1] = 0
                else:
                    res[X[x] == 1] = bias
            return res

    def calc_applications(self, conditional_precisions, bias=0):
        if self.cust_sig and self.c_u > 0:
            return zo_sigmoid((conditional_precisions + bias - self.c_u), c=self.c_u, tau=self.tau)
        return sigmoid(self.tau * (conditional_precisions - self.c_u))

    def calc_penalty(self, scores, applications, X):
        penalty_func = parabula_penalty if self.alt_penalty else log_penalty
        if not self.penalty:
            return 0, 0, 0
        penalty = 0
        undef_cp = False
        undef_app = False
        for x in self.population_list:
            x = int(x.item())
            max_score = torch.max(scores[X[x] == 1])
            if max_score < 0.55:
                undef_cp = True
            x_penalty = penalty_func(max_score, self.penalty_scalar)
            penalty += x_penalty
            self.penalty_map[x] = x_penalty.clone().detach()
        max_application = torch.max(applications)
        if max_application < 0.55:
            undef_app = True
        app_penalty = penalty_func(max_application, self.penalty_scalar)
        penalty += app_penalty
        self.penalty_map["app"] = app_penalty.clone().detach()
        return penalty, 1.2 if undef_cp else 0, 1.3 if undef_app else 0

    @staticmethod
    def calc_epoch_loss(applications, losses):
        normalization_term = applications.mean()
        if normalization_term == 0:
            return torch.Tensor([float("Inf")])
        return (1 / normalization_term) * ((applications * losses).mean())

    @staticmethod
    def calc_accuracy(applications: torch.Tensor, Y_pred: torch.Tensor, Y: torch.Tensor):
        return sum(applications * (Y_pred == Y)) / Y.size(dim=0)

    def calc_fairness_loss(self, scores):
        if not self.fair:
            return 0
        # X = self.get_unified_Xtrain_col()
        # fairloss = dp_loss(X, scores, X)
        # with torch.no_grad():
        fairloss = 0
        Y = scores
        mode = "smooth"
        e_total = Y.type(torch.float).mean().item()
        # gaps = torch.zeros_like()
        for x in self.population_list:
            x = int(x.item())
            # e_x_0 = Y[self.data.Xtrain[x] == 0].type(torch.float).mean().item()
            e_x_1 = Y[self.data.Xtrain[x] == 1].type(torch.float).mean()
            self.fairness_map[mode][x] = {"0": 0, "1": e_x_1.item()}
            fairloss += torch.pow(10*(e_x_1 - e_total), 2)
        self.fairness_map[mode]["total"] = e_total

        mode = "zero_one"
        with torch.no_grad():
            Y = (scores > 0.5).type(torch.int)
            e_total = Y.type(torch.float).mean().item()
            for x in self.population_list:
                x = int(x.item())
                # e_x_0 = Y[self.data.Xtrain[x] == 0].type(torch.float).mean().item()
                e_x_1 = Y[self.data.Xtrain[x] == 1].type(torch.float).mean()
                self.fairness_map[mode][x] = {"0": 0, "1": e_x_1.item()}
            self.fairness_map[mode]["total"] = e_total

        self.fairness_map["fairloss"] = fairloss
        return self.fairness_param * fairloss

    def calc_epoch_metrics(self, Y_pred, val_applications, train_applications, val_loss, train_loss, penalty,
                           train_epoch_loss,
                           val_epoch_loss,
                           undef_cp, undef_apps, bias, Ytrain_pred):
        self.metric_manager.calc_mean_apply(val_applications)
        # self.metric_manager.calc_system_utility(val_applications, self.data.Yval, Y_pred, self.c_s)
        self.metric_manager.calc_induced_accuracy(val_applications, val_loss)
        self.metric_manager.calc_train_induced_accuracy(Ytrain_pred, self.data.Ytrain, train_applications.clone().detach())
        self.metric_manager.calc_loss_per_group(self.population_list, self.data.Xtrain, train_applications.clone().detach(),
                                                train_loss.clone().detach(), smoothed=True)
        self.metric_manager.calc_loss_per_group(self.population_list, self.data.Xval, val_applications,
                                                val_loss)
        # self.metric_manager.calc_system_induced_utility(val_applications, self.data.Yval, Y_pred, self.c_s)
        # self.metric_manager.calc_mean_pass_screening(val_applications, self.data.Yval)
        self.metric_manager.update_model_weights(self.model, self.population_list, hasXParams=self.model.withXParams)
        self.metric_manager.calc_applications_per_group(train_applications.clone().detach(), self.data.Xtrain,
                                                        self.population_list, smoothed=True)
        self.metric_manager.calc_applications_per_group(val_applications, self.data.Xval,
                                                        self.population_list, smoothed=False)
        self.metric_manager.calc_accuracy_per_group(val_loss, self.data.Xval, self.population_list)
        self.metric_manager.calc_train_accuracy_per_group(self.data.Xtrain, self.data.Ytrain, Ytrain_pred, self.population_list)
        self.metric_manager.update_precision_per_group(self.population_list, self.cp_map, self.data.Xtrain, bias,
                                                       smoothed=True)
        self.metric_manager.update_precision_per_group(self.population_list, self.tester.cp_map)
        if self.app_bias:
            self.metric_manager.calc_bias_per_group(bias, self.data.Xtrain, self.population_list)
        if self.penalty:
            self.metric_manager.update_metric("penalty", penalty.clone().detach())
            self.metric_manager.calc_penalty_per_group(self.penalty_map, self.population_list)
        self.metric_manager.update_metric("train_loss", train_epoch_loss.clone().detach())
        self.metric_manager.update_metric("val_loss", val_epoch_loss)
        self.metric_manager.update_metric("undefined_cp", undef_cp)
        self.metric_manager.update_metric("no_applications", undef_apps)
        if self.fair:
            self.metric_manager.update_metric("fairloss", self.fairness_map["fairloss"].clone().detach())
            self.metric_manager.calc_fairness_constraint_per_group(self.fairness_map, self.population_list)

    def train(self, epochs: int, verbose: bool = False, early_stop: int = 30):
        best_val_loss = np.Inf
        val_epoch_loss = np.inf
        prev_scores = 0
        consecutive_no_improvement = 0
        epochs_run = 0

        # FAIRNESS
        # dp_loss = fairtorch.DemographicParityLoss(sensitive_classes=[0, 1], alpha=100)
        # criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            if epoch > 0 and epoch % 1000 == 0:
                print(f"epoch {epoch}", flush=True)
            epochs_run = epoch
            self.opt.zero_grad()
            scores = self.model(self.data.Xtrain, self.data.Ztrain)
            precision_scores = scores if self.precision_tau is None else self.model(self.data.Xtrain, self.data.Ztrain,
                                                                                    self.precision_tau)
            self.data.Ytrain = self.data.Ytrain.squeeze()
            loss = self.loss(scores, self.data.Ytrain.type(torch.float))
            cp = self.calc_all_conditional_precisions(precision_scores, self.data.Ytrain, self.data.Xtrain.squeeze())
            bias = torch.zeros_like(self.data.Ytrain) if (
                    (not self.app_bias) or epoch == 0) else self.calc_application_bias(
                precision_scores, prev_scores.squeeze(),
                self.data.Ytrain, self.data.Xtrain.squeeze())
            applications = self.calc_applications(cp, bias)
            penalty, undef_cp, undef_app = self.calc_penalty(scores, applications, self.data.Xtrain)
            clean_loss = self.calc_epoch_loss(applications, loss)

            fairloss = self.calc_fairness_loss(scores)
            epoch_loss = clean_loss + penalty + fairloss
            epoch_loss.backward()
            self.opt.step()
            prev_scores = scores.clone().detach()

            with torch.no_grad():
                Yval_scores = self.model(self.data.Xval, self.data.Zval)
                Yval_pred = (Yval_scores > 0.5).type(torch.int)
                val_loss = torch.ne(Yval_pred, self.data.Yval).type(torch.int)
                Ytrain_pred = (scores > 0.5).type(torch.int)
                val_cp = self.tester.calc_all_conditional_precisions(Ytrain_pred, self.data.Ytrain.squeeze(),
                                                                     self.data.Xtrain.squeeze(),
                                                                     self.data.Yval.squeeze(),
                                                                     self.data.Xval.squeeze())
                val_applications = self.tester.calc_applications(val_cp)
                val_norm = self.tester.calc_normalization_term(val_applications)
                val_epoch_loss = self.tester.calc_epoch_loss(val_norm, val_applications, val_loss)
                self.calc_epoch_metrics(Yval_pred, val_applications, applications, val_loss, loss, penalty,
                                        epoch_loss,
                                        val_epoch_loss, undef_cp, undef_app, bias, Ytrain_pred)
                if early_stop:
                    if val_epoch_loss < best_val_loss:
                        consecutive_no_improvement = 0
                        best_val_loss = val_epoch_loss

                    else:
                        consecutive_no_improvement += 1
                        if consecutive_no_improvement >= early_stop:
                            break
        name = "induced_accuracy"
        print(
            f"trained {epochs_run} epochs. Induced accuracy : {self.metric_manager.get_final_value(name)}")
        return self.metric_manager


class NonStrategicTrainer(StrategicTrainer):
    def __init__(self, model, data: DataContainer, opt, loss,
                 population_list: torch.Tensor, tau, c_s, metric_c_u=0.7, run_name=""):
        super().__init__(model, data, opt, loss,
                         population_list, tau, c_u=0, c_s=c_s, app_bias=False, run_name=run_name)
        self.strategic_metrics = {}
        self.metric_c_u = metric_c_u
        self.tester.c_u = metric_c_u

    def calc_all_conditional_precisions(self, scores, Y, X):
        self.strategic_metrics["cp"] = super().calc_all_conditional_precisions(scores, Y, X)
        res = torch.zeros_like(Y).type(torch.float)
        return res

    def calc_applications(self, conditional_precisions, bias=0):
        self.strategic_metrics["applications"] = zo_sigmoid((self.strategic_metrics["cp"] - self.metric_c_u),
                                                            c=self.metric_c_u, tau=self.tau)
        # self.strategic_metrics["applications"] = super().calc_applications(self.strategic_metrics["cp"])
        data_size = conditional_precisions.shape[0]
        return torch.full([data_size], 1 / data_size)

    def calc_epoch_metrics(self, Y_pred, val_applications, train_applications, val_loss, train_loss, penalty,
                           train_epoch_loss,
                           val_epoch_loss,
                           undef_cp, undef_apps, bias, Ytrain_pred):
        train_applications = self.strategic_metrics["applications"]
        super().calc_epoch_metrics(Y_pred, val_applications, train_applications, val_loss, train_loss, penalty,
                                   train_epoch_loss,
                                   val_epoch_loss,
                                   undef_cp, undef_apps, None, Ytrain_pred)


class GridSearchTrainer(StrategicTrainer):
    def __init__(self, data: DataContainer, population_list, c_u, tau, loss, n_samples):
        super().__init__(None, data, None, loss, population_list, tau, c_u, c_s=0)
        self.min_weight = -2
        self.max_weight = 2
        self.n_samples = n_samples
        self.tester = StrategicTester(data.Xtrain, data.Ztrain, data.Ytrain, None, population_list, c_u)

    def train(self, epochs: int, verbose: bool = False, early_stop: int = 7):
        x0_w_vals = torch.linspace(self.min_weight, self.max_weight, self.n_samples)
        x1_w_vals = torch.clone(x0_w_vals)
        bias_vals = torch.clone(x0_w_vals)

        min_loss = np.Inf
        min_loss_w = None
        min_loss_test_loss = 0

        min_test_loss = np.Inf
        min_test_loss_w = None
        min_test_loss_smooth_loss = 0

        for x0_w in x0_w_vals:
            for x1_w in x1_w_vals:
                for bias in bias_vals:
                    model = GridSearchLinearRegressionModel(x0_w, x1_w, bias, self.tau)
                    scores = model(self.data.Xtrain, self.data.Ztrain)
                    # Smoothed loss
                    self.data.Ytrain = self.data.Ytrain.squeeze()
                    loss = self.loss(scores, self.data.Ytrain.type(torch.float))
                    cp = self.calc_all_conditional_precisions(scores, self.data.Ytrain, self.data.Xtrain.squeeze())
                    applications = self.calc_applications(cp)
                    current_loss = self.calc_epoch_loss(applications, loss)
                    # Unsmoothed loss
                    Ypred = (scores > 0.5).type(torch.int)
                    test_cp = self.tester.calc_all_conditional_precisions(Ypred, self.data.Ytrain,
                                                                          self.data.Xtrain.squeeze())
                    test_applications = self.tester.calc_applications(test_cp)
                    test_norm = self.tester.calc_normalization_term(test_applications)
                    binary_loss = torch.ne(Ypred, self.data.Ytrain).type(torch.int)
                    test_current_loss = self.tester.calc_epoch_loss(test_norm, test_applications, binary_loss)

                    if current_loss <= min_loss:
                        min_loss = current_loss
                        min_loss_w = (x0_w, x1_w, bias)
                        min_loss_test_loss = test_current_loss

                    if test_current_loss <= min_test_loss:
                        min_test_loss = test_current_loss
                        min_test_loss_w = (x0_w, x1_w, bias)
                        min_test_loss_smooth_loss = current_loss

        print("--------------------------------------------------------------------------------")
        print(f"Min smooth loss is {min_loss} for weights {min_loss_w} and the real loss for"
              f" those weights is {min_loss_test_loss}")
        print("--------------------------------------------------------------------------------")
        print(f"Min real loss is {min_test_loss} for weights {min_test_loss_w} and the real loss for"
              f" those weights is {min_test_loss_smooth_loss}")
        print("--------------------------------------------------------------------------------")


class NonStrategicThresholdTrainer(NonStrategicTrainer):
    def __init__(self, model: LinearRegressionModel, data: DataContainer, opt, loss, population_list: torch.Tensor, tau,
                 c_s, prec_tau=2, c_u=0.7, c_u_tolerance=0, run_name=""):
        super().__init__(model, data, opt, loss, population_list, tau, c_s, metric_c_u=c_u, run_name=run_name)
        self.precision_tau = prec_tau
        self.losses = []
        self.thresholds = None
        self.threshold_metrics = MetricManager(xname="Threshold", output_final_vals_name=run_name)
        self.c_u_tolerance = c_u_tolerance

    # def plot_threshold_loss_tradeoff(self):
    #     assert self.thresholds is not None and len(self.thresholds) == len(self.losses)
    #     plt.plot(self.thresholds, self.losses)
    #     plt.title('Threshold Loss_Tradeoff')
    #     plt.xlabel('Threshold')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.show()

    def calc_threshold_metrics(self, train_applications, val_applications, train_loss,
                               val_loss, Ytrain_pred):
        self.threshold_metrics.calc_mean_apply(val_applications)
        self.threshold_metrics.calc_induced_accuracy(val_applications, val_loss)
        self.threshold_metrics.calc_induced_accuracy(train_applications, train_loss, "train_induced_accuracy")
        self.threshold_metrics.calc_loss_per_group(self.population_list, self.data.Xtrain, train_applications,
                                                   train_loss.clone().detach(), smoothed=True)
        self.threshold_metrics.calc_loss_per_group(self.population_list, self.data.Xval, val_applications,
                                                   val_loss)

        self.threshold_metrics.calc_train_accuracy_per_group(self.data.Xtrain, self.data.Ytrain, Ytrain_pred,
                                                          self.population_list)

        self.threshold_metrics.calc_applications_per_group(train_applications.clone().detach(), self.data.Xtrain,
                                                           self.population_list, smoothed=True)
        self.threshold_metrics.calc_applications_per_group(val_applications, self.data.Xval,
                                                           self.population_list, smoothed=False)
        self.threshold_metrics.calc_accuracy_per_group(val_loss, self.data.Xval, self.population_list)
        self.threshold_metrics.update_precision_per_group(self.population_list, self.cp_map, self.data.Xtrain, None,
                                                          smoothed=True)
        self.threshold_metrics.update_precision_per_group(self.population_list, self.tester.cp_map)
        # if self.app_bias:
        #     self.threshold_metrics.calc_bias_per_group(bias, self.data.Xtrain, self.population_list)
        # if self.penalty:
        #     self.metric_manager.update_metric("penalty", penalty.clone().detach())
        # self.threshold_metrics.update_metric("train_loss", train_epoch_loss.clone().detach())
        # self.threshold_metrics.update_metric("val_loss", val_epoch_loss)
        # self.threshold_metrics.update_metric("undefined_cp", undef_cp)
        # self.threshold_metrics.update_metric("no_applications", undef_apps)

    def train(self, epochs: int, verbose: bool = False, early_stop: int = 30):
        metrics = super().train(epochs, verbose, early_stop)
        self.c_u = self.metric_c_u + self.c_u_tolerance
        self.app_bias = True
        self.thresholds = np.linspace(-0.01, 1.01, 1000)
        best_induced_acc = -1
        best_thresh = np.Inf
        with torch.no_grad():
            scores = self.model(self.data.Xtrain, self.data.Ztrain)
            for t in self.thresholds:
                # self.model.set_bias(t)
                # precision_scores = scores if self.precision_tau is None else self.model(self.data.Xtrain,
                #                                                                         self.data.Ztrain,
                #                                                                         self.precision_tau)
                self.data.Ytrain = self.data.Ytrain.squeeze()
                # loss = self.loss(scores, self.data.Ytrain.type(torch.float))
                Ytrain_pred = (scores > t).type(torch.int)
                cp = self.tester.calc_all_conditional_precisions(Ytrain_pred, self.data.Ytrain,
                                                                      self.data.Xtrain.squeeze(),
                                                                 self.data.Ytrain, self.data.Xtrain)
                applications = self.tester.calc_applications(cp)
                train_loss = torch.ne(Ytrain_pred, self.data.Ytrain).type(torch.int)
                induced_acc = 1 - ((applications * train_loss).sum() / applications.sum())
                if induced_acc > best_induced_acc:
                    best_induced_acc = induced_acc
                    best_thresh = t
                Yval_scores = self.model(self.data.Xval, self.data.Zval)
                Yval_pred = (Yval_scores > t).type(torch.int)
                val_loss = torch.ne(Yval_pred, self.data.Yval).type(torch.int)
                val_cp = self.tester.calc_all_conditional_precisions(Ytrain_pred, self.data.Ytrain.squeeze(),
                                                                     self.data.Xtrain.squeeze(),
                                                                     self.data.Yval.squeeze(),
                                                                     self.data.Xval.squeeze())
                val_applications = self.tester.calc_applications(val_cp)
                val_norm = self.tester.calc_normalization_term(val_applications)
                # val_t_loss = self.tester.calc_epoch_loss(val_norm, val_applications, val_loss)
                # t_loss = self.tester.calc_epoch_loss(val_norm, val_applications, val_loss)
                self.calc_threshold_metrics(applications, val_applications,
                                            train_loss, val_loss, Ytrain_pred)


            self.data.Ytrain = self.data.Ytrain.squeeze()
            Ytrain_pred = (scores > best_thresh).type(torch.int)
            cp = self.tester.calc_all_conditional_precisions(Ytrain_pred, self.data.Ytrain,
                                                             self.data.Xtrain.squeeze(),
                                                             self.data.Ytrain, self.data.Xtrain)
            applications = self.tester.calc_applications(cp)

            Yval_scores = self.model(self.data.Xval, self.data.Zval)
            Yval_pred = (Yval_scores > best_thresh).type(torch.int)
            val_loss = torch.ne(Yval_pred, self.data.Yval).type(torch.int)
            loss = torch.ne(Ytrain_pred, self.data.Ytrain).type(torch.int)
            val_cp = self.tester.calc_all_conditional_precisions(Ytrain_pred, self.data.Ytrain.squeeze(),
                                                                 self.data.Xtrain.squeeze(),
                                                                 self.data.Yval.squeeze(),
                                                                 self.data.Xval.squeeze())
            val_applications = self.tester.calc_applications(val_cp)
            self.calc_threshold_metrics(applications, val_applications,
                                        loss, val_loss, Ytrain_pred)
        induced_acc = 1 - ((val_applications * val_loss).sum() / val_applications.sum())
        print(f"Induced acc : {induced_acc}")
        self.threshold_metrics.set_xvals(self.thresholds)
        return self.threshold_metrics
