from sklearn.svm import SVR, SVC

from data import *
from trainer import *
import torch
from torch import nn
from model import *
from copy import deepcopy
from torch.utils.data import random_split
from torch.optim import SGD


def learn(c_u, c_s, N, n_samples, tau, penalty_scalar=10, alt_sig=True, data=None, seed=42):
    if data is None:
        DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
        data = get_train_and_val_datasets(DG, train_ratio=0.7, seed=seed)

    population_list = torch.Tensor([*range(N)])

    model = LinearRegressionModel(data.Xtrain, data.Ztrain)
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=0.03)
    trainer = StrategicTrainer(model, data, opt, loss, population_list, tau, c_u, c_s, penalty_scalar=penalty_scalar,
                               c_u_tolerance=0.02,
                               cust_sig=alt_sig, penalty=True, alt_penalty=False, run_name="learn")

    metrics = trainer.train(35000, early_stop=30000)

    metrics.plot_metrics()


SEED = 0
NAIVE_EPOCHS = 35000


def run_naive(data, with_x=True, plot_all=True, c_u=0.7, prec_tau=1, app_tau=1, lr=0.1, model_std=1.0, model_seed=0,
              naive_init=False, c_tolerance=0.02, penalty_param=6, fair=False, fairness_param=3, iters=NAIVE_EPOCHS):
    set_seeds(SEED)
    N = data.Xtrain.shape[0]
    population_list = torch.Tensor([*range(N)])

    model = LinearRegressionModel(data.Xtrain, data.Ztrain, std=model_std,
                                  seed=model_seed) if with_x else LinearRegressionModel(None,
                                                                                        data.Ztrain,
                                                                                        withXParams=False,
                                                                                        std=model_std,
                                                                                        seed=model_seed)
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=lr)
    naive_trainer = NonStrategicTrainer(model, data,
                                        opt, loss, population_list, tau=app_tau, metric_c_u=c_u,
                                        c_s=0, run_name="naive")
    to_print = " without group weights" if not with_x else ""
    print("running naive classifier" + to_print + ":", flush=True)
    metrics = naive_trainer.train(iters, early_stop=iters)
    if plot_all:
        metrics.plot_metrics()
    return metrics


SEMI_STR_EPOCHS = 35000


def run_semi_str(data, with_x=True, plot_all=True, c_u=0.7, prec_tau=1, app_tau=1, lr=0.03, model_std=1.0, model_seed=0,
                 naive_init=False, c_tolerance=0.02, penalty_param=6, fair=False, fairness_param=3, iters=SEMI_STR_EPOCHS):
    set_seeds(SEED)
    N = data.Xtrain.shape[0]
    population_list = torch.Tensor([*range(N)])

    model = LinearRegressionModel(data.Xtrain, data.Ztrain, std=model_std,
                                  seed=model_seed) if with_x else LinearRegressionModel(None,
                                                                                        data.Ztrain,
                                                                                        withXParams=False,
                                                                                        std=model_std,
                                                                                        seed=model_seed)
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=lr)
    semi_trainer = NonStrategicThresholdTrainer(model, data,
                                                opt, loss, population_list, tau=app_tau, c_u=c_u,
                                                c_u_tolerance=c_tolerance, c_s=0, prec_tau=prec_tau,
                                                run_name="semi_str")
    to_print = " without group weights" if not with_x else ""
    print("running semi-strategic classifier" + to_print + ":", flush=True)
    metrics = semi_trainer.train(iters, early_stop=iters)
    if plot_all:
        metrics.plot_metrics()
    return metrics


STR_EPOCHS = 35000


def run_strategic(data, with_x=True, plot_all=True, c_u=0.7, prec_tau=2, app_tau=3, lr=0.1, model_std=1.0, model_seed=0,
                  naive_init=False, c_tolerance=0.02, penalty_param=6,  iters=STR_EPOCHS, fair=False, fairness_param=3):
    set_seeds(SEED)
    N = data.Xtrain.shape[0]
    population_list = torch.Tensor([*range(N)])

    model = LinearRegressionModel(data.Xtrain, data.Ztrain, std=model_std,
                                  seed=model_seed) if with_x else LinearRegressionModel(None,
                                                                                        data.Ztrain,
                                                                                        withXParams=False,
                                                                                        std=model_std,
                                                                                        seed=model_seed)
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=lr)
    if naive_init:
        naive_trainer = NonStrategicTrainer(model, data,
                                            opt, loss, population_list, tau=app_tau, metric_c_u=c_u, c_s=0)
        naive_trainer.train(iters, early_stop=iters)
        loss = nn.BCELoss(reduction='none')
        opt = SGD(model.parameters(), lr=lr)
    strategic_trainer = StrategicTrainer(model,
                                         data,
                                         opt,
                                         loss,
                                         population_list,
                                         c_u=c_u,
                                         tau=app_tau,
                                         c_s=0.4,
                                         penalty=True,
                                         precision_tau=prec_tau,
                                         c_u_tolerance=c_tolerance,
                                         fair=fair,
                                         penalty_scalar=penalty_param,
                                         fairness_param=fairness_param,
                                         run_name="str")
    to_print = " without group weights" if not with_x else ""
    print("running strategic classifier" + to_print + ":", flush=True)
    metrics = strategic_trainer.train(iters, early_stop=iters)
    if plot_all:
        metrics.plot_metrics()
    return metrics


def sanity_check(c_u, c_s, N, n_samples, tau, data=None, seed=42):
    # data generation
    if data is None:
        DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
        data = get_train_and_val_datasets(DG, train_ratio=0.7, seed=seed)

    population_list = torch.Tensor([*range(N)])

    model = LinearRegressionModel(data.Xtrain, data.Ztrain)
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=0.03)
    model_copy = deepcopy(model)
    loss_copy = nn.BCELoss(reduction='none')
    opt_copy = SGD(model_copy.parameters(), lr=0.03)

    # # Strategic trainer with c_u = 0
    str_trainer = StrategicTrainer(model, data, opt, loss,
                                   population_list, tau, c_u=0, c_s=0)
    str_trainer.train(30000, early_stop=30000)
    str_scores = model(data.Xval, data.Zval)
    YVal_pred_str = (str_scores > 0.5).type(torch.int)
    str_loss = torch.ne(YVal_pred_str, data.Yval).type(torch.int).sum() / YVal_pred_str.shape[0]
    print(f'Strategic classifier loss is: {str_loss}')

    # # Non-Strategic trainer
    naive_trainer = NonStrategicTrainer(model_copy, data,
                                        opt_copy, loss_copy, population_list, tau, c_s=0)
    naive_trainer.train(30000, early_stop=30000)
    naive_scores = model_copy(data.Xval, data.Zval)
    YVal_pred_naive = (naive_scores > 0.5).type(torch.int)
    naive_loss = torch.ne(YVal_pred_naive, data.Yval).type(torch.int).sum() / YVal_pred_naive.shape[0]
    print(f'naive classifier loss is: {naive_loss}')

    # scikit classifier
    naive_cls = LinearClassifier(np.unique(data.Xtrain))
    naive_cls.fit(data.Xtrain.numpy(), data.Ztrain.numpy(), data.Ytrain.numpy())
    Yval_pred = naive_cls(data.Xval, data.Zval)
    sk_naive_loss = 1 - (np.count_nonzero(Yval_pred == data.Yval.numpy()) / Yval_pred.shape[0])
    print(f'Scikit-learn naive classifier loss is: {sk_naive_loss}')


def run_naive_before_strategic(c_u, c_s, N, n_samples, tau, data=None, seed=42):
    # data generation
    if data is None:
        DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
        data = get_train_and_val_datasets(DG, train_ratio=0.7, seed=seed)

    population_list = torch.Tensor([*range(N)])

    model = LinearRegressionModel(data.Xtrain, data.Ztrain)
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=0.03)
    model_copy = deepcopy(model)

    # Strategic trainer from scratch
    print(f'################ Regular Random Init ###################')

    no_init_trainer = StrategicTrainer(model, data, opt, loss,
                                       population_list, tau, c_u, c_s, penalty=True, run_name="no_init_strategic")
    no_init_metrics = no_init_trainer.train(30000, early_stop=30000)

    no_init_metrics.plot_metrics()

    # Non-Strategic trainer
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model_copy.parameters(), lr=0.03)

    print(f'################ Naive Run ###################')

    naive_trainer = NonStrategicTrainer(model_copy, data,
                                        opt, loss, population_list, tau, c_s=0)
    naive_metrics = naive_trainer.train(30000, early_stop=30000)
    naive_metrics.plot_metrics()

    # # Strategic trainer
    print(f'################ Strategic with Naive Init ###################')
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model_copy.parameters(), lr=0.03)
    str_trainer = StrategicTrainer(model_copy, data, opt, loss,
                                   population_list, tau, c_u, c_s, penalty=True, run_name="naive_init_strategic")
    naive_init_metrics = str_trainer.train(30000, early_stop=30000)
    naive_init_metrics.plot_metrics()


def run_compare_multiple_inits(c_u, c_s, N, n_samples, tau, data=None, penalty_scalar=10, alt_sig=True, num_runs=5,
                               seed=42):
    if data is None:
        DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
        data = get_train_and_val_datasets(DG, train_ratio=0.7, seed=seed)

    population_list = torch.Tensor([*range(N)])

    for i in range(num_runs):
        print(f'################ Run {i} ###################')
        model = LinearRegressionModel(data.Xtrain, data.Ztrain)
        loss = nn.BCELoss(reduction='none')
        opt = SGD(model.parameters(), lr=0.03)
        trainer = StrategicTrainer(model, data, opt, loss,
                                   population_list, tau, c_u, c_s, penalty_scalar,
                                   cust_sig=alt_sig, penalty=True, alt_penalty=False, precision_tau=2,
                                   run_name="multi_init_run_" + "i")

        metrics = trainer.train(35000, early_stop=30000)

        metrics.plot_metrics()


def get_group_naive_learning_matrix(c_u=0.7, c_s=0.4, N=4, n_samples=10000, data=None, seed=42):
    if data is None:
        DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
        data = get_train_and_val_datasets(DG, train_ratio=0.7, seed=seed)

    Xtrain = data.Xtrain.squeeze()
    Ztrain = data.Ztrain
    Ytrain = data.Ytrain.squeeze()
    Xval = data.Xval.squeeze()
    Zval = data.Zval
    Yval = data.Yval.squeeze()

    Ztrain_per_x = []
    Ytrain_per_x = []
    Zval_per_x = []
    Yval_per_x = []
    linear_accs = []
    non_linear_accs = []

    for x in range(N):
        Ztrain_per_x.append(Ztrain[:, Xtrain[x] == 1])
        Ytrain_per_x.append(Ytrain[Xtrain[x] == 1])
        Zval_per_x.append(Zval[:, Xval[x] == 1])
        Yval_per_x.append(Yval[Xval[x] == 1])
        linear_accs.append([])
        non_linear_accs.append([])

    for i in range(N):
        cls = LogisticRegression(penalty=None, solver="saga", C=1, max_iter=3000)
        cls.fit(Ztrain_per_x[i].numpy().T, Ytrain_per_x[i].numpy())
        for j in range(N):
            linear_accs[i].append(cls.score(Zval_per_x[j].numpy().T, Yval_per_x[j].numpy().T))

    plot_heatmap_with_values(linear_accs, "Linear")

    for i in range(N):
        cls = SVC(kernel="rbf", C=100, gamma=0.1)
        cls.fit(Ztrain_per_x[i].numpy().T, Ytrain_per_x[i].numpy())
        for j in range(N):
            non_linear_accs[i].append(cls.score(Zval_per_x[j].numpy().T, Yval_per_x[j].numpy().T))

    plot_heatmap_with_values(non_linear_accs, "Non-Linear")


def learn_weights_independently(c_u, c_s, N, n_samples, tau, data=None, seed=42):
    set_seeds(seed)

    if data is None:
        DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
        data = get_train_and_val_datasets(DG, train_ratio=0.7, seed=seed)

    learn(c_u=c_u, c_s=c_s, N=N, n_samples=n_samples, tau=tau, penalty_scalar=2, data=data, seed=seed)

    Xtrain = data.Xtrain.squeeze()
    Ztrain = data.Ztrain
    Ytrain = data.Ytrain.squeeze()
    Xval = data.Xval.squeeze()
    Zval = data.Zval
    Yval = data.Yval.squeeze()

    b_x_list = []
    w_dict = {}
    for i in range(Ztrain.shape[0]):
        w_dict[i] = []
    for x in range(N):
        # print(data.Xtrain.shape)
        # print((data.Xtrain[x] == 1).shape)
        Ztrain_x = Ztrain[:, Xtrain[x] == 1]
        Ytrain_x = Ytrain[Xtrain[x] == 1]
        Zval_x = Zval[:, Xval[x] == 1]
        Yval_x = Yval[Xval[x] == 1]
        # print(Ztrain_x.shape, Ytrain_x.shape)
        cls = LogisticRegression(penalty=None, solver="saga", C=1)
        cls.fit(Ztrain_x.numpy().T, Ytrain_x.numpy())
        b_x_list.append(cls.intercept_[0])
        for i in range(Ztrain.shape[0]):
            w_dict[i].append(cls.coef_[0][i])
        print(f"Group {x} coefs: {cls.coef_}, bias: {cls.intercept_}")
        Yval_x_proba = cls.predict_proba(Zval_x.numpy().T)
        Yval_x_pred = (Yval_x_proba[:, 1] > 0.5).astype(int)
        sk_acc = np.count_nonzero(Yval_x_pred == Yval_x.numpy()) / Yval_x_pred.shape[0]
        print(f'Classifier acc for group {x} is: {sk_acc}')

    w_z_list = []
    for i in range(Ztrain.shape[0]):
        w_i_vals = w_dict[i]
        w_z_list.append(sum(w_i_vals) / len(w_i_vals))

    print(f"b_x: {b_x_list}, w_z: {w_z_list}")

    model = LinearRegressionModel(Xtrain, Ztrain)
    for x in range(N):
        model.set_parameter(x, b_x_list[x])
    for z in range(Ztrain.shape[0]):
        model.set_parameter(N + z, w_z_list[z])

    population_list = torch.Tensor([*range(N)])
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=0.03)
    trainer = StrategicTrainer(model, data, opt, loss, population_list, tau, c_u, c_s, penalty_scalar=2,
                               cust_sig=True, penalty=True, alt_penalty=False, run_name="learn_ws_independently")

    metrics = trainer.train(30000, early_stop=30000)

    metrics.plot_metrics()


def learn_tau_change(c_u, c_s, N, n_samples, prec_tau, penalty_scalar=10, alt_sig=True, data=None, seed=42):
    if data is None:
        DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
        data = get_train_and_val_datasets(DG, train_ratio=0.7, seed=seed)

    population_list = torch.Tensor([*range(N)])

    model = LinearRegressionModel(data.Xtrain, data.Ztrain)
    model_copy = deepcopy(model)
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=0.03)
    trainer = StrategicTrainer(model, data, opt, loss, population_list, 2, c_u, c_s, penalty_scalar,
                               cust_sig=alt_sig, penalty=True, alt_penalty=False, run_name="baseline_for_tau_change")

    metrics = trainer.train(35000, early_stop=30000)
    metrics.plot_metrics()

    new_opt = SGD(model_copy.parameters(), lr=0.03)
    new_trainer = StrategicTrainer(model_copy, data, new_opt, loss, population_list, 2, c_u, c_s, penalty_scalar,
                                   cust_sig=alt_sig, penalty=True, alt_penalty=False, precision_tau=prec_tau,
                                   run_name="tau_change")

    new_metrics = new_trainer.train(35000, early_stop=30000)
    new_metrics.plot_metrics()


def compare_application_bias(c_u, c_s, N, n_samples, prec_tau, penalty_scalar=10, alt_sig=True, data=None, seed=42):
    if data is None:
        DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
        data = get_train_and_val_datasets(DG, train_ratio=0.7, seed=seed)

    population_list = torch.Tensor([*range(N)])

    model = LinearRegressionModel(data.Xtrain, data.Ztrain)
    model_copy = deepcopy(model)
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=0.03)
    trainer = StrategicTrainer(model, data, opt, loss, population_list, 2, c_u, c_s, penalty_scalar,
                               cust_sig=alt_sig, penalty=True, alt_penalty=False, app_bias=False,
                               precision_tau=prec_tau, run_name="baseline_for_app_bias")

    metrics = trainer.train(35000, early_stop=30000)
    metrics.plot_metrics()

    new_opt = SGD(model_copy.parameters(), lr=0.03)
    new_trainer = StrategicTrainer(model_copy, data, new_opt, loss, population_list, 2, c_u, c_s, penalty_scalar,
                                   cust_sig=alt_sig, penalty=True, alt_penalty=False, precision_tau=prec_tau,
                                   run_name="app_bias")

    new_metrics = new_trainer.train(35000, early_stop=30000)
    new_metrics.plot_metrics()


def run_naive_and_configure_bias(c_u, c_s, N, n_samples, data=None, seed=42):
    if data is None:
        DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
        data = get_train_and_val_datasets(DG, train_ratio=0.7, seed=seed)

    population_list = torch.Tensor([*range(N)])

    model = LinearRegressionModel(data.Xtrain, data.Ztrain)
    model_copy = deepcopy(model)
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=0.03)
    trainer = StrategicTrainer(model, data, opt, loss, population_list, 2, c_u, c_s, penalty_scalar=10,
                               cust_sig=True, penalty=True, alt_penalty=False, app_bias=True,
                               precision_tau=2, run_name="naive_for_bias_baseline")
    metrics = trainer.train(35000, early_stop=30000)
    metrics.plot_metrics()

    new_opt = SGD(model_copy.parameters(), lr=0.03)
    naive_trainer = NonStrategicThresholdTrainer(model_copy, data,
                                                 new_opt, loss, population_list, tau=2, c_s=0)
    naive_metrics = naive_trainer.train(35000, early_stop=30000)
    naive_metrics.plot_metrics()


def run_without_x_weights(c_u, c_s, N, n_samples, prec_tau, penalty_scalar=10, alt_sig=True, data=None, seed=42):
    if data is None:
        DG = DataGeneratorZ2D(c_u, c_s, N, n_samples)
        data = get_train_and_val_datasets(DG, train_ratio=0.7, seed=seed)

    population_list = torch.Tensor([*range(N)])

    model = LinearRegressionModel(data.Xtrain, data.Ztrain)
    no_x_model = LinearRegressionModel(None, data.Ztrain, withXParams=False)
    for z in range(data.Ztrain.shape[0]):
        no_x_model.set_parameter(z, model.get_parameter(N + z))
    loss = nn.BCELoss(reduction='none')
    opt = SGD(model.parameters(), lr=0.03)
    trainer = StrategicTrainer(model, data, opt, loss, population_list, 2, c_u, c_s, penalty_scalar,
                               cust_sig=alt_sig, penalty=True, alt_penalty=False, app_bias=True,
                               precision_tau=prec_tau, run_name="naive_for_no_x_baseline")
    metrics = trainer.train(35000, early_stop=30000)
    metrics.plot_metrics()

    new_opt = SGD(no_x_model.parameters(), lr=0.03)
    no_x_trainer = StrategicTrainer(no_x_model, data, new_opt, loss, population_list, 2, c_u, c_s, penalty_scalar,
                                    cust_sig=alt_sig, penalty=True, alt_penalty=False, app_bias=True,
                                    precision_tau=prec_tau, run_name="no_x_weights")
    naive_metrics = no_x_trainer.train(35000, early_stop=30000)
    naive_metrics.plot_metrics()
