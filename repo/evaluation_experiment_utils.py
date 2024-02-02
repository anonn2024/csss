import os
from enum import Enum
from dataset_prep_utils import *
import typing
from itertools import product

BASE_DIR = "./prec_acc_runs"
RESULT_DIR = BASE_DIR + "/tables"


class RunType(Enum):
    NAIVE = 1,
    NAIVE_NO_Z = 2,
    STR = 3,
    STR_NO_Z = 4,
    SEMI_STR = 5,
    SEMI_STR_NO_Z = 6,
    SHORT_NAIVE = 7,
    SHORT_NAIVE_NO_Z = 8,
    SHORT_SEMI_STR = 9,
    SHORT_SEMI_STR_NO_Z = 10

    @staticmethod
    def with_x(run_type):
        return run_type in [RunType.NAIVE, RunType.STR, RunType.SEMI_STR, RunType.SHORT_NAIVE, RunType.SHORT_SEMI_STR]

    @staticmethod
    def is_short(run_type):
        return run_type in [RunType.SHORT_NAIVE, RunType.SHORT_NAIVE_NO_Z, RunType.SHORT_SEMI_STR,
                            RunType.SHORT_SEMI_STR_NO_Z]

    @staticmethod
    def run_function(run_type):
        if run_type in [RunType.NAIVE, RunType.NAIVE_NO_Z, RunType.SHORT_NAIVE, RunType.SHORT_NAIVE_NO_Z]:
            return run_naive
        if run_type in [RunType.STR, RunType.STR_NO_Z]:
            return run_strategic
        return run_semi_str

    def __str__(self):
        run_type_strs = {
            RunType.NAIVE: "naive",
            RunType.NAIVE_NO_Z: "naive_no_z",
            RunType.STR: "str",
            RunType.STR_NO_Z: 'str_no_z',
            RunType.SEMI_STR: "semi_str",
            RunType.SEMI_STR_NO_Z: "semi_str_no_z",
            RunType.SHORT_NAIVE: "short_naive",
            RunType.SHORT_NAIVE_NO_Z: "short_naive_no_z",
            RunType.SHORT_SEMI_STR: "short_semi_str",
            RunType.SHORT_SEMI_STR_NO_Z: "short_semi_str_no_z"
        }

        return run_type_strs[self]


class RunConfig:
    def __init__(self, name, desc, data, plot_path, c_u_list, prec_tau_list, app_tau_list, penalty_param, fairness_param, c_tolerance):
        self.name = name
        self.desc = desc
        self.data = data
        self.plot_path = plot_path
        self.c_u_list = c_u_list
        self.prec_tau_list = prec_tau_list
        self.app_tau_list = app_tau_list
        self.penalty_param = penalty_param
        self.c_tolerance = c_tolerance
        self.fairness_param = fairness_param


class RunResult:
    def __init__(self, name, descriptor, c_u, prec_tau, app_tau, penalty_param, c_tolerance, fairness_param, run_type, br_stats,
                 results, seed, model_seed):
        self.name = name
        self.descriptor = descriptor
        self.c_u = c_u
        self.prec_tau = prec_tau
        self.app_tau = app_tau
        self.run_type = run_type
        self.c_tolerance = c_tolerance
        self.penalty_param = penalty_param
        self.fairness_param = fairness_param
        self.br_stats = br_stats
        self.prec = results["precision_per_group"]
        self.acc = results["accuracy_per_group"]
        self.induced_acc = results["induced_accuracy"]
        self.max_induced_acc = results["max_induced_acc"]
        self.train_acc = results["train_accuracy_per_group"]
        self.train_induced_acc = results["train_induced_accuracy"]
        self.train_max_induced_acc = results["train_max_induced_acc"]
        self.seed = seed
        self.model_seed = model_seed
        self.fair = results.get("fairness_constraint_per_group")
        self.fair_loss = results.get("fairloss")

    def to_dict(self):
        res = {
            "data_name": self.name,
            "descriptor": self.descriptor,
            "run_type": str(self.run_type),
            "c_u": self.c_u,
            "prec_tau": self.prec_tau,
            "app_tau": self.app_tau,
            "penalty_param": self.penalty_param,
            "c_tolerance": self.c_tolerance,
            "prec": self.prec,
            "acc": self.acc,
            "induced_acc": self.induced_acc,
            "max_induced_acc": self.max_induced_acc,
            "train_acc": self.train_acc,
            "train_induced_acc": self.train_induced_acc,
            "train_max_induced_acc": self.train_max_induced_acc,
            "base_rate": self.br_stats['base_rate'],
            "size": self.br_stats['size'],
            "seed": self.seed,
            "model_seed": self.model_seed
        }
        if self.fair is not None:
            res["fair_con"] = self.fair
            res["fair_loss"] = self.fair_loss
            res["fairness_param"] = self.fairness_param,

        return res

    @staticmethod
    def combine_results_list(run_result_list):
        res = []
        for result in run_result_list:
            res.append(result.to_dict())

        return res


class FilterCondition:
    def __init__(self, Z_cond=None, X_cond=None, Y_cond=None, fraction_to_remove=0, description='full'):
        self.Z_cond = eval(Z_cond) if Z_cond is not None and isinstance(Z_cond, str) else Z_cond
        self.X_cond = eval(X_cond) if X_cond is not None and isinstance(X_cond, str) else X_cond
        self.Y_cond = eval(Y_cond) if Y_cond is not None and isinstance(Y_cond, str) else Y_cond
        self.fraction_to_remove = fraction_to_remove
        self.description = description

    def to_dict(self):
        return {
            "Z_cond": self.Z_cond,
            "X_cond": self.X_cond,
            "Y_cond": self.Y_cond,
            "fraction_to_remove": self.fraction_to_remove,
            "description": self.description
        }

    def is_empty(self):
        return self.Z_cond is None and self.X_cond is None and self.Y_cond is None


def create_and_save_experiment_config(dataset_path, dataset_name, num_groups, conds, c_u_list, prec_tau_list,
                                      app_tau_list, penalty_param, c_tolerance, fairness_param, path_from_base):
    config = {
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "num_groups": num_groups,
        "conds": conds,
        "c_u_list": c_u_list,
        "prec_tau_list": prec_tau_list,
        "app_tau_list": app_tau_list,
        "penalty_param": penalty_param,
        "c_tolerance": c_tolerance,
        "fairness_param": fairness_param
    }

    path = BASE_DIR + '/' + path_from_base + ".json"
    with open(path, "w") as outfile:
        json.dump(config, outfile)
    return config


def calc_fraction_to_remove_for_base_rate(path, num_groups, target_group, goal_br_list):
    """

    :param path:
    :param num_groups:
    :param target_group:
    :param goal_br_list:
    :return: list of condition dictionaries
    """
    Z, X, Y = load_data_from_csv(path, num_groups)
    stats = split_data_into_DataContainer(Z, X, Y).get_stats()
    orig_br = stats['base_rate'][target_group]
    conds = []
    for goal_br in goal_br_list:
        if goal_br > orig_br:
            frac_to_remove = (goal_br - orig_br) / (goal_br * (1 - orig_br))
            y_val = 0
        else:
            frac_to_remove = (orig_br - goal_br) / (orig_br * (1 - goal_br))
            y_val = 1
        conds.append({
            "Z_cond": f"lambda z: z[{target_group}] == 1",
            "Y_cond": f"lambda y: y == {y_val}",
            "fraction_to_remove": frac_to_remove,
            "description": f"br_{target_group}_{goal_br}"
        })
    return conds


def filter_data_by_conditions(Z, X, Y, fc, seed=42):
    set_seeds(seed)
    if fc.is_empty() or fc.fraction_to_remove == 0:
        return Z, X, Y
    all_indices = np.arange(Z.shape[1])
    Z_indices = all_indices if fc.Z_cond is None else np.asarray(fc.Z_cond(Z)).nonzero()[0]
    X_indices = all_indices if fc.X_cond is None else np.asarray(fc.X_cond(X)).nonzero()[0]
    Y_indices = all_indices if fc.Y_cond is None else np.asarray(fc.Y_cond(Y)).nonzero()[0]
    indices_intersect = sorted(list(set(Z_indices) & set(X_indices) & set(Y_indices)))
    amount_to_remove = int(len(indices_intersect) * fc.fraction_to_remove)
    assert (amount_to_remove > 0)
    sample = np.random.choice(indices_intersect, amount_to_remove, replace=False)
    to_leave = np.setdiff1d(all_indices, sample)
    Z = Z[:, to_leave]
    X = X[:, to_leave]
    Y = Y[to_leave]
    return Z, X, Y


def create_sub_datasets_from_dataset_by_conditions(path, num_groups, base_name, cond_lists, seed=42,
                                                   reduce_all_br_percent=0):
    """

    :param path:
    :param num_groups:
    :param base_name:
    :param cond_lists: list of lists of filter-condition-dictionaries: [[{"Z_cond": Z_cond,
                                                                         "X_cond": X_cond,
                                                                         "Y_cond": Y_cond,
                                                                         "fraction_to_remove": fraction_to_remove},
                                                                         "description": description
                                                                         ]]
    :param seed:
    :return:
    """
    set_seeds(seed)
    print("------------------------------------------------")
    print("creating new datasets:")

    Z, X, Y = load_data_from_csv(path, num_groups)
    if reduce_all_br_percent > 0:
        assert (reduce_all_br_percent < 1)
        Z, X, Y = remove_percent_from_each_group(Z, X, Y, num_groups, reduce_all_br_percent, seed=0)
    data = split_data_into_DataContainer(Z, X, Y, seed=seed)
    # Result dictionary of {DataName : Data}
    result_datasets = {}

    for i in range(len(cond_lists)):
        new_Ztrain, new_Xtrain, new_Ytrain = data.Ztrain, data.Xtrain, data.Ytrain
        new_Zval, new_Xval, new_Yval = data.Zval, data.Xval, data.Yval
        desc = ''
        if reduce_all_br_percent > 0:
            desc += f"br_remove_{reduce_all_br_percent}_"
        for conds in cond_lists[i]:
            fc = FilterCondition(**conds)
            new_Ztrain, new_Xtrain, new_Ytrain = filter_data_by_conditions(new_Ztrain, new_Xtrain, new_Ytrain, fc, seed)
            new_Zval, new_Xval, new_Yval = filter_data_by_conditions(data.Zval, data.Xval, data.Yval, fc, seed)
            desc += fc.description + '_'
        desc = desc[:-1]
        new_data = DataContainer(new_Ztrain, new_Zval, None, new_Xtrain, new_Xval, None, new_Ytrain, new_Yval, None)
        print(f"\nCreated new variation dataset {base_name + '_' + desc}:")
        new_br_stats = new_data.get_stats(print_out=True)

        result_datasets[desc] = {}
        result_datasets[desc]["data"] = new_data
        result_datasets[desc]["br_stats"] = new_br_stats

    print("------------------------------------------------")
    return result_datasets


def run_prec_acc_experiment(data, c_u, run_name: str, desc, seed, model_seed=0, prec_tau=2, app_tau=3, penalty_param=6,
                            c_tolerance=0.02, fairness_param=0, plot_dir='', print_plots=False, iters=25000):
    res_list = []
    fair = fairness_param >= 0

    def plot_prec_and_acc(metrics, name, is_fair=fair):
        plot_path = None if not plot_dir else plot_dir + '/' + name
        prec_path = None if plot_path is None else plot_path + '_prec'
        acc_path = None if plot_path is None else plot_path + '_acc'
        train_acc_path = None if plot_path is None else plot_path + '_train_acc'
        prec_res = metrics.plot_single_metric("precision_per_group", ylim=[0.45, 0.9], plot_title=name + '_prec',
                                              path=prec_path, print_out=print_plots)
        acc_res = metrics.plot_single_metric("accuracy_per_group", ylim=[0.7, 0.95], plot_title=name + '_acc',
                                             path=acc_path, print_out=print_plots)
        train_acc_res = metrics.plot_single_metric("train_accuracy_per_group", ylim=[0.7, 0.95],
                                                   plot_title=name + '_train_acc',
                                                   path=train_acc_path, print_out=print_plots)
        fair_res = {}
        if is_fair:
            fair_path = None if plot_path is None else plot_path + '_fair'
            fair_res = metrics.plot_single_metric("fairness_constraint_per_group", ylim=[0.7, 0.95],
                                                                  plot_title=name + '_fair',
                                                                  path=fair_path, print_out=print_plots)
        res = {**prec_res, **acc_res, **train_acc_res, **fair_res}
        return res

    for run_type in RunType:
        if run_type in [RunType.NAIVE_NO_Z, RunType.SEMI_STR_NO_Z]:
            continue
        if RunType.is_short(run_type):
            continue
        run_iters = min(10000, iters) if RunType.is_short(run_type) else iters
        with_z = RunType.with_x(run_type)

        name = desc + '_' + str(run_type) + '_' + str(c_u)
        run_func = RunType.run_function(run_type)
        metrics = run_func(data,
                           with_z=with_z,
                           plot_all=False,
                           c_u=c_u,
                           prec_tau=prec_tau,
                           app_tau=app_tau,
                           iters=run_iters,
                           lr=0.1,
                           c_tolerance=c_tolerance,
                           penalty_param=penalty_param,
                           fairness_param=fairness_param,
                           model_seed=model_seed,
                           fair=fair)
        res = plot_prec_and_acc(metrics, name)
        br_stats = data.get_stats()

        res_list.append(
            RunResult(run_name, desc, c_u, prec_tau, app_tau, penalty_param, c_tolerance, fairness_param, run_type, br_stats, res, seed,
                      model_seed))

    return res_list


def run_multiple(configs: typing.List[RunConfig], result_path, seed=None, model_seed=0):
    '''
    :param configs:
    A list of RunConfigs with:
    name, data, plot_path, c_u_list,
    :return:
    result a list of RunResult:
    name, c_u, results
    where result
    '''
    res = []
    for config in configs:
        for c_u in config.c_u_list:
            for pt in config.prec_tau_list:
                for at in config.app_tau_list:
                    res += run_prec_acc_experiment(data=config.data,
                                                   c_u=c_u,
                                                   run_name=config.name,
                                                   desc=config.desc,
                                                   seed=seed,
                                                   model_seed=model_seed,
                                                   prec_tau=pt,
                                                   app_tau=at,
                                                   penalty_param=config.penalty_param,
                                                   c_tolerance=config.c_tolerance,
                                                   fairness_param=config.fairness_param,
                                                   plot_dir=config.plot_path)

    res = RunResult.combine_results_list(res)

    path = result_path + '_' + datetime.datetime.now().strftime("%m_%d-%H_%M") + ".json"
    with open(path, "w") as outfile:
        json.dump(res, outfile)
    return res


def add_missing_combinations(df, col_names):
    # Extract the existing values in the specified columns
    value_ranges = [df[col].unique() for col in col_names]

    # Generate all possible combinations of values within the inferred ranges
    combinations = list(product(*value_ranges))

    # Create a DataFrame with the generated combinations for specified columns
    all_combinations_df = pd.DataFrame(combinations, columns=col_names)

    # Merge the original DataFrame with the DataFrame of all combinations
    merged_df = pd.merge(df[col_names], all_combinations_df, on=col_names, how='right', indicator=True)

    # Identify missing combinations
    missing_combinations = merged_df[merged_df['_merge'] == 'right_only'].drop('_merge', axis=1)

    # Append the missing combinations to the original DataFrame
    df = pd.concat([df, missing_combinations], ignore_index=True)

    return df


def generate_table(dir_list, name='res'):
    jsons = []
    for dir in dir_list:
        jsons += [dir + '/' + f for f in os.listdir(dir) if
                  not f.startswith('.') and os.path.isfile(os.path.join(dir, f))]
    results = []
    for js in jsons:
        with open(js, 'r') as file:
            # print(f"file: {js}")
            data = json.load(file)
            results += data

    df = pd.json_normalize(results).round(8)

    # column_names = ["c_u", "seed", "model_seed", "penalty_param", "fairness_param"]
    # value_ranges = [[0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85], list(range(0, 10)), list(range(0, 10))]

    # df = add_missing_combinations(df, column_names)

    df.to_csv(RESULT_DIR + '/' + name + '_' + datetime.datetime.now().strftime("%m_%d-%H_%M") + ".csv", index=False)
