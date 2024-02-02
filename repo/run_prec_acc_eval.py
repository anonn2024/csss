import argparse
from evaluation_experiment_utils import *


def run(config_path, result_path, seed, model_seed=0, br_reduce=0.0):
    with open(config_path, 'r') as file:
        data = json.load(file)
        path = data["dataset_path"]
        dataset_name = data['dataset_name']
        num_groups = data["num_groups"]
        cond_list = data['conds']
        c_u_list = data['c_u_list']
        prec_tau_list = data['prec_tau_list']
        app_tau_list = data['app_tau_list']
        penalty_param = data["penalty_param"]
        c_tolerance = data["c_tolerance"]
        fairness_param = data["fairness_param"]

    split_data_variations = create_sub_datasets_from_dataset_by_conditions(path, num_groups=num_groups,
                                                                           base_name=dataset_name,
                                                                           cond_lists=cond_list, seed=seed, reduce_all_br_percent=br_reduce)

    configs = [RunConfig(dataset_name, key, split_data_variations[key]['data'], None, c_u_list, prec_tau_list, app_tau_list, penalty_param, fairness_param, c_tolerance) for key
               in split_data_variations]
    run_multiple(configs, result_path=result_path, seed=seed, model_seed=model_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('result_path')
    parser.add_argument('seed')
    parser.add_argument('model_seed')
    parser.add_argument("br_reduce")
    args = parser.parse_args()
    config_path_arg = args.config_path
    result_path_arg = args.result_path
    seed_arg = int(args.seed)
    model_seed_arg = int(args.model_seed)
    br_reduce_arg = float(args.br_reduce)
    run(config_path_arg, result_path_arg, seed_arg, model_seed_arg, br_reduce_arg)
