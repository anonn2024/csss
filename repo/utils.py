import datetime

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import json


def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def plot_heatmap_with_values(data, title="Heatmap"):
    """
    Create a heatmap from a list of lists with annotations.

    Parameters:
    - data: List of lists containing numerical values.
    """
    # Convert the list of lists to a NumPy array
    data_array = np.array(data)

    # Create a heatmap using imshow
    plt.imshow(data_array, cmap='viridis', interpolation='nearest')

    # Add colorbar to show the scale
    plt.colorbar()

    # Add annotations to each cell
    for i in range(len(data_array)):
        for j in range(len(data_array[0])):
            plt.text(j, i, '{:.4f}'.format(data_array[i, j]), ha='center', va='center', color='black')

    # Show row and column indices
    plt.xticks(np.arange(len(data_array[0])), labels=np.arange(len(data_array[0])))
    plt.yticks(np.arange(len(data_array)), labels=np.arange(len(data_array)))

    # Display the plot
    plt.title(title)
    plt.show()


def dict_to_json_file(input_dict, file_path):
    """
    Convert a dictionary to a JSON file.

    Parameters:
    - input_dict: The dictionary to be converted to JSON.
    - file_path: The file path where the JSON file will be saved.
    """
    with open(file_path, 'w') as json_file:
        json.dump(input_dict, json_file, indent=4)


def generate_density_points(start, end, num_points, visualize=False):
    """
    Generate an array of values from a range with higher density in the middle and lower density on the edges.

    Parameters:
    - start: The start of the range.
    - end: The end of the range.
    - num_points: The number of unique points in the array.
    - visualize: If True, visualize the density function.

    Returns:
    - A numpy array with randomly sampled unique points based on the density distribution, sorted.
    """
    # Create an array of values from start to end
    x = np.linspace(start, end, num_points)

    # Calculate the weights using a Hann window function
    weights = 0.5 * (1 - np.cos(2 * np.pi * (x - start) / (end - start)))
    if weights[0] == 0:
        weights[0] = weights[1]
        weights[-1] = weights[-2]
    # print(weights)

    # Check for zero weights before normalization
    if np.any(weights == 0):
        raise ValueError("Zero weights detected. Adjust the function parameters to avoid this issue.")

    # Normalize the weights to make the sum equal to 1
    weights /= np.sum(weights)

    # Generate random indices based on the weights
    indices = np.random.choice(num_points, num_points, p=weights)

    # Generate the result array based on the random indices
    result_array = x[indices]

    # Ensure unique values
    result_array = np.unique(result_array)

    # while len(result_array) < num_points:
    #     # If the number of unique values is less than num_points, generate additional values
    #     additional_values = generate_density_points(start, end, num_points - len(result_array))
    #     result_array = np.concatenate((result_array, additional_values))

    # Sort the result array after adding all the additional values
    result_array = np.sort(result_array)

    if visualize:
        # Visualize the density function
        plt.plot(x, weights, label='Density Function')
        plt.title('Density Function Visualization')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=(1,1))
        plt.grid()
        plt.show()

    return result_array[:num_points]  # Trim to ensure exactly num_points values


def zo_sigmoid(X: torch.Tensor, c=0.5, tau=3):
    X_moved = torch.clamp(X + c, 0, 1)
    internal = torch.pow(((X_moved * (1 - c)) / (c * (1 - X_moved))), -tau)
    return 1 / (1 + internal)


def log_penalty(x: torch.Tensor, a):
    eps = 0.001
    return - torch.log(x + eps) / a


def parabula_penalty(x: torch.Tensor, a):
    return a * (x - 0.5) ** 2


class MetricManager:
    def __init__(self, output_final_vals=True, output_final_vals_name='', xname="Epoch"):
        self.__metrics = {}
        self.__nested_metric_names = []
        self.__paired_metric_names = []
        self.__final_values = {}
        self.output_final_vals = output_final_vals
        self.output_final_vals_name = output_final_vals_name
        self.xname = xname
        self.xvals = None

    def set_xvals(self, xvals):
        self.xvals = xvals

    def update_metric(self, name, val):
        if self.__metrics.get(name) is None:
            self.__metrics[name] = []

        self.__metrics[name].append(val)

    def update_nested_metric(self, main_name, nested_name, val, paired=False):
        if self.__metrics.get(main_name) is None:
            self.__nested_metric_names.append(main_name)
            self.__metrics[main_name] = {}
            if paired:
                to_append = self.get_base_paired_name(main_name)
                if to_append not in self.__paired_metric_names:
                    self.__paired_metric_names.append(to_append)

        if self.__metrics[main_name].get(nested_name) is None:
            self.__metrics[main_name][nested_name] = []

        self.__metrics[main_name][nested_name].append(val)

    @staticmethod
    def get_base_paired_name(name):
        res = name
        if name.startswith("smoothed"):
            res = name[9:]
        return res

    def calc_mean_apply(self, applications, name="mean_apply"):
        val = applications.type(torch.float).mean()
        self.update_metric(name, val)

    def calc_mean_pass_screening(self, applications, Y, name="mean_pass_screening"):
        val = (applications * Y).type(torch.float).mean()
        self.update_metric(name, val)

    def calc_induced_accuracy(self, application, loss, name="induced_accuracy"):
        val = 1 - ((application * loss).sum() / application.sum())
        self.update_metric(name, val)
        return val

    def calc_train_induced_accuracy(self, Ytrain_pred, Ytrain, application, name="train_induced_accuracy"):
        train_loss = torch.ne(Ytrain_pred, Ytrain).type(torch.int)
        val = 1 - ((application * train_loss).sum() / application.sum())
        self.update_metric(name, val)
        return val

    def get_final_value(self, name):
        final_val = self.__metrics[name][-1]
        if isinstance(final_val, torch.Tensor):
            final_val = final_val.item()
        return final_val

    def calc_accuracy_per_group(self, val_loss, Xval, population_list, name="accuracy_per_group"):
        for x in population_list:
            x = int(x.item())
            val = 1 - (val_loss.squeeze()[Xval[x] == 1]).type(torch.float).mean()
            self.update_nested_metric(name, str(x), val)

    def calc_train_accuracy_per_group(self, Xtrain, Ytrain, Ytrain_pred, population_list, name="train_accuracy_per_group"):
        for x in population_list:
            x = int(x.item())
            train_loss = torch.ne(Ytrain_pred, Ytrain).type(torch.int)
            val = 1 - (train_loss.squeeze()[Xtrain[x] == 1]).type(torch.float).mean()
            self.update_nested_metric(name, str(x), val)

    def calc_penalty_per_group(self, penalty_map, population_list, name="penalty_per_group"):
        for x in population_list:
            x = int(x.item())
            val = penalty_map[x]
            self.update_nested_metric(name, str(x), val)
        app_val = penalty_map["app"]
        self.update_metric("application_penalty", app_val)

    def calc_bias_per_group(self, bias, X, population_list, name="bias_per_group"):
        for x in population_list:
            x = int(x.item())
            val = (bias.squeeze()[X[x] == 1]).type(torch.float).mean()
            self.update_nested_metric(name, str(x), val)

    def calc_applications_per_group(self, applications, X, population_list, smoothed=False):
        name = "smoothed_applications_per_group" if smoothed else "applications_per_group"
        for x in population_list:
            x = int(x.item())
            val = (applications[X[x] == 1]).type(torch.float).mean()
            if not smoothed and val > 0.99:
                val -= (0.02 * x)
            self.update_nested_metric(name, str(x), val, paired=True)

    # the opposite of induced accuracy
    def calc_loss_per_group(self, population_list, X, applications, loss, smoothed=False):
        name = "smoothed_loss_per_group" if smoothed else "loss_per_group"
        for x in population_list:
            x = int(x.item())
            val = ((applications.squeeze()[X[x] == 1] * loss.squeeze()[
                X[x] == 1]).sum()
                   / applications.squeeze()[X[x] == 1].sum())
            # if smoothed:
            #     val = min(1, val)
            self.update_nested_metric(name, str(x), val, paired=True)

    def calc_fairness_constraint_per_group(self, fairness_map, population_list):
        for mode in ["smooth", "zero_one"]:
            name = "smoothed_fairness_constraint_per_group" if mode == "smooth" else "fairness_constraint_per_group"
            for x in population_list:
                x = int(x.item())
                val = fairness_map[mode][x]["1"]
                self.update_nested_metric(name, str(x), val, paired=True)
            val_total =  fairness_map[mode]["total"]
            self.update_nested_metric(name, "total", val_total, paired=True)

    def calc_system_utility(self, applications, Y, Y_pred, c_s, name="system_utility"):
        val = (applications * Y_pred * (Y - c_s)).type(torch.float).mean()
        self.update_metric(name, val)

    def update_fairness_per_group(self, fairness_map, population_list):
        pass

    def calc_system_induced_utility(self, applications, Y, Y_pred, c_s, name="system_induced_utility"):
        val = (applications.squeeze()[applications == 1] * Y_pred.squeeze()[applications == 1]
               * (Y.squeeze()[applications == 1] - c_s)).type(torch.float).mean()
        self.update_metric(name, val)

    def update_precision_per_group(self, population_list, precisions, X=None, bias=None, smoothed=False):
        name = "smoothed_precision_per_group" if smoothed else "precision_per_group"
        for x in population_list:
            x = int(x.item())
            # print(bias)
            b = 0 if (not smoothed or bias is None) else (bias.squeeze()[X[x] == 1]).type(torch.float).mean()
            val = precisions[x] + b
            self.update_nested_metric(name, str(x), val, paired=True)

    def update_model_weights(self, model, population_list, hasXParams=True, name="model_weights"):
        params_obj = list(model.parameters())
        param_vals = []
        for p in params_obj:
            for w in p:
                param_vals.append(w.item())
        x_param_len = len(population_list) if hasXParams else 0
        if hasXParams:
            for x in population_list:
                x = int(x.item())
                self.update_nested_metric(name + "_X", f"b_{x}", param_vals[x])
        z_norm = 0
        for z in range(len(param_vals) - 1 - x_param_len):
            w_z = param_vals[z + x_param_len]
            z_norm += w_z ** 2
        z_norm = np.sqrt(z_norm)
        for z in range(len(param_vals) - 1 - x_param_len):
            z_w_normalized = param_vals[z + x_param_len] / z_norm
            self.update_nested_metric(name + "_Z", f"w_{z}", z_w_normalized)
        self.update_nested_metric(name + "_X", "bias", param_vals[-1])

    def plot_precisions(self):
        list_len = None
        x_range = None
        for key, value_list in self.__metrics["precision_per_group"].items():
            # Generate a common range based on the length of the first list
            if list_len is None:
                list_len = len(value_list)
                x_range = range(list_len) if self.xvals is None else self.xvals
            else:
                assert (len(value_list) == list_len)

            # Plot each list in the list of lists with corresponding label
            # clamp list for plot purposes
            plt.plot(x_range, value_list, label="r_" + key)
        plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
        for key, value_list in self.__metrics["smoothed_precision_per_group"].items():
            # Generate a common range based on the length of the first list
            assert (len(value_list) == list_len)

            # Plot each list in the list of lists with corresponding label
            # clamp list for plot purposes
            plt.plot(x_range, value_list, '--', label="s_" + key)

        # Set plot title and labels
        plt.title('Precision Plot')
        plt.xlabel(self.xname)
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1,1))
        plt.grid()
        plt.show()

    def plot_metrics(self):
        list_len = None
        x_range = None
        for key, value_list in self.__metrics.items():
            if key in self.__nested_metric_names:
                continue
            # Generate a common range based on the length of the first list
            if list_len is None:
                list_len = len(value_list)
                x_range = range(list_len) if self.xvals is None else self.xvals

            # Plot each list in the list of lists with corresponding label
            # clamp list for plot purposes
            value_list = [min(x, 1.5) for x in value_list]
            plt.plot(x_range, value_list[:len(x_range)], label=key)
            final_val = value_list[-1]
            if isinstance(final_val, torch.Tensor):
                final_val = final_val.item()
            self.__final_values[key] = final_val

        # Set plot title and labels
        plt.title('Metric Plot')
        plt.xlabel(self.xname)
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1,1))
        plt.grid()
        plt.show()

        for name in self.__nested_metric_names:
            if self.get_base_paired_name(name) in self.__paired_metric_names:
                continue
            self.__final_values[name] = {}
            for key, value_list in self.__metrics[name].items():
                # Generate a common range based on the length of the first list

                # Plot each list in the list of lists with corresponding label
                # clamp list for plot purposes
                x_range = range(list_len) if self.xvals is None else self.xvals
                plt.plot(x_range, value_list[:len(x_range)], label=key)
                final_val = value_list[-1]
                if isinstance(final_val, torch.Tensor):
                    final_val = final_val.item()
                self.__final_values[name][key] = final_val
            if name == "accuracy_per_group":
                value_list = self.__metrics["induced_accuracy"]
                list_len = len(value_list)
                x_range = range(list_len) if self.xvals is None else self.xvals
                plt.plot(x_range, value_list[:len(x_range)], color='black', label="induced_accuracy")

            if name == "train_accuracy_per_group":
                value_list = self.__metrics["train_induced_accuracy"]
                list_len = len(value_list)
                x_range = range(list_len) if self.xvals is None else self.xvals
                plt.plot(x_range, value_list[:len(x_range)], color='black', label="train_induced_accuracy")

            if name == "penalty_per_group":
                value_list = self.__metrics["application_penalty"]
                list_len = len(value_list)
                x_range = range(list_len) if self.xvals is None else self.xvals
                plt.plot(x_range, value_list[:len(x_range)], color='black', label="application_penalty")

            # Set plot title and labels
            plt.title(name)
            plt.xlabel(self.xname)
            plt.ylabel('Value')
            plt.legend(bbox_to_anchor=(1,1))
            plt.show()

        for name in self.__paired_metric_names:
            self.__final_values[name] = {}
            self.__final_values["smoothed_" + name] = {}
            for key, value_list in self.__metrics[name].items():
                # Generate a common range based on the length of the first list
                if list_len is None:
                    list_len = len(value_list)
                    x_range = range(list_len) if self.xvals is None else self.xvals

                # Plot each list in the list of lists with corresponding label
                # clamp list for plot purposes
                plt.plot(x_range, value_list[:len(x_range)], label="r_" + key)
                final_val = value_list[-1]
                if isinstance(final_val, torch.Tensor):
                    final_val = final_val.item()
                self.__final_values[name][key] = final_val
            plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
            for key, value_list in self.__metrics["smoothed_" + name].items():
                # Generate a common range based on the length of the first list
                final_val = value_list[-1]
                if isinstance(final_val, torch.Tensor):
                    final_val = final_val.item()
                self.__final_values["smoothed_" + name][key] = final_val

                # Plot each list in the list of lists with corresponding label
                # clamp list for plot purposes
                # print(name)
                plt.plot(x_range, value_list[:len(x_range)], '--', label="s_" + key, alpha=0.4)

            # Set plot title and labels
            plt.title(name)
            plt.xlabel(self.xname)
            plt.ylabel('Value')
            plt.legend(bbox_to_anchor=(1,1))
            plt.grid()
            plt.show()

        if self.output_final_vals:
            filename1 = datetime.datetime.now().strftime("_%m_%d-%H_%M")
            path = "results/" + self.output_final_vals_name + filename1
            dict_to_json_file(self.__final_values, path)

    def plot_single_metric(self, name, ylim=None, plot_title='', path='', print_out=False):
        plot_title = name if not plot_title else plot_title
        if ylim is not None:
            plt.ylim(ylim)
        list_len = None
        x_range = None
        if name not in self.__nested_metric_names:
            value_list = self.__metrics[name]
            list_len = len(value_list)
            x_range = range(list_len) if self.xvals is None else self.xvals
            plt.plot(x_range, value_list, label=plot_title)

            final_val = value_list[-1]
            if isinstance(final_val, torch.Tensor):
                final_val = final_val.item()
            return {name: final_val}

        elif self.get_base_paired_name(name) not in self.__paired_metric_names:
            final_vals = {name: {}}
            for key, value_list in self.__metrics[name].items():
                # Generate a common range based on the length of the first list
                if list_len is None:
                    list_len = len(value_list)
                    x_range = range(list_len) if self.xvals is None else self.xvals

                # Plot each list in the list of lists with corresponding label
                # clamp list for plot purposes
                plt.plot(x_range, value_list[:len(x_range)], label=key)

                final_val = value_list[-1]
                if isinstance(final_val, torch.Tensor):
                    final_val = final_val.item()
                final_vals[name][key] = final_val
            if name == "accuracy_per_group":
                value_list = self.__metrics["induced_accuracy"]
                list_len = len(value_list)
                x_range = range(list_len) if self.xvals is None else self.xvals
                plt.plot(x_range, value_list[:len(x_range)], color='black', label="induced_accuracy")
                final_val = value_list[-1]
                if isinstance(final_val, torch.Tensor):
                    final_val = final_val.item()
                final_vals["induced_accuracy"] = final_val
                max_induced_acc = float(np.nanmax(value_list[2500:])) if len(value_list) > 2500 else np.nan
                if self.xvals is not None:
                    max_induced_acc = float(np.nanmax(value_list[:-1]))
                final_vals["max_induced_acc"] = max_induced_acc

            if name == "train_accuracy_per_group":
                value_list = self.__metrics["train_induced_accuracy"]
                list_len = len(value_list)
                x_range = range(list_len) if self.xvals is None else self.xvals
                plt.plot(x_range, value_list[:len(x_range)], color='black', label="train_induced_accuracy")
                final_val = value_list[-1]
                if isinstance(final_val, torch.Tensor):
                    final_val = final_val.item()
                final_vals["train_induced_accuracy"] = final_val
                max_induced_acc = float(np.nanmax(value_list[2500:])) if len(value_list) > 2500 else np.nan
                if self.xvals is not None:
                    max_induced_acc = float(np.nanmax(value_list[:-1]))
                final_vals["train_max_induced_acc"] = max_induced_acc

            plt.title(plot_title)
            plt.xlabel(self.xname)
            plt.ylabel('Value')
            plt.legend(bbox_to_anchor=(1,1))
            plt.grid()
            if path:
                plt.savefig(fname=path)
            if print_out:
                plt.show()
            plt.close()
            return final_vals

        else:
            final_vals = {name: {}, "smoothed_" + name: {}}
            for key, value_list in self.__metrics[name].items():
                # Generate a common range based on the length of the first list
                if list_len is None:
                    list_len = len(value_list)
                    x_range = range(list_len) if self.xvals is None else self.xvals

                # Plot each list in the list of lists with corresponding label
                # clamp list for plot purposes
                plt.plot(x_range, value_list[:len(x_range)], label="r_" + key)
                final_val = value_list[-1]
                if isinstance(final_val, torch.Tensor):
                    final_val = final_val.item()
                final_vals[name][key] = final_val
            plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
            for key, value_list in self.__metrics["smoothed_" + name].items():
                # Generate a common range based on the length of the first list

                final_val = value_list[-1]
                if isinstance(final_val, torch.Tensor):
                    final_val = final_val.item()
                final_vals["smoothed_" + name][key] = final_val
                # Plot each list in the list of lists with corresponding label
                # clamp list for plot purposes
                plt.plot(x_range, value_list[:len(x_range)], '--', label="s_" + key, alpha=0.4)
            if name == "fairness_constraint_per_group":
                value_list = self.__metrics["fairloss"]
                list_len = len(value_list)
                x_range = range(list_len) if self.xvals is None else self.xvals
                plt.plot(x_range, value_list[:len(x_range)], color='black', label="fairloss")
                final_val = value_list[-1]
                if isinstance(final_val, torch.Tensor):
                    final_val = final_val.item()
                final_vals["fairloss"] = final_val

            # Set plot title and labels
            plt.title(plot_title)
            plt.xlabel(self.xname)
            plt.ylabel('Value')
            plt.legend(bbox_to_anchor=(1,1))
            plt.grid()
            if path:
                plt.savefig(fname=path)
            if print_out:
                plt.show()
            plt.close()
            return final_vals

