import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from data import DataContainer
from exploration_experiments import *
from model import LinearClassifier


def load_dataset(full_path, column_names=None, y_name=None, has_header=False):
    assert (column_names is None or y_name in column_names)
    # load the dataset as a numpy array
    header = 0 if has_header and column_names is not None else None
    df = pd.read_csv(full_path, header=header, names=column_names, na_values='?')
    # drop rows with missing
    df = df.dropna()
    # split into inputs and outputs
    X, y = df.drop(y_name, axis=1), df[y_name]
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    print("Dataset columns are:\n", X.columns, f"and classification objective is: {y_name}")
    return X, y


def print_default_column_names(full_path):
    df = pd.read_csv(full_path)
    print(df.columns)


def encode_binary_columns(dataframe, columns_to_encode):
    for column in columns_to_encode:
        if column in dataframe.columns:
            dataframe[column] = LabelEncoder().fit_transform(dataframe[column])
        else:
            print(f"column: {column} is not a valid column name!")


def one_hot_encode_columns(dataframe, columns_to_encode):
    """
    One-hot encodes specified columns in a pandas DataFrame in-place.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - columns_to_encode (list): List of column names to be one-hot encoded.

    Returns:
    None (the original DataFrame is modified in-place).
    """

    # One-hot encode specified columns in-place
    for column in columns_to_encode:
        if column in dataframe.columns:
            # Use get_dummies to perform one-hot encoding
            one_hot_encoded = pd.get_dummies(dataframe[column], prefix=column)

            # Drop the original column and concatenate the one-hot encoded columns
            dataframe.drop(column, axis=1, inplace=True)
            dataframe = pd.concat([dataframe, one_hot_encoded], axis=1)
        else:
            print(f"column: {column} is not a valid column name!")

    print("After one-hot encoding, dataset columns are:\n", dataframe.columns)
    return dataframe


def extract_group_columns(df, col_names):
    df.reset_index(inplace=True, drop=True)
    X = df[col_names]
    print(f"group columns are: {X.columns}")
    population_list_dict = {i: x for x, i in enumerate(col_names)}
    X.rename(columns=population_list_dict, inplace=True)
    Z = df.drop(col_names, axis=1)
    return X, Z


def scale_01(df, col_names):
    scaler = MinMaxScaler()
    scaler.fit(df[col_names])
    df[col_names] = scaler.transform(df[col_names])
    return df


def split_data_into_DataContainer(X, Z, Y, seed=42):
    Xtrain_df, Xval_df, Ztrain_df, Zval_df, Ytrain_df, Yval_df = train_test_split(X, Z, Y, test_size=0.3,
                                                                                  random_state=seed)
    Xtrain_df = Xtrain_df.T
    Xval_df = Xval_df.T
    Ztrain_df = Ztrain_df.T
    Zval_df = Zval_df.T
    Ytrain_df = Ytrain_df.T
    Yval_df = Yval_df.T

    Xtrain = torch.Tensor(Xtrain_df.values)
    Xval = torch.Tensor(Xval_df.values)
    Ztrain = torch.Tensor(Ztrain_df.values)
    Zval = torch.Tensor(Zval_df.values)
    Ytrain = torch.Tensor(Ytrain_df)
    Yval = torch.Tensor(Yval_df)
    return DataContainer(Xtrain, Xval, None, Ztrain, Zval, None, Ytrain, Yval, None)


def run_naive_cls_and_get_stats(data: DataContainer, run_name='', plot_dir=''):
    data.get_stats()

    naive_cls = LinearClassifier(np.unique(data.Xtrain))
    naive_cls.fit(data.Xtrain.numpy(), data.Ztrain.numpy(), data.Ytrain.numpy())
    Yval_pred = naive_cls(data.Xval, data.Zval)
    sk_naive_loss = 1 - (np.count_nonzero(Yval_pred == data.Yval.numpy()) / Yval_pred.shape[0])
    print(f'Scikit-learn classifier loss is: {sk_naive_loss}')

    def plot_prec_and_acc(metrics, name):
        metrics.plot_single_metric("precision_per_group", ylim=[0.45, 0.9], plot_title=name + '_prec', path=plot_dir + '/' + name + '_prec')
        metrics.plot_single_metric("accuracy_per_group", ylim=[0.7, 0.95], plot_title=name + '_acc', path=plot_dir + '/' + name + '_acc')

    num_of_groups = data.Xtrain.shape[0]
    get_group_naive_learning_matrix(data=data, N=num_of_groups)
    metrics = run_naive(data, plot_all=False)
    plot_prec_and_acc(metrics, run_name + '_naive')
    metrics = run_naive(data, with_x=False, plot_all=False)
    plot_prec_and_acc(metrics, run_name + '_naive_no_x')
    metrics = run_strategic(data, plot_all=False)
    plot_prec_and_acc(metrics, run_name + '_str')
    metrics = run_strategic(data, with_x=False, plot_all=False)
    plot_prec_and_acc(metrics, run_name + '_str_no_x')


def combine_categories(df, col_name, transformation_dict: dict):
    # Clean up occupation column by removing any leading/trailing spaces
    df[col_name] = df[col_name].str.strip()
    # Replace occupation categories with new categories
    df[col_name] = df[col_name].replace(transformation_dict)

    # Check value counts of new occupation column
    # df.drop([col_name], axis=1, inplace=True)
    print(f"New narrowed categories : \n{df[col_name].value_counts()}")
    return df


def drop_columns(df, to_drop):
    for col in to_drop:
        df.drop(col, axis=1, inplace=True)


def remove_examples_by_condition(X, Z, Y, amount_to_leave, X_cond=None, Z_cond=None, Y_cond=None, percent=False, seed=42):
    set_seeds(seed)
    assert(not percent or amount_to_leave < 1)
    X.reset_index(inplace=True, drop=True)
    Z.reset_index(inplace=True, drop=True)
    all_indices = X.index.values.tolist()
    X_indices = all_indices if X_cond is None else np.asarray(X_cond(X)).nonzero()[0]
    Z_indices = all_indices if Z_cond is None else np.asarray(Z_cond(Z)).nonzero()[0]
    Y_indices = all_indices if Y_cond is None else np.where(Y_cond(Y))[0].tolist()
    indices_intersect = sorted(list(set(X_indices) & set(Z_indices) & set(Y_indices)))
    amount_to_remove = len(indices_intersect) - amount_to_leave if not percent else int(
        len(indices_intersect) * (1 - amount_to_leave))
    assert(amount_to_remove > 0)
    sample = np.random.choice(indices_intersect, amount_to_remove, replace=False)
    X.drop(index=sample, axis=0, inplace=True)
    Z.drop(index=sample, axis=0, inplace=True)
    X.reset_index(inplace=True, drop=True)
    Z.reset_index(inplace=True, drop=True)
    Y = np.delete(Y, sample, axis=0)
    return X, Z, Y


def remove_percent_from_each_group(X, Z, Y, num_groups, percent_to_remove, negative=True, seed=42):
    for i in range(num_groups):
        y_type_to_remove = 0 if negative else 1
        X, Z, Y = remove_examples_by_condition(X, Z, Y,
                                               amount_to_leave=1 - percent_to_remove,
                                               X_cond=lambda x: x[str(i)] == 1,
                                               Y_cond=lambda y: y == y_type_to_remove,
                                               percent=True,
                                               seed=seed)
    return X, Z, Y


def transform_column_by_ranges_inplace(df, column_name, bins=None, num_bins=None, equal_samples=False, category_prefix="group"):
    # Ensure the input parameters are valid
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a Pandas DataFrame.")
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # If bins are not provided, calculate equally divided bins based on the specified number of bins
    if bins is None:
        if num_bins is None:
            raise ValueError("Either 'bins' or 'num_bins' must be provided.")
        if equal_samples:
            bins = list(pd.qcut(df[column_name], q=num_bins, labels=None, retbins=True)[1])
        else:
            bins = list(pd.cut(df[column_name], bins=num_bins, labels=None, retbins=True, include_lowest=True)[1])
    else:
        min_value = df[column_name].min()
        bins.insert(0, min_value)
        max_value = df[column_name].max()
        bins.append(max_value)

    categories = [f'{category_prefix}_{i}' for i in range(len(bins) - 1)]

    # Update the values in the specified column based on the specified ranges and categories
    df[column_name] = pd.cut(df[column_name], bins=bins, labels=categories, include_lowest=True)
    return bins


def save_data_to_csv(X, Z, y, filepath):
    """
    Save Pandas DataFrame X, Z, and NumPy array y to a CSV file.

    Parameters:
    - X: Pandas DataFrame
    - Z: Pandas DataFrame
    - y: NumPy ndarray
    - filepath: str, path where the CSV file will be saved
    """
    # Save X, Z, and y to CSV with the number of X columns encoded in the filename
    num_X_columns = len(X.columns)
    filepath_with_columns = f"{filepath}_{num_X_columns}.csv"
    print(filepath_with_columns)

    combined_data = pd.concat([X, Z], axis=1)
    combined_data['y'] = pd.Series(y)

    combined_data.to_csv(filepath_with_columns, index=False)

    return filepath_with_columns


def load_data_from_csv(filepath, num_X_columns):
    """
    Load data from a CSV file and return X, Z, and y.

    Parameters:
    - filepath: str, path of the CSV file to load

    Returns:
    - X: Pandas DataFrame
    - Z: Pandas DataFrame
    - y: NumPy ndarray
    """

    # Load data from CSV
    combined_data = pd.read_csv(filepath)

    # Separate X and Z based on the number of X columns
    X = combined_data.iloc[:, :num_X_columns]
    Z = combined_data.iloc[:, num_X_columns:-1]

    # Extract y as a NumPy array
    y = combined_data['y'].values

    return X, Z, y
