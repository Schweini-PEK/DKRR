import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import scale
from utils import kernels


def general_loader(dataset, k, kernel, sigma=1.0, nu=1.0, scaling=1.0):
    """Loading all datasets.

    Load and standardize dataset from folder, then perform kernelization.
    Args:
        kernel (basestring): The type of kernels.
        dataset (basestring): Would come with x suffix if it is x RBF type dataset (e.g., 'abalone_d' for dense RBF).
        k (int): The value of k passed from command line.
        sigma (float): The parameter used for RBF kernels.
        nu (float): Parameter for Matern kernel.
        scaling (float): Rescale the dataset to a smaller size.

    Returns: Gram matrix A, k value, and standardized data.

    """

    # The kernel_func methods depends on the datasets. See Table 4.

    # Linear Kernels
    # if dataset in {'gisette', 'protein'}:
    #     if dataset == 'gisette':
    #         k_default = 12
    #         f = open(f"data/{dataset}.data", 'r')
    #         data = []
    #         for row in f.readlines():
    #             data.append((row.strip()).split(' '))
    #         f.close()
    #         data = np.array(data).astype(float)
    #
    #     else:  # protein
    #         k_default = 10
    #         data = loadmat(f"data/{dataset}.mat")['X'].toarray()
    #
    #     data = scale(data)
    #     a = kernels.normalize_linear_kernel(data)

    k_default = 10
    x = None
    y = None

    if dataset in {'iris'}:
        k_default = 5
        f = open(f'data/krr/{dataset}.data', 'r')
        rows = f.readlines()
        rows.remove('\n')
        data = np.empty([len(rows), 4])

        for i in range(len(rows)):
            row = (rows[i].strip()).split(',')
            data[i] = row[:4]

    elif dataset in {'gas'}:
        k_default = 10

        f = open(f'data/krr/GAS/batch7.dat', 'r')
        rows = f.readlines()

        data = np.empty([len(rows), 128])
        y = np.empty([len(rows)])

        for i in range(len(rows)):
            row = (rows[i].strip()).split(' ')
            y[i] = row[0]
            for j in range(len(row[1:])):
                ele = row[1:][j].split(':')
                data[i][j] = ele[1]

    elif dataset in {'CT'}:
        df = pd.read_csv("data/slice_localization_data.csv")
        df = df.drop("patientId", axis=1)
        df = df.sample(frac=1).reset_index(drop=True)
        y = df["reference"].to_numpy()
        data = df.drop("reference", axis=1)

    elif dataset in {'abalone10', 'abalone30'}:
        k_default = 10
        file = open(f"data/{dataset}.pkl", 'rb')
        data = pickle.load(file)
        file.close()

    else:
        raise ValueError(f"Wrong dataset name: {dataset}.")

    # Get the Gram matrix A
    data = scale(data)
    slice = int(data.shape[0] * scaling)
    data = data[:slice]
    if y is not None:
        y = y[:slice]

    if kernel == 'rbf':
        a = kernels.dense_rbf_kernel(data, data, sigma=sigma)
    elif kernel == 'sparse_rbf':
        a = kernels.sparse_rbf_kernel(data, data, sigma=sigma)
    elif kernel == 'matern':
        a = kernels.matern_kernel(data, data, sigma=sigma, nu=nu)
    elif kernel == 'null':
        a = None
    else:
        raise ValueError(f"Wrong dataset name: {dataset}.")

    # Set the value of k to the default value since there is no input from command lines
    if not k:
        k = k_default

    x = data

    return a, k, x, y
