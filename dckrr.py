import time
import math
import sklearn
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model._ridge import _solve_cholesky_kernel
import scipy.spatial.distance
from scipy.special import gamma
from mpi4py import MPI
from utils import dataloader
from utils.parser import parse_config
from logbesselk import log_bessel_k


args = parse_config("configs/krr.yaml")

for _ in range(args.trials):
    # Initialize MPI info
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_world_size = comm.Get_size()

    if mpi_rank == 0:
        # Prepare dataset and distribution process
        a, k, x, y = dataloader.general_loader(args.dataset, args.k, kernel=args.kernel, sigma=5, nu=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=11)
        n_samples, n_features = x_train.shape

        ave, res = divmod(n_samples, mpi_world_size)
        count = np.array([ave + 1 if r < res else ave for r in range(mpi_world_size)])
        count_x = count * n_features
        count_y = count
        displacements_x = np.array([sum(count_x[:r]) for r in range(mpi_world_size)])
        displacements_y = np.array([sum(count_y[:r]) for r in range(mpi_world_size)])

    else:
        x_train = None
        y_train = None
        x_test = None
        y_test = None
        count_x = np.zeros(mpi_world_size, dtype=int)
        count_y = np.zeros(mpi_world_size, dtype=int)
        displacements_x = None
        displacements_y = None

    # Distributing
    comm.Bcast(count_x, root=0)
    comm.Bcast(count_y, root=0)
    x_train_local = np.zeros(count_x[mpi_rank])
    y_train_local = np.zeros(count_y[mpi_rank])

    communication_timer = time.time()
    comm.Scatterv([x_train, count_x, displacements_x, MPI.DOUBLE], x_train_local, root=0)
    comm.Scatterv([y_train, count_y, displacements_y, MPI.DOUBLE], y_train_local, root=0)

    # comm.Bcast(x_test, root=0)
    # comm.Bcast(y_test, root=0)
    print(f"COMM time: {time.time() - communication_timer}")

    for r in range(mpi_world_size):
        # Action on each node
        if mpi_rank == r:
            fit_timer = time.time()

            x_train_local = x_train_local.reshape(count_y[mpi_rank], -1)  # reshape it back
            print(f"RANK: {mpi_rank}, shape of training: {x_train_local.shape}")

            dists = scipy.spatial.distance.cdist(x_train_local / 5, x_train_local / 5, metric="euclidean")
            K = dists
            nu = 1
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * nu) * K
            K.fill((2 ** (1.0 - nu)) / gamma(nu))
            K *= tmp ** nu

            partitions = np.array_split(tmp, 8, axis=0)
            recv_buf = np.zeros(1)
            for partition in partitions:
                recv_buf = np.concatenate((recv_buf, log_bessel_k(v=nu, x=partition)))

            tmp = np.exp(recv_buf[1:])
            K *= tmp

            # Perform Cholesky for KRR
            dual_coef_ = _solve_cholesky_kernel(K, y_train_local, 0.1)

            print(f"Training finished in {time.time() - fit_timer}")
