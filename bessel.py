import math
import time
import numpy as np
import tensorflow as tf
import scipy.spatial.distance
from scipy.special import gamma, kv
from mpi4py import MPI
from logbesselk import log_bessel_k

from utils import dataloader, kernels, parser

block_max_size = 4000000

args = parser.parse_config("configs/bessel.yaml")
n_gpus = len(tf.config.list_physical_devices('GPU'))
print(f"dataset: {args.dataset}")

nu = args.nu
scalings = args.scaling
saved_times = np.zeros((2, len(scalings)))

for i, scaling in enumerate(scalings):
    gpu_times = []
    cpu_times = []
    for trial in range(args.trials):  # Repeat trials times
        # Init MPI
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        mpi_world_size = comm.Get_size()
        print(f"Total Nodes: {mpi_world_size} RANK: {mpi_rank} number of GPUS: {n_gpus}")

        if mpi_rank == 0:
            # Here we prepare the dataset and compute the rest of the Matern kernel on node 0
            a, k, x, _ = dataloader.general_loader(args.dataset, args.k, kernel=args.kernel,
                                                   sigma=args.sigma, nu=args.nu, scaling=scaling)

            x = np.tile(x, (5, 1))

            full_timer = time.time()
            ex = np.atleast_2d(x)

            # Other parts of matern kernel except Bessel function
            dists = scipy.spatial.distance.cdist(x / args.sigma, x / args.sigma, metric="euclidean")
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * nu) * K
            K.fill((2 ** (1.0 - nu)) / gamma(nu))
            K *= tmp ** nu

            tmp = tmp.reshape(-1, 1)
            ave, res = divmod(tmp.shape[0], mpi_world_size)
            count = np.array([ave + 1 if r < res else ave for r in range(mpi_world_size)])
            displ = np.array([sum(count[:r]) for r in range(mpi_world_size)])

            gpu_timer = time.time()

            print(f"The shape of dataset is {x.shape}, init time is {time.time() - full_timer}")

        else:
            tmp = None
            count = np.zeros(mpi_world_size, dtype=int)
            displ = None

        # MPI communications
        comm.Bcast(count, root=0)
        recv_buf = np.zeros(count[mpi_rank])

        scatter_timer = time.time()
        comm.Scatterv([tmp, count, displ, MPI.DOUBLE], recv_buf, root=0)
        print(f"Scattering TIME: {time.time() - scatter_timer}")

        for r in range(mpi_world_size):
            if mpi_rank == r:
                # Computer Bessel function with distributed nodes
                start_timer = time.time()
                if recv_buf.shape[0] > block_max_size:
                    partitions = np.array_split(recv_buf, np.ceil(recv_buf.shape[0] / block_max_size))
                    recv_buf = np.zeros(1)
                    for partition in partitions:
                        recv_buf = np.concatenate((recv_buf, log_bessel_k(v=nu, x=partition)))

                    recv_buf = np.exp(recv_buf[1:])

                else:
                    recv_buf = np.exp(log_bessel_k(v=nu, x=recv_buf))
                if mpi_rank == 0:
                    print(f"RANK: {mpi_rank} finish shape {recv_buf.shape} in {time.time() - start_timer}")

        # MPI communications
        send_buf_2 = recv_buf
        recv_buf_2 = np.zeros(sum(count))

        gather_timer = time.time()
        comm.Gatherv(send_buf_2, [recv_buf_2, count, displ, MPI.DOUBLE], root=0)
        print(f"Gather TIME: {time.time() - gather_timer}")

        if mpi_rank == 0:
            # compute matern kernel with sklearn for validation
            print(f"GPU TIME {time.time() - gpu_timer}")
            reshaped = np.reshape(recv_buf_2, (K.shape[0], K.shape[1]))
            gpu_matern = K * reshaped
            print(f"FULL TIME: {time.time() - full_timer}")

            cpu_timer = time.time()
            a, k, x = dataloader.general_loader(args.dataset, args.k,
                                            kernel=args.kernel, sigma=args.sigma, nu=args.nu, scaling=scaling)
            x = np.atleast_2d(x)
            x = np.tile(x, (5, 1))

            dists = scipy.spatial.distance.cdist(x / args.sigma, x / args.sigma, metric="euclidean")
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * nu) * K
            K.fill((2 ** (1.0 - nu)) / gamma(nu))
            K *= tmp ** nu
            cpu_matern = K * kv(nu, tmp)
            print(f"CPU: t={time.time() - cpu_timer}")

            # Check the accuracy of the proposed function
            print("2 norm: ", np.linalg.norm(gpu_matern - cpu_matern, 2))
            print("f norm: ", np.linalg.norm(gpu_matern - cpu_matern, 'fro'))
            print("f norm: ", np.linalg.norm(gpu_matern - cpu_matern, 'nuc'))

    break

