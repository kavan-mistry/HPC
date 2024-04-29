from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_sum = np.random.randint(0, 100)

if rank == 0:
    print("Rank 0 is gathering")

global_sums = None
if rank == 0:
    global_sums = np.empty(size, dtype=int)
comm.Gather(np.array(local_sum, dtype=int), global_sums, root=0)

if rank == 0:
    print("Rank 0 gathered the following local sums : ", global_sums)