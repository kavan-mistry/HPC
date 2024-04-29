from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("Rank 0 is broadcasting")
else:
    print("Process : ", rank, " , is waiting to receive data from Rank 0")

data = np.empty(1, dtype=int)
if rank == 0:
    data[0] = np.random.randint(0, 100)
comm.Bcast(data, root=0)

print("Process : ", rank, "received data : ", data[0])

