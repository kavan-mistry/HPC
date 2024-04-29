from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("Rank 0 is scattering")
else:
    print("Process : ", rank, " , is waiting to receive scattered data from Rank 0")

if rank == 0:
    send_data = np.arange(size) * 10  
else:
    send_data = None

recv_data = np.empty(1, dtype=int)
comm.Scatter(send_data, recv_data, root=0)

print("Process : ", rank, "received data : ", recv_data[0])