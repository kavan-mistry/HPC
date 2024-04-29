from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

send_data = np.array([rank], dtype=np.int32)
recv_data = np.empty([size], dtype=np.int32)

comm.Allgather(send_data, recv_data)
print("Rank:", rank, "recieved", recv_data.tolist())