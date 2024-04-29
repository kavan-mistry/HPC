import mpi4py
from mpi4py import MPI

comm = MPI.COMM_WORLD # get the communicator object
rank = comm.Get_rank() # get the rank of the current process
name = MPI.Get_processor_name() # get the name of the current processor
size = comm.Get_size() # get the number of processes

a = 10
b = 5

if rank == 0:
    print('rank = ', rank, ', addition : ' ,a+b)
if rank == 1:
    print('rank = ', rank, ', multiplication :' ,a*b)
if rank == 2:
    print('rank = ', rank, ', division : ' ,a/b)
if rank == 3:
    print('rank = ', rank, ', subtraction : ' ,a-b)