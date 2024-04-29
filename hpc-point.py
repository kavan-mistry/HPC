import mpi4py
from mpi4py import MPI

import numpy as np

comm = MPI.COMM_WORLD # get the communicator object
rank = comm.Get_rank() # get the rank of the current process
name = MPI.Get_processor_name() # get the name of the current processor
size = comm.Get_size() # get the number of processes
randNum = np.zeros(1)

if rank ==1:
    randNum = np.random.random_sample(1)
    print("process", rank,"drew the number", randNum[0])
    comm.Recv(randNum, dest=0)
    comm.Recv(randNum, source=0)
    print("process", rank,"received the number", randNum[0])
    
if rank ==0:
    print("process", rank,"before receiving has the number", randNum[0])
    comm.Recv(randNum, source=1)
    print("process", rank,"received the number", randNum[0])
    randNum *=2
    comm.Recv(randNum, source=0)