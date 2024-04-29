import mpi4py
from mpi4py import MPI

comm = MPI.COMM_WORLD # get the communicator object
rank = comm.Get_rank() # get the rank of the current process
name = MPI.Get_processor_name() # get the name of the current processor
size = comm.Get_size() # get the number of processes
universe_size = comm.Get_attr(MPI.UNIVERSE_SIZE) # get the expected number of processes

print("Welcome to PDPU", "Process name:", name , "Process id:", rank , "Number of cores:", size)