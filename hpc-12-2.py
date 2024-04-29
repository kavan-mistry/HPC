import mpi4py
from mpi4py import MPI

comm = MPI.COMM_WORLD # get the communicator object
rank = comm.Get_rank() # get the rank of the current process
name = MPI.Get_processor_name() # get the name of the current processor
size = comm.Get_size() # get the number of processes

if rank == 0:
    message = "Hello from process 0"
    comm.send(message, dest=1)
    
    received_message = comm.recv(source=1)
    print(f"Process 0 received message: {received_message}")
    
elif rank == 1:
    received_message = comm.recv(source=0)
    print(f"Process 1 received message: {received_message}")
    
    reply = "Hello from process 1"
    comm.send(reply, dest=0)