import mpi4py
from mpi4py import MPI
import time

import numpy as np

comm = MPI.COMM_WORLD # get the communicator object
rank = comm.Get_rank() # get the rank of the current process
name = MPI.Get_processor_name() # get the name of the current processor
size = comm.Get_size() # get the number of processes

randNum = np.zeros(1)

if rank == 0:
    message = "Hello from process 0 (Async)"
    req_send = comm.isend(message, dest=1)  # Non-blocking send
    print(f"Process {rank} sent message: {message}")
    time.sleep(1)  # Simulate some other task
    req_send.wait()  # Wait for the send operation to complete
elif rank == 1:
    req_recv = comm.irecv(source=0)  # Non-blocking receive
    time.sleep(0.5)  # Simulate some other task
    print(f"Process {rank} waiting to receive message...")
    received_message = req_recv.wait()  # Wait for the receive operation to complete
    print(f"Process {rank} received message: {received_message}")