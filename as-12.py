from mpi4py import MPI
# Create MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
# Data to be sent from rank 0
send_data = None
if rank == 0:
    send_data = Particle(6, 46)
# Scatter data from rank 0 to other processes
recv_data = comm.scatter([send_data] * size, root=0)
# Print received data on each process
print(f"Process {rank} received data: x={recv_data.x}, y={recv_data.y}")
MPI.Finalize()
