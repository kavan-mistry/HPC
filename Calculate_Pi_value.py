#!/usr/bin/env python
# coding: utf-8

# In[10]:


from mpi4py import MPI
import random
import time


# In[ ]:


def calculate_pi(rank, num_processes, terms):
    partial_sum = 0.0
    for i in range(rank, terms, num_processes):
        if i % 2 == 0:
            partial_sum += 1.0 / (2 * i + 1)
        else:
            partial_sum -= 1.0 / (2 * i + 1)
    return partial_sum * 4

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    terms = 100000000

    start_time = time.time()

    partial_pi = calculate_pi(rank, size, terms)
    print(f"Process {rank} calculated: Pi = {partial_pi}, Time = {time.time() - start_time} seconds")

    if rank == 0:
        total_pi = partial_pi
        for i in range(1, size):
            partial_result, partial_time = comm.recv(source=i)
            total_pi += partial_result
            print(f"Process {i} received: Pi = {partial_result}, Time = {partial_time} seconds")

        print("Number of processes:", size)
        print("Estimated Pi:", total_pi)
        print("Execution time:", time.time() - start_time, "seconds")
    else:
        comm.send((partial_pi, time.time() - start_time), dest=0)


# In[ ]:




