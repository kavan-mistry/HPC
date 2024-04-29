#!/usr/bin/env python
# coding: utf-8

# In[20]:


import mpi4py
from mpi4py import MPI


# In[21]:


import numpy as np
import cv2


# In[22]:


comm = MPI.COMM_WORLD # get the communicator object
rank = comm.Get_rank() # get the rank of the current process
name = MPI.Get_processor_name() # get the name of the current processor
size = comm.Get_size() # get the number of processes
universe_size = comm.Get_attr(MPI.UNIVERSE_SIZE) # get the expected number of processes


# In[23]:


img = cv2.imread('noice.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img


# In[24]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img1 = mpimg.imread("noice.png")
plt.imshow(img1)
plt.show()


# In[25]:


h, w = img.shape


# In[26]:


# divide the image into four equal parts
# each process will work on one part
x1 = rank * (w // 4)
x2 = (rank + 1) * (w // 4)
y1 = 0
y2 = h


# In[27]:


# apply non-local means denoising algorithm on the assigned part
# you can use other denoising algorithms as well
denoised = cv2.fastNlMeansDenoising(img[y1:y2, x1:x2], None, 10, 7, 21)


# In[28]:


# gather the denoised parts from all processes
# the root process (rank 0) will receive the results
denoised_parts = comm.gather(denoised, root=0)


# In[29]:


# if the current process is the root process
if rank == 0:
    # initialize an empty array to store the final denoised image
    final = np.zeros((h, w), dtype=np.uint8)

    # concatenate the denoised parts horizontally
    for i in range(size):
        x1 = i * (w // 4)
        x2 = (i + 1) * (w // 4)
        final[:, x1:x2] = denoised_parts[i]

    # save the final denoised image
    cv2.imwrite('denoised4.jpg', final)


# In[ ]:




