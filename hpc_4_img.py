import numpy as np
import time
from scipy.signal import medfilt2d
from PIL import Image
import matplotlib.pyplot as plt
import mpi4py
from mpi4py import MPI

def denoise_image(image):
    if len(image.shape) not in [2, 3]:
        raise ValueError("Invalid image format")
    # If image is RGB, apply denoising separately to each channel
    if len(image.shape) == 3:
        denoised_image = np.stack([medfilt2d(channel, kernel_size=3) for channel in image.transpose(2, 0, 1)], axis=-1)
    else:
        denoised_image = medfilt2d(image, kernel_size=3)
    return denoised_image

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

image_path = ["noice.png","noise2.png","noise3.png","noise4.jpg"]

images = [np.array(Image.open(path)) for path in image_path]

image_chunks = list(chunks(images, len(images) // size))

# Scatter the chunks to each process
local_chunks = comm.scatter(image_chunks, root=0)

# Apply denoising algorithm to each chunk of images
denoised_chunks = [denoise_image(chunk) for chunk in local_chunks]

# Gather denoised chunks from all processes
all_denoised_chunks = comm.gather(denoised_chunks, root=0)

if rank == 0:
    # Combine denoised chunks into a single list of denoised images
    denoised_images = [image for sublist in all_denoised_chunks for image in sublist]

    # Plot the input and denoised images for each image
    for i, (input_image, denoised_image) in enumerate(zip(images, denoised_images)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(input_image, cmap='gray')
        axes[0].set_title(f'Input Image {i+1}')
        axes[0].axis('off')
        axes[1].imshow(denoised_image, cmap='gray')
        axes[1].set_title(f'Denoised Image {i+1}')
        axes[1].axis('off')
        plt.show()
        
start_time = MPI.Wtime()
end_time = MPI.Wtime()

print("Process", rank, "took", end_time - start_time, "seconds")

