## This code randomly select 64 training data and visualize them.
import os
import numpy as np
import random
from skimage import measure
import matplotlib.pyplot as plt

dir_path = '/Users/pengyuchen/Documents/Work/NETs_rotated_subset'

# Get a list of all files in the directory
file_list = os.listdir(dir_path)
# Select 64 random files from the directory
random_files = random.sample(file_list, 64)

# Create figure and subplots
fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(16, 16), subplot_kw={'projection': '3d'})
isosurface_value = 0.4
count = 0

# Loop over subplots
for i in range(8):
    for j in range(8):
        # Load the tensor from the current file
        tensor = np.loadtxt(os.path.join(dir_path, random_files[count])).reshape(32, 32, 32)

        # Create a 3D plot of the tensor
        if np.max(tensor) >= isosurface_value+0.05:
            verts, faces, normals, values = measure.marching_cubes(tensor, isosurface_value)
            axs[i, j].plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
            axs[i, j].set_xlim(0, 32)
            axs[i, j].set_ylim(0, 32)
            axs[i, j].set_zlim(0, 32)
            axs[i, j].set_xlabel('X')
            axs[i, j].set_ylabel('Y')
            axs[i, j].set_zlabel('Z')
        else:
            axs[i, j].plot([0],[0])

        count += 1

# Show the plot

save_path = os.path.join(dir_path, 'TrainingSets.png')
plt.savefig(save_path, dpi=300)
plt.clf()




