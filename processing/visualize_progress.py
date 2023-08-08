import os
import torch
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
# Directory containing .pt files
dir_path = '/Users/pengyuchen/Documents/GAN/past_data/output_images_DG_SG_SD_notranslation_rotation'
# Loop over all .pt files in directory
for file_name in os.listdir(dir_path):
    if file_name.endswith('.pt'):
        # Load .pt file and convert to numpy array
        fig_name = os.path.splitext(file_name)[0] + '.png'
        save_path = os.path.join(dir_path, 'isosurface_plots_2', fig_name)
        if os.path.exists(save_path) != 1:
            file_path = os.path.join(dir_path, file_name)
            x = torch.load(file_path).numpy()
            print(file_name)

            # Create figure and subplots
            fig, axs = plt.subplots(nrows=6, ncols=10, figsize=(10, 6), subplot_kw={'projection': '3d'})
            isosurface_value = 0.3
            count = 0

            # Adjust padding between subplots
            fig.subplots_adjust(hspace=0.0, wspace=0)

            # Loop over subplots
            for i in range(6):
                for j in range(10):
                    count = i * 10 + j
                    if np.max(x[count, 0]) >= isosurface_value+0.01:
                        verts, faces, normals, values = measure.marching_cubes(x[count, 0], isosurface_value)
                        axs[i, j].plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
                        axs[i, j].set_xlim(0, 32)
                        axs[i, j].set_ylim(0, 32)
                        axs[i, j].set_zlim(0, 32)
                        axs[i, j].set_xlabel('X')
                        axs[i, j].set_ylabel('Y')
                        axs[i, j].set_zlabel('Z')
                        axs[i, j].set_axis_off()  # hide the axes label
                    else:
                        axs[i, j].plot([0],[0])
                    # Adjust size of subplot
                    #axs[i, j].get_proj = lambda: np.dot(Axes3D.get_proj(axs[i, j]), np.diag([1.5, 1.5, 1.5, 1]))
                    count += 1

            # Save figure as .png with same name as .pt file
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            plt.clf()
