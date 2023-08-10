import os
import torch
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np

class IsosurfaceVisualizer:

    def __init__(self, isosurface_value=0.5):
        self.isosurface_value = isosurface_value

    def visualize(self, file_path, save_path):
        """
        Visualize one .pt file and save it to the specified path.

        Parameters:
        - file_path: file path for the .pt file.
        - save_path: file path for the output image.
        """

        # Load .pt file and convert to numpy array
        x = torch.load(file_path).numpy()

        # Determine the number of subplots based on the tensor's first dimension
        num_images = x.shape[0]
        nrows = int(np.ceil(np.sqrt(num_images)))
        ncols = nrows

        # Create figure and subplots
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10), subplot_kw={'projection': '3d'})
        fig.subplots_adjust(hspace=0.0, wspace=0)

        # Loop over subplots
        for i in range(nrows):
            for j in range(ncols):
                count = i * ncols + j
                if count < num_images and np.max(x[count, 0]) >= self.isosurface_value + 0.01:
                    verts, faces, _, _ = measure.marching_cubes(x[count, 0], self.isosurface_value)
                    axs[i, j].plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
                    axs[i, j].set_xlim(0, 32)
                    axs[i, j].set_ylim(0, 32)
                    axs[i, j].set_zlim(0, 32)
                    axs[i, j].set_axis_off()
                else:
                    axs[i, j].plot([0], [0])
                    axs[i, j].set_axis_off()

        # Save figure as .png
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def visualize_directory(self, dir_path, output_dir):
        """
        Visualize all .pt files in a directory and save them in the specified output directory.

        Parameters:
        - dir_path: Directory containing the .pt files.
        - output_dir: Directory where the output docs will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file_name in os.listdir(dir_path):
            if file_name.endswith('.pt'):
                input_file_path = os.path.join(dir_path, file_name)
                fig_name = os.path.splitext(file_name)[0] + '.png'
                save_path = os.path.join(output_dir, fig_name)
                if not os.path.exists(save_path):
                    self.visualize(input_file_path, save_path)
