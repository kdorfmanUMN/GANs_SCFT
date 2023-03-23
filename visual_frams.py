#Visualize the output images
import torch
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np

file = '/Users/pongwu/Documents/Work/UMN/2023/GANs_NETs/output_images_translated/3_27.pt'
x = torch.load(file).numpy()
print(x.shape)
# #


#
for i in range(x.shape[0]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    isosurface_value = 0.3

    verts, faces, normals, values = measure.marching_cubes(x[i, 0], isosurface_value)
    # Plot isosurface
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
    # Set plot limits and labels
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Save plot as a PNG image
    fig.savefig(f'isosurface_{i}.png', dpi=300)
    # Close the plot to free up memory
    plt.close(fig)




#Plot some training images
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# isosurface_value = 0.3
# # verts, faces, normals, values = measure.marching_cubes(x[7][0], isosurface_value)
# # Plot isosurface
# ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
# # Set plot limits and labels
# ax.set_xlim(0, 32)
# ax.set_ylim(0, 32)
# ax.set_zlim(0, 32)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # Show plot
# plt.show()

# #
# for i in range(x.shape[0]):
#     w = x[i][0]
#     np.savetxt('/Users/pongwu/Documents/Work/UMN/2023/GANs_NETs/output_images_translated/5_290/'+str(i)+'.rf',w.reshape(-1))




