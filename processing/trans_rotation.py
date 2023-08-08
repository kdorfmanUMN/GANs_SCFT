## Test rotation and translation function
import os
import random
import numpy as np
# import pyvista as pv
from numpy import cos, pi, mgrid
import scipy
import torch


def translate_cell(cell_data):
    x = random.randint(0, cell_data.shape[0] - 1)
    y = random.randint(0, cell_data.shape[1] - 1)
    z = random.randint(0, cell_data.shape[2] - 1)
    temp = np.roll(cell_data, (x, y, z), axis=(0, 1, 2))
    return temp


# def rand_rotation_matrix(deflection=1.0, randnums=None):
#     """
#     Creates a random rotation matrix.
#
#     deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
#     rotation. Small deflection => small perturbation.
#     randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
#     """
#     # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
#
#     if randnums is None:
#         randnums = np.random.uniform(size=(3,))
#
#     theta, phi, z = randnums
#
#     theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
#     phi = phi * 2.0*np.pi  # For direction of pole deflection.
#     z = z * 2.0*deflection  # For magnitude of pole deflection.
#
#     # Compute a vector V used for distributing points over the sphere
#     # via the reflection I - V Transpose(V).  This formulation of V
#     # will guarantee that if x[1] and x[2] are uniformly distributed,
#     # the reflected points will be uniform on the sphere.  Note that V
#     # has length sqrt(2) to eliminate the 2 in the Householder matrix.
#
#     r = np.sqrt(z)
#     Vx, Vy, Vz = V = (
#         np.sin(phi) * r,
#         np.cos(phi) * r,
#         np.sqrt(2.0 - z)
#         )
#
#     st = np.sin(theta)
#     ct = np.cos(theta)
#
#     R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
#
#     # Construct the rotation matrix  ( V Transpose(V) - I ) R.
#
#     M = (np.outer(V, V) - np.eye(3)).dot(R)
#     return M

# def rotate_cell(cell_data,M):
#     offset = (np.array(cell_data.shape)-M.dot(np.array(cell_data.shape)))
#     offset = offset/2
#     f_data = scipy.ndimage.affine_transform(cell_data, M,
#                               output_shape=cell_data.shape, offset=offset)
#     return f_data

def rotate_cell(cell_data, max_angle):
    image1 = np.squeeze(cell_data)
    # rotate along z-axis
    angle = random.uniform(-max_angle, max_angle)
    image2 = scipy.ndimage.rotate(image1, angle, order=5, mode='grid-wrap', axes=(0, 1), reshape=False)

    # rotate along y-axis
    angle = random.uniform(-max_angle, max_angle)
    image3 = scipy.ndimage.rotate(image2, angle, order=3, mode='grid-wrap', axes=(0, 2), reshape=False)

    # rotate along x-axis
    angle = random.uniform(-max_angle, max_angle)
    rotated_image = scipy.ndimage.rotate(image3, angle, order=3, mode='grid-wrap', axes=(1, 2), reshape=False)

    return rotated_image


path = '/Users/pongwu/Documents/Work/UMN/2023/GANs_NETs/Image_test'
outpath = '/Users/pongwu/Documents/Work/UMN/2023/GANs_NETs/rotated_test'
for file in os.listdir(path):
    if file.endswith('.txt'):
        v1 = np.loadtxt(os.path.join(path, file))
        v1 = v1.reshape([32, 32, 32])
        v1 = translate_cell(v1)
        v2 = rotate_cell(v1, 45)
        v3 = v2.reshape(-1)
        np.savetxt(os.path.join(outpath, file), v3)

# x, y, z = pi * mgrid[0:1:32j, 0:1:32j, 0:1:32j]
# grid = pv.StructuredGrid(x, y, z)
# grid["vol"] = v1.flatten()
# contours = grid.contour([0.8])
# pv.set_plot_theme('document')
# # p = pv.Plotter()
# # grid.plot(multi_colors=True, cpos=[-2, 5, 3],show_axes=False,show_bounds=False)
# contours.plot(color='tan', show_edges=False, smooth_shading=True, cpos=[-2, 5, 3])

# file = open("file.txt", "w")
# # Saving the 2D array in a text file
# content = str(grid["vol"])
# file.write(content)
# file.close()
