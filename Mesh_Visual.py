import torch
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

file = '/Users/pongwu/Documents/Work/UMN/2023/GANs_NETs/rotated_test/symmetry_SD_28c_21.txt'
x = np.loadtxt(file)
x = x.reshape(32,32,32)

x = np.concatenate([x, x], axis=0)
x = np.concatenate([x, x], axis=1)
x = np.concatenate([x, x], axis=2)
# #
isosurface_value = 0.8
verts, faces, normals, values = measure.marching_cubes(x, isosurface_value)
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])


# mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
#            center=mesh.get_center())
# o3d.visualization.draw_geometries([mesh])

# print('voxelization')
# voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
#                                                               voxel_size=0.04)
