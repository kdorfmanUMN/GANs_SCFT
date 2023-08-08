import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
import argparse

def extract_data(in_filename):
    # check if the file has the ".rf" extension
    if not in_filename.endswith('.rf'):
        print('File is not in the correct format')
    else:
        # check if the file exists
        if not os.path.exists(in_filename):
            print(f"File {in_filename} not found")
        else:
            # open the file
            data = np.loadtxt(in_filename,skiprows=15)
            cell_data = data[:, 0].reshape(64, 64, 64)
            cell_data = cell_data.astype('float32')
            epsilon = 0.01
            if np.max(cell_data) <= (1.0 + epsilon) and np.min(cell_data) >= (0 - epsilon):
                cell_data = data[:,0].reshape(64, 64, 64)
                return cell_data
    return

def crop_new_cell(density_data,new_cell):
    a = 3.178434
    b = 4.870116
    c = 1.835086
    # Define original grid and extend to [-1.5, 1.5] to make sure the new crop box is located in the field
    x_coords = np.linspace(-2*a, 3*a, 64*5)
    y_coords =  np.linspace(-2*b, 3*b, 64*5)
    z_coords =  np.linspace(-2*c, 3*c, 64*5)

    supercell = np.tile(density_data, (5,5,5))
    # Define interpolator
    interpolator = RegularGridInterpolator((x_coords, y_coords, z_coords), supercell,method='cubic')
    # Define crop size and translation
    grid = 32
    # Define rotated cube cell
    origin = np.array([0.5*a,0/32*b,c-c/32])
    new_x_start = origin
    new_y_start = origin
    new_z_start = origin
    new_x_end = origin + np.array([a/2,0,-c/2])
    new_y_end = origin + np.array([0,0,c])
    new_z_end = origin + np.array([0,-b,0])
    count = 0
    new_coords = np.zeros([grid**3,3])
    for k in range(grid):
        for j in range(grid):
            for i in range(grid):
                new_coords[count,:]= (new_x_end-new_x_start)*i/grid + (new_y_end-new_y_start)*j/grid + (new_z_end-new_z_start)*k/grid + origin
                count +=1


    x_crop_rotated, y_crop_rotated, z_crop_rotated = new_coords.T

    new_grid = np.zeros((grid, grid, grid))

    i, j, k = np.meshgrid(range(grid), range(grid), range(grid), indexing='ij')

    # Calculate rotated and translated coordinates of crop points
    crop_coords = np.stack([x_crop_rotated, y_crop_rotated, z_crop_rotated], axis=-1)

    # Interpolate new density data
    new_grid = interpolator(crop_coords)

    # Reshape new_grid to match original grid shape
    #return (32x32x32,) array
    return new_grid.reshape(-1,1)

input_file_path = '/Users/pengyuchen/Documents/Work/TrainingOutputs/0331_Small/45_15/Convert_Cells/574.rf'
output_file_path = '/Users/pengyuchen/Documents/Work/TrainingOutputs/0331_Small/45_15/Convert_Cells/574_t.rf'
new_cell = np.array([[0,0,0], [0.5,0,-0.5],[0, 0, 0],[0,0,-1],[0,0,0], [0, 1, 0]])
A = extract_data(input_file_path)
columnA = crop_new_cell(A,new_cell)
columnB = 1 - columnA

with open(output_file_path, 'w') as output_file:
    # Add the header to the output file
    header = 'format   1   0\ndim\n          3\ncrystal_system\n              hexagonal\nN_cell_param\n              3\ncell_param    \n      3.0e+00   6.0e+00  \ngroup_name\n          P_1\nN_monomer\n          2\nngrid\n                16 16 16\n'
    output_file.write(header)

with open(output_file_path, 'a') as output_file:

    #Write the data to the output file
    AB = np.hstack((columnA, columnB))
    np.savetxt(output_file, AB, delimiter='\t', fmt='%.16f')