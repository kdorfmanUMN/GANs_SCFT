import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
import argparse
def rand_rotation_matrix(deflection=1.0, randnums=None):
#     """
#     Creates a random rotation matrix.
#
#     deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
#     rotation. Small deflection => small perturbation.
#     randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
#     """
#     # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
#
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M
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
            cell_data = data[:, 0].reshape(32, 32, 32)
            cell_data = cell_data.astype('float32')
            epsilon = 0.01
            if np.max(cell_data) <= (1.0 + epsilon) and np.min(cell_data) >= (0 - epsilon):
                cell_data = data[:,0].reshape(32,32,32)
                return cell_data
    return

def crop_rotate(density_data):

    # Define original grid and extend to [-1.5, 1.5] to make sure the new crop box is located in the field
    x_coords = np.linspace(-1.5, 1.5, 32*3)
    y_coords =  np.linspace(-1.5, 1.5, 32*3)
    z_coords =  np.linspace(-1.5, 1.5, 32*3)

    supercell = np.tile(density_data, (3, 3, 3))

    # Define crop size and translation
    crop_sizes = [1,1,1]
    crop_translation = np.random.rand(3) - 0.5

    # Define rotated cube cell
    x_crop = np.linspace(0, crop_sizes[0], 32) - 0.5
    y_crop = np.linspace(0, crop_sizes[1], 32) - 0.5
    z_crop = np.linspace(0, crop_sizes[2], 32) - 0.5

    xx, yy, zz = np.meshgrid(x_crop, y_crop, z_crop, indexing='ij')
    # Reshape the arrays to get a matrix of size (32x32x32, 3)
    crop_coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    rotate_coords = np.dot(crop_coords, rand_rotation_matrix())
    x_crop_rotated, y_crop_rotated, z_crop_rotated = rotate_coords.T

    # Define interpolator
    interpolator = RegularGridInterpolator((x_coords, y_coords, z_coords), supercell,method='quintic')

    new_grid = np.zeros((32, 32, 32))

    i, j, k = np.meshgrid(range(32), range(32), range(32), indexing='ij')

    # Calculate rotated and translated coordinates of crop points
    crop_coords = np.stack([x_crop_rotated, y_crop_rotated, z_crop_rotated], axis=-1)
    crop_coords_translated = crop_coords + crop_translation

    # Interpolate new density data
    new_grid = interpolator(crop_coords_translated)

    # Reshape new_grid to match original grid shape
    #return (32x32x32,) array
    return new_grid


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Process images in a folder')
    parser.add_argument('input_folder', type=str, help='Path to input folder containing images to process')
    parser.add_argument('output_folder', type=str, help='Path to output folder to save processed images')
    parser.add_argument('NETs_type', type=str, help='Name of the network structure')

    # Parse arguments
    args = parser.parse_args()

    # Get input and output folder paths
    in_folder = args.input_folder
    out_folder = args.output_folder
    NETs_type = args.NETs_type

    # Process images in input folder
    for i in range(150):
        in_dir = os.path.join(in_folder,str(i),'c')
        files = os.listdir(in_dir)
        if files.count("c.rf")==1:
            for file in files:
                if file.endswith('.rf'):
                    in_filename = os.path.join(in_dir,file)
                    name, ext = os.path.splitext(file)
                    name = NETs_type + str(i) + name + '.txt'
                    out_filename = os.path.join(out_folder, name)
                    # read the 32x32x32 tensor, translate and rotate it
                    A = extract_data(in_filename)
                    if not A is None:
                        # output the rotated images
                        A_T = crop_rotate(A)
                        np.savetxt(out_filename, A_T, delimiter='\n')
                        print(i)

if __name__ == '__main__':
    main()


# in_dir = '../../TrainingData/SCFT_generated/DG/P_1/'
# out_dir = '/Users/pengyuchen/Documents/Work/Test_random_crop/'
# for file in os.listdir(in_dir):
#     if file.endswith('.txt'):
#
# cell = extract_data(os.path.join(in_dir,'SG.rf'))
# A = crop_rotate(cell)
# with open(os.path.join(out_dir,  '1.rf'), 'w') as output_file:
#         # Write the data to the output file
#     np.savetxt(output_file, A, delimiter='\t', fmt='%.16f')
