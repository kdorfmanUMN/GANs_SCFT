import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
import argparse

class DataProcessor:
    def __init__(self):
        pass

    def rand_rotation_matrix(self, deflection=1.0, randnums=None):
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

        theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0 * np.pi  # For direction of pole deflection.
        z = z * 2.0 * deflection  # For magnitude of pole deflection.

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

    def extract_data(self, in_filename):
        # check if the file has the ".rf" extension
        if not in_filename.endswith('.rf'):
            print('File is not in the correct format')
        else:
            # check if the file exists
            if not os.path.exists(in_filename):
                print(f"File {in_filename} not found")
            else:
                # open the file
                data = np.loadtxt(in_filename, skiprows=15)
                cell_data = data[:, 0].reshape(32, 32, 32)
                cell_data = cell_data.astype('float32')
                epsilon = 0.01
                if np.max(cell_data) <= (1.0 + epsilon) and np.min(cell_data) >= (0 - epsilon):
                    cell_data = data[:, 0].reshape(32, 32, 32)

                    # Read dimensions from the file
                    with open(in_filename, 'r') as file:
                        lines = file.readlines()
                        dimensions = tuple(map(int, lines[14].split()))
                    return cell_data, dimensions
        return

    def crop_rotate(self, density_data, dimensions, new_grid_size):

        # Define original grid and extend to [-1.5, 1.5] to make sure the new crop box is located in the field
        x_coords = np.linspace(-1.5, 1.5, dimensions[0] * 3)
        y_coords = np.linspace(-1.5, 1.5, dimensions[1] * 3)
        z_coords = np.linspace(-1.5, 1.5, dimensions[2] * 3)

        supercell = np.tile(density_data, (3, 3, 3))

        # Define crop size and translation
        crop_sizes = [1, 1, 1] # crop_sizes = np.random.uniform(1, 1.732, size=(3, 1))
        crop_translation = np.random.rand(3) - 0.5

        # Define rotated cube cell
        x_crop = np.linspace(0, crop_sizes[0], new_grid_size[0]) - crop_sizes[0] / 2
        y_crop = np.linspace(0, crop_sizes[1], new_grid_size[1]) - crop_sizes[1] / 2
        z_crop = np.linspace(0, crop_sizes[2], new_grid_size[2]) - crop_sizes[2] / 2

        xx, yy, zz = np.meshgrid(x_crop, y_crop, z_crop, indexing='ij')
        # Reshape the arrays to get a matrix of size (32x32x32, 3)
        crop_coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        rotate_coords = np.dot(crop_coords, self.rand_rotation_matrix())
        x_crop_rotated, y_crop_rotated, z_crop_rotated = rotate_coords.T

        # Define interpolator
        interpolator = RegularGridInterpolator((x_coords, y_coords, z_coords), supercell, method='quintic')

        # Calculate rotated and translated coordinates of crop points
        crop_coords = np.stack([x_crop_rotated, y_crop_rotated, z_crop_rotated], axis=-1)
        crop_coords_translated = crop_coords + crop_translation

        # Interpolate new density data
        new_grid = interpolator(crop_coords_translated)

        return new_grid

    def process_files(self, in_filename, out_filename, new_grid_size):
        # Implementation of the main processing logic
        cell_data, dimensions = self.extract_data(in_filename)
        if cell_data is not None:
            cell_data_T = self.crop_rotate(cell_data, dimensions, new_grid_size)
            np.savetxt(out_filename, cell_data_T, delimiter='\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images for training')
    parser.add_argument('in_filename', type=str, help='Path to input image')
    parser.add_argument('out_filename', type=str, help='Path to output image')
    parser.add_argument('--grid', nargs=3, type=int, default=[32, 32, 32], help='Size of the new grid')
    args = parser.parse_args()

    data_processor = DataProcessor()
    data_processor.process_files(args.in_filename, args.out_filename, tuple(args.grid))