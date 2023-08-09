import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
import argparse


class DataProcessor:
    @staticmethod
    def rand_rotation_matrix(deflection=1.0, randnums=None):
        """
        Creates a random rotation matrix.

        deflection: the magnitude of the rotation (0: no rotation, 1: full rotation)
        randnums: 3 random numbers in [0, 1]. Auto-generated if None.
        """
        # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        if randnums is None:
            randnums = np.random.uniform(size=(3,))
        theta, phi, z = 2.0 * np.array([deflection * randnums[0] * np.pi, randnums[1] * np.pi, deflection * randnums[2]])
        r = np.sqrt(z)
        V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))
        st, ct = np.sin(theta), np.cos(theta)
        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
        M = (np.outer(V, V) - np.eye(3)).dot(R)
        return M


    @staticmethod
    def extract_data(in_filename):
        """
        Extracts data from the given file if it exists and has a valid format.
        """
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
        """
        Crops and rotates the data.
        """
        coords = np.linspace(-2, 2, dimensions[0] * 4) #need slight modification if Ngrid on three axes are not identical.
        supercell = np.tile(density_data, (4, 4, 4))
        crop_sizes = [1, 1, 1]
        crop_translation = np.random.rand(3) - 0.5

        crop = [np.linspace(0, s, g) - s / 2 for s, g in zip(crop_sizes, new_grid_size)]
        crop_coords = np.stack(np.meshgrid(*crop, indexing='ij'), axis=-1)
        rotate_coords = np.dot(crop_coords.reshape(-1, 3), self.rand_rotation_matrix())

        interpolator = RegularGridInterpolator((coords, coords, coords), supercell, method='quintic')
        crop_coords_translated = rotate_coords + crop_translation
        new_grid = interpolator(crop_coords_translated)

        return new_grid
    def process_files(self, in_filename, out_filename, new_grid_size):
        """
        Main processing logic.
        """
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
