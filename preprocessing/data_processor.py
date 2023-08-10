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
        theta, phi, z = 2.0 * np.array(
            [deflection * randnums[0] * np.pi, randnums[1] * np.pi, deflection * randnums[2]])
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
            return

        if not os.path.exists(in_filename):
            print(f"File {in_filename} not found")
            return

        with open(in_filename, 'r') as file:
            lines = file.readlines()
            dimensions = tuple(map(int, lines[14].split()))

        if len(dimensions) != 3:
            return

        data = np.loadtxt(in_filename, skiprows=15)
        cell_data = data[:, 0].reshape(dimensions).astype('float32')
        epsilon = 0.01

        if not np.max(cell_data) <= (1.0 + epsilon) and np.min(cell_data) >= (0 - epsilon):
            return

        return cell_data, dimensions

    def crop_rotate(self, density_data, dimensions, new_grid_size):
        """
        Crops and rotates the data.
        """
        # Tile the original cell to a supercell
        coords = np.linspace(-2, 2,
                             dimensions[0] * 4)  # need slight modification if original Ngrid on three axes are not identical.
        supercell = np.tile(density_data, (4, 4, 4))
        # Set up interpolator using the supercell.
        interpolator = RegularGridInterpolator((coords, coords, coords), supercell, method='quintic')

        # Define size of the cropped region. [1, 1, 1] is identical to the original cell.
        crop_sizes = [1, 1, 1]

        # Compute grid coordinates for the cropped region.
        crop = [np.linspace(0, s, g) - s / 2 for s, g in zip(crop_sizes, new_grid_size[::-1])]
        crop_coords = np.stack(np.meshgrid(*crop, indexing='ij'), axis=-1)

        # Apply random translation and rotation to grid coordinates to get new grids
        crop_translation = np.random.rand(3) - 0.5
        rotate_coords = np.dot(crop_coords.reshape(-1, 3), self.rand_rotation_matrix())
        crop_coords_translated = rotate_coords + crop_translation

        # Interpolate data on new grid
        new_grid = interpolator(crop_coords_translated)

        return new_grid

    def process_files(self, in_filename, out_filename, new_grid_size=(32,32,32)):
        """
        Main processing logic. Extracts data from the input file, crops and rotates it, and saves it to the output file.
        """
        cell_data, dimensions = self.extract_data(in_filename)
        if cell_data is not None:
            cell_data_T = self.crop_rotate(cell_data, dimensions, new_grid_size)
            np.savetxt(out_filename, cell_data_T, delimiter='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process docs for training')
    parser.add_argument('--in_filename', type=str, help='Path to input image')
    parser.add_argument('--out_filename', type=str, help='Path to output image')
    parser.add_argument('--grid', nargs=3, type=int, default=[32, 32, 32], help='Size of the new grid')
    args = parser.parse_args()

    data_processor = DataProcessor()
    data_processor.process_files(args.in_filename, args.out_filename, tuple(args.grid))
