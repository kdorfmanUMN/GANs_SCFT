import os
import random
import numpy as np
import scipy
import argparse

#extract monomer density from in_filename(.rf)
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
            with open(in_filename, 'r') as f:
                lines = f.readlines()

            # find the index of the line with "ngrid"
            ngrid_idx = lines.index('ngrid\n')

            # extract the data 2 lines after the line with "ngrid"
            data = lines[ngrid_idx + 2:]

            # extract the first column of data
            first_col = [line.split()[0] for line in data]
            cell_data = np.array(first_col).astype('float32')
            if np.max(cell_data)<=1 and np.min(cell_data) >=0:
                cell_data = cell_data.reshape(32,32,32)
                return cell_data
    return

def translate_cell(cell_data):
    x = random.randint(0, cell_data.shape[0] - 1)
    y = random.randint(0, cell_data.shape[1] - 1)
    z = random.randint(0, cell_data.shape[2] - 1)
    translated_cell = np.roll(cell_data, (x, y, z), axis=(0, 1, 2))
    return translated_cell

def rotate_cell(cell_data,max_angle):
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
                        A_T = translate_cell(A)
                        A_TR = rotate_cell(A_T, 45)
                        # output the rotated images
                        A_TR = A_TR.reshape(-1)
                        np.savetxt(out_filename, A_TR, delimiter='\n')

if __name__ == '__main__':
    main()
