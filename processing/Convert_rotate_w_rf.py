##read file:
#
import os
## Test rotation and translation function
import os
import random
import numpy as np
from numpy import cos, pi, mgrid
import scipy

#extract monomer from in_filename(.rf) and output out_filename(.txt)
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
            A = np.array(first_col).reshape(32,32,32)
            second_col = [line.split()[1] for line in data]
            B = np.array(second_col).reshape(32, 32, 32)
            return A, B
    return

def translate_cell(cell_data_A, cell_data_B):
    x = random.randint(0, cell_data_A.shape[0] - 1)
    y = random.randint(0, cell_data_A.shape[1] - 1)
    z = random.randint(0, cell_data_A.shape[2] - 1)
    translated_A = np.roll(cell_data_A, (x, y, z), axis=(0, 1, 2))
    translated_B = np.roll(cell_data_B, (x, y, z), axis=(0, 1, 2))
    return translated_A, translated_B

def rotate_cell(cell_data_A, cell_data_B,max_angle):
    imageA_1 = np.squeeze(cell_data_A)
    imageB_1 = np.squeeze(cell_data_B)
    # rotate along z-axis
    angle = random.uniform(-max_angle, max_angle)
    imageA_2 = scipy.ndimage.rotate(imageA_1, angle, order=5, mode='grid-wrap', axes=(0, 1), reshape=False)
    imageB_2 = scipy.ndimage.rotate(imageB_1, angle, order=5, mode='grid-wrap', axes=(0, 1), reshape=False)
    # rotate along y-axis
    angle = random.uniform(-max_angle, max_angle)
    imageA_3 = scipy.ndimage.rotate(imageA_2, angle, order=3, mode='grid-wrap', axes=(0, 2), reshape=False)
    imageB_3 = scipy.ndimage.rotate(imageB_2, angle, order=3, mode='grid-wrap', axes=(0, 2), reshape=False)

    # rotate along x-axis
    angle = random.uniform(-max_angle, max_angle)
    rotated_image_A = scipy.ndimage.rotate(imageA_3, angle, order=3, mode='grid-wrap', axes=(1, 2), reshape=False)
    rotated_image_B = scipy.ndimage.rotate(imageB_3, angle, order=3, mode='grid-wrap', axes=(1, 2), reshape=False)
    return rotated_image_A, rotated_image_B

def write_file(output_filename, A, B):
    flatA = A.reshape(-1)
    flatB = B.reshape(-1)
    stacked = np.hstack((flatA[:, np.newaxis],flatB[:, np.newaxis]))
    np.savetxt(output_filename, stacked)


in_folder = '/Users/pengyuchen/Documents/Work/SCFT_Training_Sets/SD/symmetry'
out_folder = '/Users/pengyuchen/Documents/Work/NET_rotated_W'
for i in range(150):
    in_dir = os.path.join(in_folder,str(i),'w')
    files = os.listdir(in_dir)
    if files.count("w.rf")==1:
        for file in files:
            if file.endswith('.rf'):
                in_filename = os.path.join(in_dir,file)
                name, ext = os.path.splitext(file)
                name = 'symmetry_SD_' + str(i) + name + '.txt'
                out_filename = os.path.join(out_folder,name)
                #read the 32x32x32 tensors, translate and rotate them
                A, B = extract_data(in_filename)
                A = A.astype('float64')
                B = B.astype('float64')
                A_T, B_T = translate_cell(A, B)
                A_TR, B_TR = rotate_cell(A_T,B_T,45)
                #output the rotated images
                write_file(out_filename,A_TR,B_TR)