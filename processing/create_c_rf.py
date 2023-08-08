import torch
import numpy as np

# Open the input and output files
input_file_path = '/Users/pengyuchen/Documents/Work/TrainingOutputs/0325A/199_120.pt'

data = torch.load(input_file_path).numpy()

for i in range(64):
    # Construct the input and output file paths
    output_file_path = f'/Users/pengyuchen/Documents/Work/TrainingOutputs/0325A/199_120/{i}.rf'

    with open(output_file_path, 'w') as output_file:
        # Add the header to the output file
        header = 'format   1   0\ndim\n          3\ncrystal_system\n              orthorhombic\nN_cell_param\n              3\ncell_param    \n      3.0e+00   3.0e+00  3.0e+00 \ngroup_name\n          P_1\nN_monomer\n          2\nngrid\n                   32        32        32\n'
        output_file.write(header)

    with open(output_file_path, 'a') as output_file:

        #Write the data to the output file
        columnA = data[i][0].reshape(-1,1)
        columnB = 1 - columnA
        AB = np.hstack((columnA, columnB))
        np.savetxt(output_file, AB, delimiter='\t', fmt='%.16f')