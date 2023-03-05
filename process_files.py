#This file edits the .rf outputs from SCFT and output .txt file with the density data of the first monomer
import os

#extract monomer from in_filename(.rf) and output out_filename(.txt)
def extract_monomer_density(in_filename, out_filename):
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

            # write the first column of data to a text file
            with open(out_filename, 'w') as f:
                f.write('\n'.join(first_col))
    return

in_folder = '/Users/pengyuchen/Documents/Work/SCFT_Training_Sets/SD/symmetry'
out_folder = '/Users/pengyuchen/Documents/Work/NETs_dataset'
for i in range(100):
    in_dir = os.path.join(in_folder,str(i),'out')
    files = os.listdir(in_dir)
    if files.count("c.rf")==1:
        for file in files:
            if file.endswith('.rf'):
                in_filename = os.path.join(in_dir,file)
                name, ext = os.path.splitext(file)
                name = 'symmetry_SD_' + str(i) +  name + '.txt'
                out_filename = os.path.join(out_folder,name)
                extract_monomer_density(in_filename,out_filename)