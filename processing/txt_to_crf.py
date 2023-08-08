import numpy as np
import os, shutil

def txt_to_crf(inputname,outputname):
    data = np.loadtxt(inputname)
    columnA = (data - np.min(data))/(np.max(data)-np.min(data))

    with open(outputname, 'w') as output_file:
            # Add the header to the output file
        header = 'format   1   0\ndim\n          3\ncrystal_system\n              orthorhombic\nN_cell_param\n              3\ncell_param    \n      3.5e+00   3.5e+00  3.5e+00 \ngroup_name\n          P_1\nN_monomer\n          2\nmesh\n                   32        32        32\n'
        output_file.write(header)

    with open(outputname, 'a') as output_file:
            # Write the data to the output file
        columnA = columnA.reshape(-1, 1)
        columnB = 1 - columnA
        AB = np.hstack((columnA, columnB))
        np.savetxt(output_file, AB, delimiter='\t', fmt='%.16f')
    return



#
# # dir_path ='/Users/pengyuchen/Documents/Work/TrainingData/NETs_rotated_subset_crf/SCFT_examine'
# dir_path = '../../TrainingData/NETs_Transformed_SCFT/'
# count = 0
#
# for filename in os.listdir(os.path.dirname(dir_path)):
#     print(filename)
#     if filename.endswith('.txt') ==1:
#         path = os.path.join(dir_path,str(count))
#         os.makedirs(path,exist_ok=True)
#         os.makedirs(os.path.join(path,'out'), exist_ok=True)
#         # os.makedirs(os.path.join(path, 'c'), exist_ok=True)
#         # os.makedirs(os.path.join(path, 'w'), exist_ok=True)
#         shutil.copyfile(os.path.join(dir_path,'sample',"command"),os.path.join(path,"command"))
#         shutil.copyfile(os.path.join(dir_path, 'sample',"param"), os.path.join(path, "param"))
#         shutil.copyfile(os.path.join(os.path.dirname(dir_path), os.path.basename(filename)), os.path.join(path, "rgrid.rf"))
#         count +=1

input_dir = '../../TrainingOutputs/0330_Large/40_100/rgrid/'
output_dir = '../../TrainingOutputs/0330_Large/40_100/random_scft/'
count = 0
for file_name in os.listdir(input_dir):
    if file_name.endswith('.rf'):
        path = os.path.join(output_dir,str(count))
        os.makedirs(path,exist_ok=True)
        os.makedirs(os.path.join(path, 'out'), exist_ok=True)
        param_files = ["param1","param2","param3","param4","param5","param6","param7","param8", "param9"]
        chosen_param = param_files[count%9]
        shutil.copyfile(os.path.join(output_dir, 'sample', "command"), os.path.join(path, "command"))
        shutil.copyfile(os.path.join(output_dir, 'sample', chosen_param), os.path.join(path, "param"))
        shutil.copyfile(os.path.join(input_dir,file_name),os.path.join(path,'rgrid.rf'))
        count += 1