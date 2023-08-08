import os
import shutil

def process_bf(infile_path,outfile_path):
    with open(infile_path, 'r') as f:
        lines = f.readlines()
    lines[4] = '              triclinic\n'
    lines[6] = '              6\n'
    cell_param_line = lines[8].strip().split()
    cell_param_line += ['0.000', '0.000', '1.5707963']
    lines[8] = '      ' + '    '.join(cell_param_line) + '\n'

    lines[14] = '             17000\n'

    with open(outfile_path, 'w') as f:
        f.writelines(lines)



home_dir = '../../TrainingOutputs/0331_Small/45_15/'
# Step 1: Find all folders that contain c.rf
folders_with_crf = []
for root, dirs, files in os.walk(os.path.join(home_dir,"random_scft"), topdown=True):
    if "c.rf" in files:
        folder = os.path.dirname(os.path.join(root))
        num = os.path.split(folder)[-1]
        folders_with_crf.append(num)

for num in folders_with_crf:
    path = os.path.join(home_dir,"triclinic_scft",num)
    os.makedirs(path)
    os.makedirs(os.path.join(path,'out'))

    in_wbf = os.path.join(home_dir,'random_scft',num,'out','w.bf')
    out_wbf = os.path.join(path,'w.bf')
    process_bf(in_wbf,out_wbf)

    shutil.copyfile(os.path.join(home_dir,'triclinic_scft','sample','param'),os.path.join(home_dir,'triclinic_scft',num,'param'))
    shutil.copyfile(os.path.join(home_dir, 'triclinic_scft', 'sample', 'command'),
                    os.path.join(home_dir, 'triclinic_scft', num, 'command'))

