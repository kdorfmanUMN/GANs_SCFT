import os, shutil

# dir_path ='/Users/pengyuchen/Documents/Work/TrainingData/NETs_rotated_subset_crf/SCFT_examine'
dir_path = '../../TrainingData/NETs_Transformed_SCFT/'
count = 0

for filename in os.listdir(os.path.dirname(dir_path)):
    print(filename)
    if filename.endswith('.txt') ==1:
        count +=1
        path = os.path.join(dir_path,str(count))
        os.makedirs(path,exist_ok=True)
        os.makedirs(os.path.join(path,'out'), exist_ok=True)
        # os.makedirs(os.path.join(path, 'c'), exist_ok=True)
        # os.makedirs(os.path.join(path, 'w'), exist_ok=True)
        shutil.copyfile(os.path.join(dir_path,'sample',"command"),os.path.join(path,"command"))
        shutil.copyfile(os.path.join(dir_path, 'sample',"param"), os.path.join(path, "param"))
        shutil.copyfile(os.path.join(os.path.dirname(dir_path), os.path.basename(filename)), os.path.join(path, "rgrid.rf"))

