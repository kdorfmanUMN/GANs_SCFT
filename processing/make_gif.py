import imageio.v2 as imageio
import os

# dir_path = '/Users/pengyuchen/Documents/Work/TrainingOutputs/0331_Small/isosurface_plots'

# sort the file names
#file_names = sorted([fn for fn in os.listdir(dir_path) if fn.endswith('.png')], key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))
#
dir_path = '/Users/pengyuchen/Documents/GAN/GAN_manuscript/Figure1/DG/image'
# #
# # # sort the file names
file_names = sorted([fn for fn in os.listdir(dir_path) if fn.endswith('.png')], key=lambda x: (int(x.split('A.')[0])))


# create a list of images in the order you want
images = []
for i in range(len(file_names)):
    images.append(imageio.imread(os.path.join(dir_path,file_names[i])))

# create the gif
imageio.mimsave(os.path.join(dir_path,'animation.gif'), images, duration = 0.25)

