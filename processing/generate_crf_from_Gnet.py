import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset

# Set random seed for reproducibility
manualSeed = 500
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

class NetsDataset(Dataset):
    def __init__(self, nets_dir, img_dir, transform=None, target_transform=None):
        self.nets_dir = nets_dir
        nets_list = os.listdir(self.nets_dir)
        self.nets_list = []
        for file in nets_list:
            if file.endswith('.txt'):
                self.nets_list.append(file)
    def __len__(self):
        return len(self.nets_list)
    def __getitem__(self, idx):
        nets_path = os.path.join(self.nets_dir, self.nets_list[idx])
        nets = np.loadtxt(nets_path).astype('float32')
        assert len(nets) == 32 * 32 * 32, "nets shape not on 32 * 32 * 32"
        nets_pt = torch.from_numpy(nets).reshape((1,32, 32, 32))
        return nets_pt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class CircularPadCTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True):
        super(CircularPadCTranspose3d, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding=0, bias=bias)
        self.kernel_size = kernel_size
        self.padding = padding
        nn.init.normal_(self.conv_transpose.weight, 0.0, 0.02)

    def forward(self, x):
        pad_dims = ((self.padding+1)//2, self.padding//2,)*3
        x = F.pad(x, pad_dims, mode='circular')
        x = self.conv_transpose(x)
        return x

class Generator(nn.Module):
    def __init__(self, ngpu, latent_dim = 100):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            CircularPadCTranspose3d(self.latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Root directory for dataset
    # dataroot = "~/Downloads/celeba"

    # Number of workers for dataloader
    workers = 1

    # Batch size during training
    batch_size = 1

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 1

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Number of training epochs
    num_epochs = 1

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 2

    # G_weight_path = '/Users/pengyuchen/Documents/Work/TrainingOutputs/0325A/Gweights/Gweights_47_13.pt'
    # out_dir = '/Users/pengyuchen/Documents/Work/TrainingOutputs/0325A/47_13/'
    G_weight_path = '../../TrainingOutputs/0325A/Gweights_47_13.pt'
    out_dir = '../../TrainingOutputs/0325A/47_13/'

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    print(netG)
    netG.load_state_dict(torch.load(G_weight_path,map_location=device))
    # Print the model
    for i in range(1000):
        noise = torch.randn(1, nz, 1, 1, 1, device=device)
        fake = netG(noise)
        fake_cpu = fake.numpy(force = True)
        A = fake_cpu.reshape(-1,1)
        B = 1-A
        AB = np.hstack((A, B))

        with open(os.path.join(out_dir,str(i)+'.rf'), 'w') as output_file:
            # Add the header to the output file
            header = 'format   1   0\ndim\n          3\ncrystal_system\n              orthorhombic\nN_cell_param\n              3\ncell_param    \n      3.0e+00   3.0e+00  3.0e+00 \ngroup_name\n          P_1\nN_monomer\n          2\nngrid\n                   32        32        32\n'
            output_file.write(header)
        with open(os.path.join(out_dir,str(i)+'.rf'), 'a') as output_file:
            # Write the data to the output file
            np.savetxt(output_file, AB, delimiter='\t', fmt='%.16f')


