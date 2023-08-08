# from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np

# from skimage import measure
import os

from torch.utils.data import Dataset

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
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
        # rescale to [0,1] only when exceed bounds
        max_value = np.max(nets)
        min_value = np.min(nets)
        if max_value >= 1 or min_value <= 0:
            nets = (nets - min_value) / (max_value - min_value)

        assert len(nets) == 32 * 32 * 32, "nets shape not on 32 * 32 * 32"
        nets_pt = torch.from_numpy(nets).reshape((1,32, 32, 32))

        # Rescale the input data
        # max_val = torch.max(nets_pt)
        # if max_val > 1:
        #     nets_pt = nets_pt / max_val

        return nets_pt


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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
            nn.ConvTranspose3d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #(batch size, 64, 16, 16, 16).
            nn.Conv3d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch size, 128, 8, 8, 8)
            nn.Conv3d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #(batch size, 256, 4, 4, 4).
            nn.Conv3d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# class DivideTransform:
#     def __init__(self, a):
#         self.a = a
#
#     def __call__(self, sample):
#         return sample / self.a

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Root directory for dataset
    # dataroot = "~/Downloads/celeba"

    # Number of workers for dataloader
    workers = 1

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    # image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 1

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    # ngf = 64

    # Size of feature maps in discriminator
    # ndf = 64

    # Number of training epochs
    num_epochs = 100

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 4

    # Maximum value of the input density data for rescaling
    # scaling = 1

    out_dir = '../TrainingOutputs/0330_Large/'
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    # dataset = NetsDataset(nets_dir='/Users/pengyuchen/Documents/Work/NETs_dataset_transformed', img_dir=None,
    #                       transform=DivideTransform(scaling))
    dataset = NetsDataset(nets_dir='../TrainingData/NETs_Random_Crop_0/', img_dir=None)
    # Create the dataloader transforms = transforms.
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Print the model
    print(netG)
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    #
    # Print the model
    print(netD)



    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # Start training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            # Update discriminator
            netD.zero_grad()
            real_data = data.to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)
            output = netD(real_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update generator
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            # Print statistics
            if iters % 25 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Save models and generated samples
            if (iters% 150 ==0) or ((epoch==num_epochs-1)and(i==len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    torch.save(fake,os.path.join(out_dir,str(epoch)+'_'+str(i)+'.pt'))
                    torch.save(fake,os.path.join(out_dir,str(epoch)+'_'+str(i)+'.pt'))
                    torch.save(netG.state_dict(),os.path.join(out_dir,'weights','Gweights_'+str(epoch)+'_'+str(i)+'.pt'))
                    torch.save(netD.state_dict(),os.path.join(out_dir,'weights','Dweights_'+str(epoch)+'_'+str(i)+'.pt'))
                    torch.save(optimizerD.state_dict(),os.path.join(out_dir,'weights','AdamD_'+str(epoch)+'_'+str(i)+'.pt'))
                    torch.save(optimizerG.state_dict(),
                               os.path.join(out_dir, 'weights', 'AdamG_' + str(epoch) + '_' + str(i) + '.pt'))
            iters +=1
    with open(os.path.join(out_dir,'G_loss.txt'),'w') as f:
        for loss in G_losses:
            f.write(str(loss)+'\n')
    with open(os.path.join(out_dir,'D_loss.txt'),'w') as f:
        for loss in D_losses:
            f.write(str(loss)+'\n')
    print('training finished')
