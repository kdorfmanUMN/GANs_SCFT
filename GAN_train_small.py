import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
import argparse

class NetsDataset(Dataset):
    def __init__(self, nets_dir):
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
        max_value = np.max(nets)
        min_value = np.min(nets)
        if max_value >= 1 or min_value <= 0:
            nets = (nets - min_value) / (max_value - min_value)
        assert len(nets) == 32 * 32 * 32, "nets shape not on 32 * 32 * 32"
        nets_pt = torch.from_numpy(nets).reshape((1, 32, 32, 32))
        return nets_pt

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu, nz=100):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose3d(self.nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf*2, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/Users/pengyuchen/Documents/GAN/past_data/testdata', help='input dataset file')
    parser.add_argument('--out_dir_images', default='', help='output file for generated images')
    parser.add_argument('--out_dir_model', default='', help='output file for model')
    parser.add_argument('--workers', type=int, default=1, help='number of workers')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--bsize', default=128, help='batch size during training')
    parser.add_argument('--nc', default=1, help='number of channels')
    parser.add_argument('--nz', default=100, help='size of z latent vector')
    parser.add_argument('--ngf', default=32, help='size of feature maps in generator')
    parser.add_argument('--ndf', default=64, help='size of feature maps in discriminator')
    parser.add_argument('--nepochs', default=60, help='number of training epochs')
    parser.add_argument('--lr', default=0.0002, help='learning rate for optimisers')
    parser.add_argument('--beta1', default=0.5, help='beta1 hyperparameter for Adam optimiser')
    parser.add_argument('--save_iters', default=150, help='step for saving paths and generated images')
    arg = parser.parse_args()

    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    workers = int(arg.workers)
    batch_size = int(arg.bsize)
    ngpu = int(arg.ngpu)
    num_epochs = int(arg.nepochs)
    nz = int(arg.nz)
    nc = int(arg.nc)
    ngf = int(arg.ngf)
    ndf = int(arg.ndf)

    dataset = NetsDataset(nets_dir=arg.dataroot)
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
    netG.apply(weights_init)
    # Print the model
    print(netG)
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to check the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=arg.lr, betas=(arg.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=arg.lr, betas=(arg.beta1, 0.999))

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
            # one-sided label smoothing for real samples: 1 -> [0.8, 1]
            label = torch.full((b_size,), np.random.uniform(0.8,1.), dtype=torch.float, device=device)
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

            # Print statistics every 25 iters
            if iters % 25 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            # Save losses
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Save models and generated samples
            if (iters % int(arg.save_iters) ==0) or ((epoch==num_epochs-1)and(i==len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    torch.save(fake,os.path.join(arg.out_dir_images,str(epoch)+'_'+str(i)+'.pt'))
                    # torch.save(netG.state_dict(),os.path.join(out_dir_model,'Gweights_'+str(epoch)+'_'+str(i)+'.pt'))
                    # torch.save(netD.state_dict(),os.path.join(out_dir_model,'Dweights_'+str(epoch)+'_'+str(i)+'.pt'))
                    # torch.save(optimizerD.state_dict(),os.path.join(out_dir_model,'AdamD_'+str(epoch)+'_'+str(i)+'.pt'))
                    # torch.save(optimizerG.state_dict(),os.path.join(out_dir_model, 'AdamG_' + str(epoch) + '_' + str(i) + '.pt'))
            iters +=1
    with open(os.path.join(arg.out_dir_model,'G_loss.txt'),'w') as f:
        for loss in G_losses:
            f.write(str(loss)+'\n')
    with open(os.path.join(arg.out_dir_model,'D_loss.txt'),'w') as f:
        for loss in D_losses:
            f.write(str(loss)+'\n')
    print('training finished')
