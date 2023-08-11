import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
MANUAL_SEED = 999

class NetsDataset(Dataset):
    def __init__(self, nets_dir):
        self.nets_dir = nets_dir
        self.nets_list = [file for file in os.listdir(nets_dir) if file.endswith('.txt')]

    def __len__(self):
        return len(self.nets_list)

    def __getitem__(self, idx):
        nets_path = os.path.join(self.nets_dir, self.nets_list[idx])
        nets = np.loadtxt(nets_path).astype('float32')
        assert len(nets) == 32 * 32 * 32, "nets shape not on 32 * 32 * 32"
        return torch.from_numpy(nets).reshape((1, 32, 32, 32))

class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=32, nc=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose3d(self.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf=64, nc=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)


#  weights initialization on netG and netD
def weights_init(m):
    # Initialize weights for Conv and BatchNorm layers
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def initialize_models(args, device):
    netG = Generator(args.ngpu, args.nz, args.ngf, args.nc).to(device)
    if device.type == 'cuda' and args.ngpu > 1:
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(args.ngpu, args.ndf, args.nc).to(device)
    if device.type == 'cuda' and args.ngpu > 1:
        netD = nn.DataParallel(netD, list(range(args.ngpu)))
    netD.apply(weights_init)
    print(netD)

    return netG, netD


def save_losses(loss_list, filename):
    with open(filename, 'a') as f:
        for loss in loss_list:
            f.write(f"{loss}\n")


def train_GAN(args, netG, netD, criterion, dataloader, device):
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    fixed_noise = torch.randn(64, args.nz, 1, 1, 1, device=device)

    G_losses, D_losses = [], []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(args.nepochs):
        for i, data in enumerate(dataloader):

            # ---------- Update discriminator (D) ----------

            netD.zero_grad()
            # train D on real data and compute loss
            real_data = data.to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), np.random.uniform(0.8, 1.), dtype=torch.float,
                               device=device)  # one-sided label smoothing for real samples: 1 -> [0.8, 1]
            output = netD(real_data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train D on fake data and compute loss
            noise = torch.randn(b_size, args.nz, 1, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            # Update the D weights
            optimizerD.step()

            # ---------- Update generator (G) ----------

            netG.zero_grad()

            # Train G to fool D and compute loss
            label.fill_(1)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update the G weights
            optimizerG.step()

            # ---------- Monitoring and Saving ----------
            if iters % 25 == 0 or epoch == args.nepochs - 1:  # Print statistics every 25 iters
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.nepochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                # Save losses
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Save models and generated samples periodically and at the end of training
            if (iters % args.save_iters == 0) or ((epoch == args.nepochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    torch.save(fake, os.path.join(args.out_dir_images, str(epoch) + '_' + str(i) + '.pt'))
                    torch.save(netG.state_dict(),
                               os.path.join(args.out_dir_model, 'Gweights_' + str(epoch) + '_' + str(i) + '.pt'))
                    torch.save(netD.state_dict(),
                               os.path.join(args.out_dir_model, 'Dweights_' + str(epoch) + '_' + str(i) + '.pt'))
                    torch.save(optimizerD.state_dict(),
                               os.path.join(args.out_dir_model, 'AdamD_' + str(epoch) + '_' + str(i) + '.pt'))
                    torch.save(optimizerG.state_dict(),
                               os.path.join(args.out_dir_model, 'AdamG_' + str(epoch) + '_' + str(i) + '.pt'))
            iters += 1

    print('training finished')
    # Save losses to file at the end
    save_losses(G_losses, os.path.join(args.out_dir_model, 'G_loss.txt'))
    save_losses(D_losses, os.path.join(args.out_dir_model, 'D_loss.txt'))
    return


def main(args):
    print("Random Seed: ", MANUAL_SEED)
    random.seed(MANUAL_SEED)
    torch.manual_seed(MANUAL_SEED)

    dataset = NetsDataset(nets_dir=args.dataroot)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.workers)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.ngpu > 0 else "cpu")

    netG, netD = initialize_models(args, device)
    criterion = nn.BCELoss()

    train_GAN(args, netG, netD, criterion, dataloader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',required=True, help='input data folder')
    parser.add_argument('--out_dir_images', default='.', help='output dir for generated 3D images')
    parser.add_argument('--out_dir_model', default='.', help='output dir for model')
    parser.add_argument('--workers', type=int, default=1, help='number of workers')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size during training')
    parser.add_argument('--nc', type=int, default=1, help='number of channels')
    parser.add_argument('--nz', type=int, default=100, help='size of z latent vector')
    parser.add_argument('--ngf', type=int, default=64, help='size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=64, help='size of feature maps in discriminator')
    parser.add_argument('--nepochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for optimisers')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 hyperparameter for Adam optimizer')
    parser.add_argument('--save_iters', type=int, default=150, help='step for saving paths and generated images')
    args = parser.parse_args()

    main(args)
