class Generator(nn.Module):
    def __init__(self, ngpu, latent_dim=100):
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
        x = self.main(input)
        x = (x + torch.flip(x, dims=(2, 3, 4))) / 2.0
        x = (x + torch.flip(x, dims=(1, 3, 4))) / 2.0
        x = (x + torch.flip(x, dims=(1, 2, 4))) / 2.0
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
            nn.sigmoid() # use tanh instead of sigmoid
        )

        # add centrosymmetric filter
        centrosym_filter = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float)
        self.centrosym_filter = nn.Parameter(centrosym_filter.view(1, 1, 3, 3, 3), requires_grad=False)

    def forward(self, input):
        x = self.main(input)
        x = F.conv3d(x, self.centrosym_filter, padding=1)
        x = F.pad(x, ((1, 1), (1, 1), (1, 1)), mode='circular') # add circular padding
        return x
