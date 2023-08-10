import random
import torch
import torch.nn as nn
import numpy as np
import os

# Set random seed for reproducibility
MAUNUAL_SEED = 500
print("Random Seed: ", MAUNUAL_SEED)
random.seed(MAUNUAL_SEED)
torch.manual_seed(MAUNUAL_SEED)

class Generator(nn.Module):
    def __init__(self, ngpu=0, latent_dim=100):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose3d(self.latent_dim, 512, 4, 1, 0, bias=False),
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


# Function to save the generated docs as .rf files
def save_as_rf(fake, save_path):
    """
    Convert the generated tensor into the .rf format and save it.
    Parameters:
    - fake: The tensor generated from the GAN.
    - save_path: The path where to save the .rf file.
    """

    # Reshape and format the tensor data
    density_A = fake.reshape(-1, 1)
    density_AB = np.hstack((density_A, 1 - density_A))

    # Define header
    header = ('format   1   0\ndim\n          3\ncrystal_system\n'
              '              orthorhombic\nN_cell_param\n              3\n'
              'cell_param    \n      3.0e+00   3.0e+00  3.0e+00 \n'
              'group_name\n          P_1\nN_monomer\n          2\n'
              'ngrid\n                   32        32        32\n')
    # Write header and the tensor data to the file
    with open(save_path, 'w') as output_file:
        output_file.write(header)
        # Write the tensor data to the file
        np.savetxt(output_file, density_AB, delimiter='\t', fmt='%.16f')


# Function to generate docs
def generate_images(weight_path, out_dir, num_images):
    """
    Generate a set of 3D docs using the trained GAN generator.

    Parameters:
    - weight_path: Path to the pretrained generator weights.
    - out_dir: Output directory to save the generated .rf files.
    - num_images: Number of docs to generate.
    """
    # Ensure the output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a generator instance and load the weights
    generator = Generator().to(device)
    state_dict = torch.load(weight_path, map_location='cpu')
    # Handle cases where the state_dict keys have 'module.' prefix
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    generator.load_state_dict(new_state_dict)
    generator.eval()

    # Generate the specified number of docs
    with torch.no_grad():
        for i in range(num_images):
            # Assuming the generator takes random noise as input
            noise = torch.randn(1, 100, 1, 1, 1).to(device)

            # Generate a fake image
            fake_image = generator(noise)

            # Convert tensor to numpy array and save as .rf
            fake_cpu = fake_image.squeeze(0).cpu().numpy()
            save_path = os.path.join(out_dir, f'guess_{i + 1}.rf')
            save_as_rf(fake_cpu, save_path)

def main(args):
    weight_path = args.weight_path
    out_dir = args.out_dir
    num_images = args.num_images
    generate_images(weight_path, out_dir, num_images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 3D density fields using the trained GAN generator')
    parser.add_argument('--weight_path', type=str, help='Path to the pretrained generator weights')
    parser.add_argument('--out_dir', type=str, help='Output directory to save the generated .rf files')
    parser.add_argument('--num_images', type=int, default=5000, help='Number of density fields to generate')
    args = parser.parse_args()
    main(args)

