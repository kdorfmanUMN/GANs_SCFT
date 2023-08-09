import os
import random
import shutil
import argparse

def read_param(filename):
    with open(filename, 'r') as file:
        return file.readlines()


def edit_param(lines, fA, chiN, epsilon):
    lines[3] = "    {:<20} {:9.3f}\n".format(lines[3].split()[0], epsilon)
    lines[9] = "      {:<15} {:<15}{:9.3f}\n".format(lines[9].split()[0], lines[9].split()[1], fA)
    lines[10] = "                       {:<15}{:9.3f}\n".format(lines[10].split()[0], (1 - fA))
    lines[17] = "                       {:<15} {:<15}{:9.1f}\n".format(lines[17].split()[0], lines[17].split()[1], chiN)
    return lines


def write_param(path, lines):
    with open(path, 'w') as file:
        file.writelines(lines)

def main(args):
    for i in range(args.n):
        # Create random fA, chiN, epsilon
        fA = random.uniform(args.fA_min, args.fA_max)
        chiN = random.uniform(args.chiN_min, args.chiN_max)
        epsilon = random.uniform(args.epsilon_min, args.epsilon_max)
        if random.randint(0, 1):
            epsilon = 1 / epsilon

        # Create directories
        path = os.path.join(args.output_dir, args.group, str(i))
        os.makedirs(path, exist_ok=True)
        path = os.path.join(args.output_dir, args.group, str(i), 'c')
        os.makedirs(path, exist_ok=True)

        # Copy command files
        path = os.path.join(args.output_dir, args.group, str(i), 'command')
        shutil.copyfile(args.command_file, path)

        # Edit param files
        lines = read_param(args.param_file)
        lines = edit_param(lines, fA, chiN, epsilon)
        path = os.path.join(args.output_dir, args.group, str(i), 'param')
        write_param(path, lines)

        # Randomly pick an initial guess 'rgrid' file and move to subdirs
        path_in = os.path.join(args.in_dir, random.choice(args.rgrid_files))
        path_out = os.path.join(args.output_dir, args.group, str(i), 'rgrid')
        shutil.copyfile(path_in, path_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate parameters and directories for SCFT calculations.")

    # Arguments
    parser.add_argument('--in_dir', nargs='+', required=True, help="List of rgrid files.")
    parser.add_argument('--rgrid_files', nargs='+', default=['DG_fA35', 'DG_fA40', 'DG_fA45'],
                        help="List of rgrid files.")
    parser.add_argument('--param_file', default='param', help="Path to example param file.")
    parser.add_argument('--command_file', default='command', help="Path to example command file.")
    parser.add_argument('--fA_min', type=float, default=0.35, help="Minimum value for fA.")
    parser.add_argument('--fA_max', type=float, default=0.45, help="Maximum value for fA.")
    parser.add_argument('--chiN_min', type=float, default=15.0, help="Minimum value for chiN.")
    parser.add_argument('--chiN_max', type=float, default=20.0, help="Maximum value for chiN.")
    parser.add_argument('--epsilon_min', type=float, default=0.75, help="Minimum value for epsilon.")
    parser.add_argument('--epsilon_max', type=float, default=1.0, help="Maximum value for epsilon.")
    parser.add_argument('--n', type=int, default=150, help="Number of calculations.")
    parser.add_argument('--group', default='P_1', help="Group name.")
    parser.add_argument('--output_dir', default=os.getcwd(), help="Output directory.")

    opt = parser.parse_args()

    main(opt)

