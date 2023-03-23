# Open the input and output files

for i in range(64):
    # Construct the input and output file paths
    input_file_path = f'/Users/pongwu/Documents/Work/UMN/2023/GANs_NETs/output_images_translated/5_290/{i}.rf'
    output_file_path = f'/Users/pongwu/Documents/Work/UMN/2023/GANs_NETs/output_images_translated/5_290/{i}_n.rf'

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        # Add the header to the output file
        header = 'format   1   0\ndim\n          3\ncrystal_system\n              cubic\nN_cell_param\n              1\ncell_param    \n      3.0e+00\ngroup_name\n          P_1\nN_monomer\n          2\nngrid\n                   32        32        32\n'
        output_file.write(header)

        # Process the input file line by line
        for line in input_file:
            # Remove any leading or trailing whitespace
            line = line.strip()

            # If the line is not empty
            if line:
                # Split the line into a list of floats
                data = [float(x) for x in line.split()]

                # Calculate the second column as 1 minus the first column
                second_column = 1 - data[0]

                # Write the data to the output file in two columns
                output_file.write(f'{data[0]:.6f} {second_column:.6f}\n')
