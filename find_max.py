import os

folder_path = '/Users/pongwu/Documents/Work/UMN/2023/GANs_NETs/NETs_dataset_translated'
max_value = float('-inf')

for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                value = float(line.strip())
                if value > max_value:
                    max_value = value

print("Maximum value:", max_value)