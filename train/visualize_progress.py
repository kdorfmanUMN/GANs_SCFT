from train.isosurface_visualizer import IsosurfaceVisualizer

# Initialize visualizer
visualizer = IsosurfaceVisualizer(isosurface_value=0.2)

# Specify directory containing .pt files and the output directory for visualizations
dir_path = '/Users/pengyuchen/Documents/GAN/past_data/testvisual'
output_dir = '/Users/pengyuchen/Documents/GAN/past_data/testvisual/out'

visualizer.visualize_directory(dir_path, output_dir)