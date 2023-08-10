from train.isosurface_visualizer import IsosurfaceVisualizer

# Initialize visualizer
visualizer = IsosurfaceVisualizer(isosurface_value=0.5)

# Specify directory containing .pt files and the output directory for visualizations
dir_path = './output/'
output_dir = './output/isosurface_plot/'

visualizer.visualize_directory(dir_path, output_dir)