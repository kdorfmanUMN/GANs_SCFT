# GANs_SCFT (Generative Block Polymer Phase Discovery)

<!-- TABLE OF CONTENTS -->
## Table of Contents
- [Overview](#overview)
  - [Citing this work](#citing-this-work)
  - [Relevant work](#relevant-work)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Data Downloads](#data-downloads)
- [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Field Generation using the Trained Generator](#field-generation-using-the-trained-generator)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Overview
This repo is part of the submitted work [Generative Block Polymer Phase Discovery](http://doi/), which describes a method 
to train Deep Convolutional Generative Adversarial Networks (DCGANs) to generate initial guess fields for Self-Consistent
Field Theory (SCFT) simulations for phase discovery.

#### Key Highlights:
- **Data Source:** 3D density fields of five established block polymer phases from SCFT trajectories.
- **Data Augmentation:** Tiling, random translation, and rotation.
- **Purpose of DCGANs:** Fields generated from DCGANs seeded new SCFT simulations, enabling the discovery of novel phases.

<p align="center">
<img src="docs/figs/workflow.png" alt="workflow" width="800"/><br>
<font size="-1"><b>Fig. 1:</b> Overview of the polymer field theory generation process.</font>
</p>

### Citing this work
For referencing or leveraging parts of this work, please refer to:

add citation here

### Relevant work
- **GANs:** [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).
- **DCGANs:** [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). 
- **DCGAN Tutorials:** [PyTorch implementation](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) and [TensorFlow implementation](https://www.tensorflow.org/tutorials/generative/dcgan).
- **SCFT:** [Broadly Accessible Self-Consistent Field Theory for Block Polymer Materials Discovery](https://pubs.acs.org/doi/full/10.1021/acs.macromol.6b00107).
- **Software Tools:**
    - **SCFT Simulations:** [PSCF (C++/CUDA)](https://github.com/dmorse/pscfpp).
    - **Visualization of Density Fields:** [Polymer Visual](https://github.com/kdorfmanUMN/polymer_visual.).
## Getting Started

### Prerequisites
  ```sh
  pip install -r requirements.txt
  ```
### Data Downloads

Acquire both raw and processed density fields [here](https://link_to_drum_repo). 
The `data` folder consists of 18,273 3D density field data in `.txt` format, which can be used directly for training.
For training, every data point is viewed as a grayscale 3D image (1 x 32 x 32 x 32).

## Usage

### Data Preparation
- **Data Generation:** <br>
The raw density data are extracted from SCFT trajectories that converge to five known network phases: single-gyroid, single-diamond, 
singe-primitive, double-gyroid, and double-primitive. The inputs and outputs of the SCFT simulations are provided in [Data Repository for U of M (DRUM)](link).



- **Data Augmentation:** <br>

  - **Description:**  
    Each `.rf` file, representing a density field obtained from the SCFT calculation, undergoes a series of data augmentation techniques: tiling, random translation, and rotation. Fields showcasing unphysical density values, specifically where `phi_A > 1` or `phi_A < 0`, are excluded.

  - **Command Line Execution:**  
    Navigate to the preprocessing directory and execute the data processing script:
     ```sh
       cd ./preprocessing
       python data_processor.py --in_filename /path/to/input --out_filename /path/to/output --grid 32 32 32
     ```
     `--in_filename`: Path to the input density field file (`.rf`).<br>
     `--out_filename`: Path to the output 3D image file (`.txt`).<br>
     `--grid`: (Optional) A tuple specifying the output dimensions. The default is (32, 32, 32).<br>

  - **Python Method Execution:**  
    Instead of command-line execution, you can utilize it directly within a Python script:
      ```py
        processor = DataProcessor()
        processor.process_files(<input_filename>, <output_filename>)
      ```


### Training
- **Training the DCGANs with Preprocessed Data:** <br>

  Run the training script using the following commands:
   ```sh
     cd ./train
     python GAN_train.py --dataroot /path/to/data --out_dir_images /path/to/output/image/dir --out_dir_model /path/to/output/model/dir
   ```

    `--dataroot`: directory that contains processed training data. <br>
    `--out_dir_images`: directory to save generated tensor (`.pt`) during the training process.<br>
    `--out_dir_model`: directory to save model parameters (`.pt`) during the training process.<br>
    Optional arguments, such as batch size, can be found in the codes.<br>
    <br>
- **Visualizing the Training Progression:** <br>
The output images from a set of fixed noise can be visualized to track the training progression using `isosurface_visualizer'. 
An example is provided in './train/visualize_progress.py'.
  ```py
    visualizer = IsosurfaceVisualizer(isosurface_value=0.5)
    visualizer.visualize_directory(<input_directory>, <output_directory>>)
  ```
  `<input_directory>`: directory to the saved generated tensor (`.pt`).<br>
`<output_directory>`: directory to save isosurface plots (`.png`).<br>

<p align="center">
<img src="docs/figs/GANs_training.gif" alt="Animated GIF" width="800"><br>
<font size="-1"><b>Fig. 2:</b> Progression of generated density fields during the training process.</font>
</p>

### Field Generation using the Trained Generator
Generate density fields by feeding random latent vectors to the generator.
  ```sh
    cd ./postprocessing
    python generate_guess.py --weight_path ../model/Gweights_45_15.pt --out_dir /path/to/output/dir --num_images 5000
 ```
`--weight_path`: path to the pretrained generator weights. <br>
`--out_dir`: directory to save generated guesses.<br>
`--num_images`: number of density initial guesses to generate.<br>

### SCFT backends

Generated density fields are used as initial guesses for SCFT simulations at a fixed state point. 
All the PSCF inputs and outputs are provided in [DRUM](link).

## Results

- Convergence: 545/5000 <br>
- Candidate network phases: 349 <br>
- De novo generation of all known network phases <br>
- Discovery of novel network phases <br>


<p align="center">
<img src="docs/figs/histogram.png" alt="histogram" width="800"><br>
<font size="-1"><b>Fig. 3:</b>  Free energy histogram of generated candidate network phases and represented network phases.
</p>

## Contributing
For contributing or submitting pull requests, please contact the author:
- Pengyu Chen [chen6580@umn.edu](mailto:chen6580@umn.edu)

## Acknowledgments

The authors would like to thank Qingyuan Jiang, Benjamin R. Magruder, Dr. Guo Kang Cheong, Prof. Chris J. Bartel, 
and Prof. Frank S. Bates for their valuable inputs.

This work was supported primarily by the National Science Foundation through the University of Minnesota MRSEC under Award Number DMR-2011401. 
Computational resources provided in part by the Minnesota Supercomputing Institute.


