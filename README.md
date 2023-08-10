# GANs_SCFT

<!-- TABLE OF CONTENTS -->
- [Overview](##overview)
- [Citing this work](##Citing-this-work)
- [Relevant work](##Relevant-work)


## Overview
This repo is part of the published work Generative Block Polymer Phase Discovery (http://doi/), which 
describes a method to train Deep Convoluntional Generative Adversarial Networks (DCGANs) to generate initial guess 
fields for self-consistent field theory (SCFT) simulations for phase discovery.

The DCGANs were trained on a set of 3D density fields of 5 known block polymer phases (single-gyroid, single-diamond, 
single-primitive, double-gyroid, and double diamond) generated from SCFT simulation trajectories. 
A data augmentation strategy including tiling, random translation and rotation of the fields were applied.
Generated fields from the DCGANs were used as initial guess fields to seed new SCFT simulations, which leads to the discovery 
of novel phases.
<br>
<p align="center">
<img src="docs/figs/workflow.png" alt="workflow" width="550"/><br>
<font size="-1"><b>Fig. 1:</b> Workflow of generated polymer field theory.</font>
</p>

### Citing this work
If you use these codes or parts of them, as well as the informtion provided in this repo, please cite the following article:

add citation here

### Relevant work
DCGANs was first introduced in the paper Unsupervised Representation 
Learning with Deep Convolutional Generative Adversarial Networks (https://arxiv.org/abs/1511.06434). 
See also the implementation in PyTorch (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
The SCFT simulations were performed using the open-source software package PSCF (https://github.com/dmorse/pscfpp).
The visualization of polymer fields was performed using the open-source software Polymer Visual (https://github.com/kdorfmanUMN/polymer_visual.) 

## Getting Started

### Prerequisites
  ```sh
  pip install -m requirements.txt
  ```
### Data Downloads

1. Download pre-processed density fields `data` from [here](https://drum). Note: the `data` folder contains 100 example `.pdb` (~1.1 G) files for each structure.
2. Put `point_clouds` and `raw` into the `data` folder.