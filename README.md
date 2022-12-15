CS 7180
Brendan Martin, Phil Butler
Final Project


Operating System: MacOS Monterey, Version 12.3
Python version: 3.9.12
Packages: Numpy, Matplotlib, Pillow, PyTorch


Datasets:
We employed two versions of the CUB 2011 bird dataset. The first is from the official website:
http://www.vision.caltech.edu/datasets/cub_200_2011/
This includes the images as well as the bird part coordinate annotations.

The second version of the dataset is from:
https://github.com/mrlibw/ManiGAN
The "preprocessed metadata" link contains the textual image captions that were originally collected for the Reed paper.

These datasets should exist in the same folder as the code.


Instructions for running program:

The text embedder/bird classifer program can be run in the terminal with:
    python3 bow_training.py
The resulting weights are stored. Accuracy and precision/recall are output by:
    python3 bow_testing.py

The GAN can be trained by running:
    python3 gan_train.py
The resulting weights and a loss plot are stored. Accuracy and a plot of some example generated images are output by:
    python3 gan_test.py
    
    
