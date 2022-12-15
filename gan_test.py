#
# Phil Butler, Brendan Martin
# gan_test.py
# Fall 2022
# CS 7180
# Resources:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# https://github.com/soumith/dcgan.torch/blob/master/main.lua
# https://github.com/aelnouby/Text-to-Image-Synthesis/blob/master/models/gan.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from text_encode_functions import bird_classify

import text_encode_functions as tef
import gan_functions as gf


# this function will load the saved GAN and generate nine images with the generator
# a plot of the images is saved
def main():
    master_word_list = tef.gen_word_dict()
    # Get information on image classes, train-test split
    image_mapping = pd.read_csv("CUB_200_2011/CUB_200_2011/images.txt", sep = " ", header=None, names=["Index","Image"])
    image_classes = pd.read_csv("CUB_200_2011/CUB_200_2011/image_class_labels.txt", sep = " ", header=None, names=["Index","Class"])
    data_split = pd.read_csv("CUB_200_2011/CUB_200_2011/train_test_split.txt", sep = " ", header=None, names=["Index","Subset"])
    
    

    # Load text_embedder
    PATH = 'bow_weights.pth'
    text_embedder = bird_classify(500)
    text_embedder.load_state_dict(torch.load(PATH))  # it takes the loaded dictionary, not the path file itself
    text_embedder.eval()

    # Put a sample text into an embedding
    text_embedding = text_embedder(torch.rand(500))
    text_embedding = torch.reshape(text_embedding, (1, 200, 1, 1))
    
    
    # Choose a set of image indices which we wish to generate
    num_samples = 9
    chosen_classes = np.random.randint(200, size=9)
    chosen_images = np.zeros((num_samples,), dtype="int")
    for i in range(num_samples):
        chosen_images[i] = gf.sample_class(image_mapping, image_classes, data_split, chosen_classes[i], "test")
    
    

    # Load the GAN weights
    gen = gf.Generator(245)
#    discrim = gf.Discriminator(3)
    
    filename = "gen_images_pre.jpg"
    data = gf.gen_embed( gen, text_embedder, chosen_images, image_mapping, master_word_list )
    gf.graph_gen_images( data, filename )
    
    gen.load_state_dict(torch.load("weights/generator_weights.pth"))
#    discrim.load_state_dict(torch.load("weights/discriminator.pth"))
    
    
    
    
    
    print(chosen_images)
    
    filename = "gen_images.jpg"
    data = gf.gen_embed( gen, text_embedder, chosen_images, image_mapping, master_word_list )
    gf.graph_gen_images( data, filename )


if __name__ == '__main__':
    main()
