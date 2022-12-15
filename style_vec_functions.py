#
# Brendan Martin
# bow_testing.py
# Fall 2022
# CS 7180


# imports
import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.functional as TF
from PIL import Image

import text_encode_functions as tef


# Create a database containing the (width,height) of every image
# Save in file image_sizes.txt
def gen_image_sizes():
    # Load the mapping between numerical image ids and string image names
    image_dict = pd.read_csv("CUB_200_2011/CUB_200_2011/images.txt", sep = " ", header=None, names=["image_id", "image_name"])
    
    im_sizes = np.zeros( (len(image_dict), 2) )
    
    # For every image
    for i in range(image_dict["image_id"].size):
        # Open the image
        im = Image.open("CUB_200_2011/CUB_200_2011/images/" + image_dict["image_name"][i])
        # Record the width and height
        im_sizes[i,:] = im.size
        
    np.savetxt("image_sizes.txt", im_sizes, delimiter=",")
    
    return


# Create a style vector for each of the images given in image_ind
def gen_style_vec( image_ind ):
    
    # Load the dataset of part locations
    part_locs = pd.read_csv("CUB_200_2011/CUB_200_2011/parts/part_locs.txt", sep = " ", header=None, names=["image_id", "part_id", "x", "y", "visible"])
    
    # load the database of image sizes
    image_sizes = np.loadtxt("image_sizes.txt", delimiter=",")
    
    style_vecs = np.zeros((len(image_ind), 15*3)) #format: (images, part-locations)

    for i, ind in enumerate(image_ind):
        # Extract all part locations associated with image number "ind"
        part_vals = ( part_locs[ part_locs["image_id"] == ind ] )[["x", "y", "visible"]].to_numpy()
        
        # Normalize coordinates to range [0,1]
        part_vals[:,:2] = part_vals[:,:2] / image_sizes[ind-1]
        
        # Flatten the part locations into a vector
        style_vecs[i] = part_vals.flatten()


    return style_vecs


# This will generate text embeddings + style vectors for 20 random images and print the result
def main(argv):

#    master_word_list = tef.gen_word_dict()
    
    image_dir_path = "CUB_200_2011/CUB_200_2011/images"
    class_names = os.listdir(image_dir_path)
    class_names.remove(".DS_Store")
    
    # Get information on image classes, train-test split
    image_mapping = pd.read_csv("CUB_200_2011/CUB_200_2011/images.txt", sep = " ", header=None, names=["Index","Image"])
    image_classes = pd.read_csv("CUB_200_2011/CUB_200_2011/image_class_labels.txt", sep = " ", header=None, names=["Index","Class"])
    data_split = pd.read_csv("CUB_200_2011/CUB_200_2011/train_test_split.txt", sep = " ", header=None, names=["Index","Subset"])
    
    
    # Select a set of random image name
    num_images = 20
    image_ind = np.random.randint(image_mapping.shape[0], size=num_images)
    image_names = [ image_mapping.iloc[image_ind[i],:]["Image"][:-4] for i in range(num_images) ]
    
    
    # Generate a csv file containing the size of every image
    gen_image_sizes()
    
    
    # Generate a numerical embedding for the images' captions
    # Embedding is 200 features long
    embedding = svf.gen_style_vec( image_ind, image_names )
    print("Embedding shape:")
    print(embedding.shape)
    print("Embedding:")
    print(embedding)

    return

if __name__ == "__main__":
    main(sys.argv)
