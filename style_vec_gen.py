#
# Brendan Martin
# style_vec_gen.py
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

import text_encode_functions as tef
import style_vec_functions as svf


# main function
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
#    print(image_ind.shape)
    num_images = 20
    image_ind = np.random.randint(image_mapping.shape[0], size=num_images)
    image_names = [ image_mapping.iloc[image_ind[i],:]["Image"][:-4] for i in range(num_images) ]
    
    
    # Generate a csv file containing the size of every image
    svf.gen_image_sizes()
    
    
    # Generate a numerical embedding for the images' captions
    # Embedding is 200 features long
    embedding = svf.gen_style_vec( image_ind, image_names )
    print(embedding.shape)
    
    print(embedding)
    
    
    


    

    return

if __name__ == "__main__":
    main(sys.argv)
