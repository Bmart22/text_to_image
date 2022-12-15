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

import text_encode_functions as tef

# this function will load the saved bird classifier and print the accuracy, precision, and recall as calculated on the test set
def main(argv):

    master_word_list = tef.gen_word_dict()
    
    image_dir_path = "CUB_200_2011/CUB_200_2011/images"
    class_names = os.listdir(image_dir_path)
    class_names.remove(".DS_Store")
    
    # Get information on image classes, train-test split
    image_mapping = pd.read_csv("CUB_200_2011/CUB_200_2011/images.txt", sep = " ", header=None, names=["Index","Image"])
    image_classes = pd.read_csv("CUB_200_2011/CUB_200_2011/image_class_labels.txt", sep = " ", header=None, names=["Index","Class"])
    data_split = pd.read_csv("CUB_200_2011/CUB_200_2011/train_test_split.txt", sep = " ", header=None, names=["Index","Subset"])
    
    # hyperparameters
    num_test_batches = 100

    # instantiate network
    model = tef.bird_classify( len(master_word_list) )
    model.load_state_dict(torch.load("bow_weights.pth"))
    
    
    
    
    
    
    # Select a set of random image names
    num_images = 20
    image_ind = np.random.randint(image_mapping.shape[0], size=num_images)
    image_names = [ image_mapping.iloc[image_ind[i],:]["Image"][:-4] for i in range(num_images) ]
    
    # Generate a numerical embedding for the images' captions
    # Embedding is 200 features long
    embedding = tef.gen_embed( model, image_names, master_word_list )
    print(embedding.shape)
    
    
    
    
    
    
    # Test the effectiveness of the model for classification
    accuracy, precision, recall = tef.precision_recall( model, num_test_batches, master_word_list, image_mapping, image_classes, data_split, "train" )
    print(f"Train Accuracy: {accuracy}")
    print(f"Train Precision: {precision}")
    print(f"Train Recall: {recall}")


    accuracy, precision, recall = tef.precision_recall( model, num_test_batches, master_word_list, image_mapping, image_classes, data_split, "test" )
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")


    

    return

if __name__ == "__main__":
    main(sys.argv)
