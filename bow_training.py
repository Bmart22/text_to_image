#
# Brendan Martin
# bow_training.py
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

# main function
def main(argv):

    master_word_list = tef.gen_word_dict()
    
    image_dir_path = "CUB_200_2011/CUB_200_2011/images"
#    class_names = os.listdir(image_dir_path)
#    class_names.remove(".DS_Store")

    # Get information on image classes, train-test split
    image_ind = pd.read_csv("CUB_200_2011/CUB_200_2011/images.txt", sep = " ", header=None, names=["Index","Image"])
    image_classes = pd.read_csv("CUB_200_2011/CUB_200_2011/image_class_labels.txt", sep = " ", header=None, names=["Index","Class"])
    data_split = pd.read_csv("CUB_200_2011/CUB_200_2011/train_test_split.txt", sep = " ", header=None, names=["Index","Subset"])

    # hyperparameters
    learning_rate = 1e-0
    num_epochs = 10
    num_train_batches = 100


    # instantiate network
    model = tef.bird_classify( len(master_word_list) )


    # training components
    loss_func = nn.CrossEntropyLoss() #loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #stochastic gradient descent

    # storage for loss plot
    curr_training_loss = np.zeros( (num_train_batches, ) )
    training_loss = np.ones( (num_train_batches*num_epochs, ) )


    # Perform training
    for e in range(num_epochs): #for every epoch
        print( f"Epoch {e}: \n" )
        tef.train_loop( model, num_train_batches, loss_func, optimizer, master_word_list, image_ind, image_classes, data_split, curr_training_loss ) #train data
        print(curr_training_loss.mean())
        training_loss[e*num_train_batches : (e+1)*num_train_batches] = curr_training_loss #Store training loss
    print("Finished\n")



    # Plot loss graph
    plt.figure()
    # Plot the training loss after each batch is processed
    plt.plot( np.arange(0, num_train_batches * num_epochs)/num_train_batches, training_loss )
#    plt.plot( training_loss )
#    # Plot the test set loss after each epoch
#    plt.plot( np.arange(1, num_epochs+1), test_loss, "r." )
#    plt.legend(["Train loss", "Test loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over training")
    plt.savefig("loss_plot.jpg")

    # Save the model weights
#    torch.save(model.state_dict(), "bow_weights.pth")
    

    return

if __name__ == "__main__":
    main(sys.argv)
