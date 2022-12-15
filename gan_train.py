#
# Phil Butler, Brendan Martin
# gan_train.py
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


# This will train the GAN on the training dataset, save the GAN weights, and save a loss plot
def main():
    master_word_list = tef.gen_word_dict() # length = 500
    # Get information on image classes, train-test split
    image_mapping = pd.read_csv("CUB_200_2011/CUB_200_2011/images.txt", sep = " ", header=None, names=["Index","Image"])
    image_classes = pd.read_csv("CUB_200_2011/CUB_200_2011/image_class_labels.txt", sep = " ", header=None, names=["Index","Class"])
    data_split = pd.read_csv("CUB_200_2011/CUB_200_2011/train_test_split.txt", sep = " ", header=None, names=["Index","Subset"])
    

    # Load text_embedder
    PATH = 'bow_weights.pth'
    text_embedder = bird_classify(len(master_word_list))
    text_embedder.load_state_dict(torch.load(PATH))  # it takes the loaded dictionary, not the path file itself
    text_embedder.eval()

    # Put a sample text into an embedding
    text_embedding = text_embedder(torch.rand(500))
    text_embedding = torch.reshape(text_embedding, (1, 200, 1, 1))

    # Doing a dry run of the untrained generator on one image
    gen = gf.Generator(245)
#    output = gen(text_embedding)
#    print("Gen output:")
#    print(output.shape)
    
    discrim = gf.Discriminator(3)
    
    
    # hyperparameters
    learning_rate = 1e-3
    num_epochs = 50
    num_train_batches = 50
    
    # training components
    loss_func = nn.BCELoss() #loss function
    optimizerG = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(discrim.parameters(), lr=learning_rate, betas=(0.5, 0.999))
#    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #stochastic gradient descent
    
    # storage for loss plot
    curr_training_loss = np.zeros( (2, num_train_batches, ) )
    training_loss = np.ones( (2, num_train_batches*num_epochs) )
    
    
    
    # Perform training
    for e in range(num_epochs): #for every epoch
        print( f"Epoch {e}: \n" )
        gf.train_loop( gen, discrim, text_embedder, num_train_batches, loss_func, optimizerG, optimizerD, master_word_list, image_mapping, image_classes, data_split, curr_training_loss ) #train data
        print(curr_training_loss.mean())
        training_loss[:, e*num_train_batches : (e+1)*num_train_batches] = curr_training_loss #Store training loss
    print("Finished\n")
    
    
    # Plot loss graph
    plt.figure()
    # Plot the training loss after each batch is processed
    plt.plot( np.arange(0, num_train_batches * num_epochs)/num_train_batches, training_loss.T )
    plt.legend(["Generator loss", "Discriminator loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over training")
    plt.savefig("gan_loss_plot.jpg")
    
    
    # Save trained GAN weights
    torch.save(gen.state_dict(), "weights/generator_weights.pth")
    torch.save(discrim.state_dict(), "weights/discriminator.pth")
    
    
    
#    gf.graph_gen_images( data )


if __name__ == '__main__':
    main()
