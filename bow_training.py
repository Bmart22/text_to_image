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

# class definition
class bird_classify(nn.Module):
    def __init__(self, input_size):
        # call nn.Module constructor
        super().__init__()
        
        
        # network architecture
        self.net = nn.Sequential(
            nn.Linear( input_size, 1024, bias=True ),
            nn.ReLU(),
            nn.Linear( 1024, 512, bias=True ),
            nn.ReLU(),
            nn.Linear( 512, 200, bias=True ),
#            nn.Softmax(),
        )

    # Define forward pass
    def forward(self, x):
        y = self.net( x )
        return y

    
def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())



def find_top_words(word_freq, num_features=200):
    
    key_value_pairs = word_freq.items()
    top_pairs = sorted( key_value_pairs, key = lambda word: word[1], reverse=True )[:num_features]
    
    top_words = [word[0] for word in top_pairs]
    
    return top_words
    
    

# Get a dictionaary that counts all of the unique words in the captions
def gen_word_dict():
    dir_path="birds/text"
    
    word_freq = {}
    
    for bird_class in os.listdir(dir_path):
        if bird_class != ".DS_Store":
            for image_name in os.listdir(dir_path + "/" + bird_class):
                with open(dir_path + "/" + bird_class + "/" + image_name, encoding = 'latin-1') as file:
                    captions = tokenize_words( file.read() )
                    for word in captions:
                        if word in word_freq:
                            word_freq[word] += 1
                        else:
                            word_freq[word] = 1

    top_words = find_top_words(word_freq, num_features=500)

    return top_words

    
def embed_caption(tokenized_words, master_word_list):
    embedding = np.zeros( (len(master_word_list),) )
    
    for word in tokenized_words:
        if word in master_word_list:
            embedding[ master_word_list.index( word ) ] = 1
        
    return embedding
    

# Randomly sample an image from a given bird class and train/test subset
def sample_class(image_ind, image_classes, data_split, img_class, subset):
    # Select all indices of images of the desired class
    class_ind = image_classes["Class"] == img_class
    
    # Select the indices of images in the desired train or test subset
    if subset == "train":
        subset_ind = data_split["Subset"] == 0
    elif subset == "test":
        subset_ind = data_split["Subset"] == 1
    
    # Select all image names in our desired subset
    images = image_ind[ class_ind & subset_ind ]
    
    # Select a random image name from our subset
    rand_index = np.random.randint( images.shape[0] )
    img = images.iloc[rand_index,:]["Image"][:-4]
    
    return img

# Generate the bag-of-words embedding for all captions
def gen_batch(master_word_list, image_ind, image_classes, data_split, subset):
    image_dir_path = "CUB_200_2011/CUB_200_2011/images"
    caption_dir_path="birds/text"
    
    num_classes = 200
    embeddings = np.zeros( (num_classes, len(master_word_list)) )

    for bird_class in range(num_classes):
        # Select a random image from this class
        chosen_img = sample_class(image_ind, image_classes, data_split, bird_class+1, subset)
        
        # Select a random caption for this image
        with open(caption_dir_path + "/" + chosen_img + ".txt", encoding = 'latin-1') as file:
            captions = file.readlines()
            cap = tokenize_words( captions[ np.random.randint(len(captions)) ] )
            # Embed caption as bag-of-words
            embeddings[bird_class] = embed_caption(cap, master_word_list)
    
    return embeddings

# Train the network on the training dataset
# Batch size is set in the dataloader
def train_loop( model, num_batches, loss_func, optimizer, master_word_list, image_ind, image_classes, data_split, loss_chart):
    num_classes = 200

    # tell model we are training, affects dropout layer behavior
    model.train()
    
    # For every batch
    for i in range(num_batches):
        
        # Get next batch
        batch = gen_batch(master_word_list, image_ind, image_classes, data_split, "train")
        shuffle_ind = np.random.permutation(num_classes) # this is also the correct classifications
        batch = torch.from_numpy(batch[shuffle_ind]).float() # shuffle batch
        
        
        pred = model(batch) # Perform forward pass
        target = torch.from_numpy(shuffle_ind)
        
        loss = loss_func(pred, target) # Compute loss
        loss_chart[i] = loss.item() # Store the loss of the current batch
        
        # Backpropagation
        optimizer.zero_grad() # set all gradients to zero
        loss.backward() # Compute the gradients for all layers of the network
        optimizer.step() # Update the model's weights based on the gradients

    return

# main function
def main(argv):

    master_word_list = gen_word_dict()
    
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
    model = bird_classify( len(master_word_list) )


    # training components
    loss_func = nn.CrossEntropyLoss() #loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #stochastic gradient descent

    # storage for loss plot
    curr_training_loss = np.zeros( (num_train_batches, ) )
    training_loss = np.ones( (num_train_batches*num_epochs, ) )


    # Perform training
    for e in range(num_epochs): #for every epoch
        print( f"Epoch {e}: \n" )
        train_loop( model, num_train_batches, loss_func, optimizer, master_word_list, image_ind, image_classes, data_split, curr_training_loss ) #train data
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
    torch.save(model.state_dict(), "bow_weights.pth")
    

    return

if __name__ == "__main__":
    main(sys.argv)
