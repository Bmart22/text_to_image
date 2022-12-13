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
    

# Randomly select an image of the given class from the given train/test split
def sample_class(image_mapping, image_classes, data_split, img_class, subset):
    # Select all indices of images of the desired class
    class_ind = image_classes["Class"] == img_class
    
    # Select the indices of images in the desired train or test subset
    if subset == "train":
        subset_ind = data_split["Subset"] == 0
    elif subset == "test":
        subset_ind = data_split["Subset"] == 1
    
    # Select all image names in our desired subset
    images = image_mapping[ class_ind & subset_ind ]
    
    # Select a random image name from our subset
    rand_index = np.random.randint( images.shape[0] )
    img = images.iloc[rand_index,:]["Image"][:-4]
    
    return img


# Generate the bag-of-words embedding for all captions
def gen_batch(master_word_list, image_mapping, image_classes, data_split, subset):
    image_dir_path = "CUB_200_2011/CUB_200_2011/images"
    caption_dir_path="birds/text"
    
    num_classes = 200
    embeddings = np.zeros( (num_classes, len(master_word_list)) )

    for bird_class in range(num_classes):
        # Select a random image from this class
        chosen_img = sample_class(image_mapping, image_classes, data_split, bird_class+1, subset)
        
        # Select a random caption for this image
        with open(caption_dir_path + "/" + chosen_img + ".txt", encoding = 'latin-1') as file:
            captions = file.readlines()
            cap = tokenize_words( captions[ np.random.randint(len(captions)) ] )
            # Embed caption as bag-of-words
            embeddings[bird_class] = embed_caption(cap, master_word_list)
    
    return embeddings
    
# Train the network on the training dataset
# Batch size is set in the dataloader
def train_loop( model, num_batches, loss_func, optimizer, master_word_list, image_mapping, image_classes, data_split, loss_chart):
    num_classes = 200

    # tell model we are training, affects dropout layer behavior
    model.train()
    
    # For every batch
    for i in range(num_batches):
        
        # Get next batch
        batch = gen_batch(master_word_list, image_mapping, image_classes, data_split, "train")
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

# Evaluates the network on the test set, returns the average loss across all batches
def precision_recall( model, num_batches, master_word_list, image_mapping, image_classes, data_split, subset ):

    num_classes = 200

    # Keep running total of loss across all batches
    tp = 0 #true positives
    fp = 0 #false positives
    fn = 0 #false negatives
    tn = 0 #true negatives
    
    model.eval() # tell model we are evaluating, affects dropout layer behavior
    for i in range(num_batches): #For every batch
    
        # Get next batch
        batch = gen_batch(master_word_list, image_mapping, image_classes, data_split, subset)
        shuffle_ind = np.random.permutation(num_classes) # this is also the correct classifications
        batch = torch.from_numpy( batch[shuffle_ind] ).float() # shuffle batch
    
        pred = model(batch) #Perform forward pass
        pred = np.argmax( pred.detach().numpy(), axis = 1 ) #get predicted classes
        y = shuffle_ind
        
        tp += np.logical_and(pred == 1, y == 1).sum()
        fp += np.logical_and(pred == 1, y == 0).sum()
        fn += np.logical_and(pred == 0, y == 1).sum()
        tn += np.logical_and(pred == 0, y == 0).sum()
        
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    precision = (tp / (tp+fp)) if (tp+fp != 0) else 0
    recall = (tp / (tp+fn)) if (tp+fn != 0) else 0
    
    return accuracy, precision, recall


# Create an embedding for each of the images in "image_names"
def gen_embed( model, image_names, master_word_list ):
    
    image_dir_path = "CUB_200_2011/CUB_200_2011/images"
    caption_dir_path="birds/text"
    
    embeddings = np.zeros( (len(image_names), len(master_word_list)) )
    
    # Select a random caption for this image
    for i in range(len(image_names)):
        with open(caption_dir_path + "/" + image_names[i] + ".txt", encoding = 'latin-1') as file:
            captions = file.readlines()
            cap = tokenize_words( captions[ np.random.randint(len(captions)) ] )
            # Embed caption as bag-of-words
            embeddings[i] = embed_caption(cap, master_word_list)
    
    model.eval() # tell model we are evaluating, affects dropout layer behavior
    
    init_embed = torch.from_numpy( embeddings ).float() # insert extra batch dimension at the beginning
    embed = model(init_embed)
    
    return embed



