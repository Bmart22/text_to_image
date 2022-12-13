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
from PIL import Image
from math import ceil

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.functional as TF

import style_vec_functions as svf
import text_encode_functions as tef

class Generator(nn.Module):
    def __init__(self, num_features=200):
        super(Generator, self).__init__()
        
        # Size of feature maps in generator
        ngf = 64
        
        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( num_features, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
#        LAYERS = 3
#        layer_1_channels = num_features - ceil((num_features - 3) / LAYERS) + 1
#        layer_2_channels = layer_1_channels - ceil((num_features - 3) / LAYERS)
#        layer_3_channels = layer_2_channels - ceil((num_features - 3) / LAYERS)
#        print(layer_3_channels)

#        self.net = nn.Sequential(
#            nn.ConvTranspose2d(num_features, layer_1_channels, 3),
#            nn.BatchNorm2d(layer_1_channels),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(layer_1_channels, layer_2_channels, 3),
#            nn.BatchNorm2d(layer_2_channels),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(layer_2_channels, layer_3_channels, 3),
#            nn.BatchNorm2d(layer_3_channels),
#            nn.ReLU(True),
#            nn.Tanh()
#        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_size, 134, kernel_size=4, stride=4, padding=1),
            nn.BatchNorm2d(134),
            nn.LeakyReLU(True),
            nn.Conv2d(134, 68, kernel_size=4, stride=4, padding=1),
            nn.BatchNorm2d(68),
            nn.LeakyReLU(True),
            nn.Conv2d(68, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
#            nn.BatchNorm2d(),
#            nn.LeakyReLU(True)
        )

    def forward(self, x):
        return self.net(x)


    
    
#def tokenize_words(text):
#    '''Transforms an email into a list of words.
#
#    Parameters:
#    -----------
#    text: str. Sentence of text.
#
#    Returns:
#    -----------
#    Python list of str. Words in the sentence `text`.
#
#    This method is pre-filled for you (shouldn't require modification).
#    '''
#    # Define words as lowercase text with at least one alphabetic letter
#    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
#    return pattern.findall(text.lower())



#def find_top_words(word_freq, num_features=200):
#
#    key_value_pairs = word_freq.items()
#    top_pairs = sorted( key_value_pairs, key = lambda word: word[1], reverse=True )[:num_features]
#
#    top_words = [word[0] for word in top_pairs]
#
#    return top_words
#
#
#
## Get a dictionaary that counts all of the unique words in the captions
#def gen_word_dict():
#    dir_path="birds/text"
#
#    word_freq = {}
#
#    for bird_class in os.listdir(dir_path):
#        if bird_class != ".DS_Store":
#            for image_name in os.listdir(dir_path + "/" + bird_class):
#                with open(dir_path + "/" + bird_class + "/" + image_name, encoding = 'latin-1') as file:
#                    captions = tokenize_words( file.read() )
#                    for word in captions:
#                        if word in word_freq:
#                            word_freq[word] += 1
#                        else:
#                            word_freq[word] = 1
#
#    top_words = find_top_words(word_freq, num_features=500)
#
#    return top_words
#
#
#def embed_caption(tokenized_words, master_word_list):
#    embedding = np.zeros( (len(master_word_list),) )
#
#    for word in tokenized_words:
#        if word in master_word_list:
#            embedding[ master_word_list.index( word ) ] = 1
#
#    return embedding
    

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
#    print(class_ind)
#    print(class_ind & subset_ind)
#    print(image_ind)
    images = image_mapping[ class_ind & subset_ind ]
    
    # Select a random image name from our subset
    rand_index = np.random.randint( images.shape[0] )
    img = images.iloc[rand_index,:]["Index"]
    
    return img


# Generate the bag-of-words embedding for all captions
def gen_batch(bird_classifier, master_word_list, image_mapping, image_classes, data_split, subset):

    image_dir_path = "CUB_200_2011/CUB_200_2011/images/"

    # Choose index of one random image from each class
    num_classes = 200
    image_ind = np.zeros((num_classes,), dtype="int")
    image_names = []
    images = np.zeros((num_classes,3,64,64))
    for bird_class in range(num_classes):
        # Select a random image from this class
        image_ind[bird_class] = sample_class(image_mapping, image_classes, data_split, bird_class+1, subset)
        image_names.append( image_mapping["Image"][image_mapping["Index"] == image_ind[bird_class]].item()[:-4] )
        
        # Read in image, resize
        img = Image.open( image_dir_path + image_names[bird_class] + ".jpg" ).resize((64,64)).convert('RGB')
        # Move channel dimension to the first dimension
        images[bird_class] = np.array(img).transpose((2,0,1))
    
    
    # Generate the text embeddings and style vectors, then concatenate them
    batch_txt = tef.gen_embed( bird_classifier, image_names, master_word_list )
    style_vec = torch.from_numpy( svf.gen_style_vec( image_ind ) )
    embedding = torch.cat((batch_txt, style_vec), dim=1)
    
    return torch.from_numpy( images ), embedding

#    image_dir_path = "CUB_200_2011/CUB_200_2011/images"
#    caption_dir_path="birds/text"
#
#    num_classes = 200
#    embeddings = np.zeros( (num_classes, len(master_word_list)) )
#
#    for bird_class in range(num_classes):
#        # Select a random image from this class
#        chosen_img = sample_class(image_ind, image_classes, data_split, bird_class+1, subset)
#
#        # Select a random caption for this image
#        with open(caption_dir_path + "/" + chosen_img + ".txt", encoding = 'latin-1') as file:
#            captions = file.readlines()
#            cap = tokenize_words( captions[ np.random.randint(len(captions)) ] )
#            # Embed caption as bag-of-words
#            embeddings[bird_class] = embed_caption(cap, master_word_list)
#
#    return embeddings
    
# Train the network on the training dataset
def train_loop( generator, discriminator, bird_classifier, num_batches, loss_func, optimizerG, optimizerD, master_word_list, image_mapping, image_classes, data_split, loss_chart):
    
    num_classes = 200

    # tell model we are training, affects dropout layer behavior
    generator.train()
    discriminator.train()
    
    # For every batch
    for i in range(num_batches):
        
        # Get next batch
        batch_img, batch_txt = gen_batch(bird_classifier, master_word_list, image_mapping, image_classes, data_split, "train")
        
        
        shuffle_ind = np.random.permutation(num_classes)
        batch_img = batch_img[shuffle_ind].float() # shuffle batch
        
        print("batch size 1")
        print(batch_txt.shape)
        
        batch_txt = torch.rand(batch_txt.shape)
        
        batch_txt = batch_txt[shuffle_ind].float() # shuffle batch
        batch_txt = batch_txt.unsqueeze(-1).unsqueeze(-1) # reshape text embedding so to work with 2D convolution
        
        print("batch size 2")
        print(batch_txt.shape)
        
        
        ''' Train discriminator on real images '''
        discriminator.zero_grad()
        # Format batch
#        real_cpu = data[0].to(device)
#        b_size = real_cpu.size(0)
        label = torch.full((batch_img.shape[0],), 1, dtype=torch.float)
        # Forward pass real batch through D
        pred = discriminator(batch_img).view(-1)
        
        print("batch size 3")
        print(batch_img.shape)
        print(pred.shape)
        # Calculate loss on all-real batch
        lossD_real = loss_func(pred, label)
        # Calculate gradients for D in backward pass
        lossD_real.backward()
#        D_x = output.mean().item()
        
        
        ''' Train discriminator on fake images '''
#        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake_img = generator(batch_txt)
        
        print("fake img")
        print(fake_img.shape)
        
        label.fill_(0)
        # Classify all fake batch with D
        pred = discriminator(fake_img.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        lossD_fake = loss_func(pred, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        lossD_fake.backward()
#        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        lossD = lossD_real + lossD_fake
        loss_chart[1,i] = lossD.item() # Store the loss of the discriminator on the current batch
        # Update D
        optimizerD.step()
        
        ''' Train generator on fake images '''
        generator.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        pred = discriminator(fake_img).view(-1)
        # Calculate G's loss based on this output
        lossG = loss_func(pred, label)
        loss_chart[0,i] = lossG.item() # Store the loss of the discriminator on the current batch
        # Calculate gradients for G
        lossG.backward()
#        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        
        
        
        
#        pred = model(batch) # Perform forward pass
#        target = torch.from_numpy(shuffle_ind)
#
#        loss = loss_func(pred, target) # Compute loss
#        loss_chart[i] = loss.item() # Store the loss of the current batch
#
#        # Backpropagation
#        optimizer.zero_grad() # set all gradients to zero
#        loss.backward() # Compute the gradients for all layers of the network
#        optimizer.step() # Update the model's weights based on the gradients

    return
    
    

# Evaluates the network on the test set, returns the average loss across all batches
def precision_recall( model, num_batches, master_word_list, image_ind, image_classes, data_split, subset ):

    num_classes = 200

    # Keep running total of loss across all batches
    tp = 0 #true positives
    fp = 0 #false positives
    fn = 0 #false negatives
    tn = 0 #true negatives
    
    model.eval() # tell model we are evaluating, affects dropout layer behavior
    for i in range(num_batches): #For every batch
    
        # Get next batch
        batch = gen_batch(master_word_list, image_ind, image_classes, data_split, subset)
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
def gen_embed( generator, bird_classifier, image_ind, image_mapping, master_word_list ):

    image_dir_path = "CUB_200_2011/CUB_200_2011/images"
    caption_dir_path="birds/text"
    
    
    image_names = []
    for ind in image_ind:
        image_names.append( image_mapping["Image"][image_mapping["Index"] == ind].item()[:-4] )

#    text = np.zeros( (len(image_names), len(master_word_list)) )
    
    text = tef.gen_embed( bird_classifier, image_names, master_word_list )
    
    
    text = torch.rand(text.shape)
    
    
    style_vec = torch.from_numpy( svf.gen_style_vec( image_ind ) ).float()

    # Select a random caption for this image
#    for i in range(len(image_names)):
#        with open(caption_dir_path + "/" + image_names[i] + ".txt", encoding = 'latin-1') as file:
#            captions = file.readlines()
#            cap = tokenize_words( captions[ np.random.randint(len(captions)) ] )
            
            

            # Embed caption as bag-of-words
#            gen_batch(bird_classifier, master_word_list, image_mapping, image_classes, data_split, subset)
#            text[i] = embed_caption(cap, master_word_list)
            

    embedding = torch.cat((text, style_vec), dim=1).float().unsqueeze(-1).unsqueeze(-1)

    generator.eval() # tell model we are evaluating, affects dropout layer behavior

#    embedding = torch.from_numpy( embeddings ).float() # insert extra batch dimension at the beginning
    images = generator(embedding)

    return images


# Create grid of plots with 9 generated images
def graph_gen_images( data, filename ):
    fig, ax = plt.subplots(3,3, figsize=(6,6))
    for x in range(3):
        for y in range(3):
            # Graph the image
            img = data[x + 3*y].squeeze().detach().numpy().transpose((1,2,0))
            img = (img + 1) / 2
            print(img.min())
            ax[x,y].imshow( img )

            # label image with the image's predicted class
#            ax[x,y].set_title(f"Prediction: {int(pred_class[x + 3*y])}")

            # turn off numbered axes
            ax[x,y].set_xticks([])
            ax[x,y].set_yticks([])

    # Title and save image to jpg
    plt.suptitle("Generated Images")
    plt.savefig(filename)

    return
