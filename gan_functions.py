#
# Brendan Martin, Phil Butler
# gan_functions.py
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
import torch.nn.functional as F

import style_vec_functions as svf
import text_encode_functions as tef

# Tutorial-based GAN

#class Generator(nn.Module):
#    def __init__(self, num_features=200):
#        super(Generator, self).__init__()
#
#        # Size of feature maps in generator
#        ngf = 64
#
#        self.net = nn.Sequential(
#            # input is Z, going into a convolution
#            nn.ConvTranspose2d( num_features, ngf * 8, 4, 1, 0, bias=False),
#            nn.BatchNorm2d(ngf * 8),
#            nn.ReLU(True),
#            # state size. (ngf*8) x 4 x 4
#            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf * 4),
#            nn.ReLU(True),
#            # state size. (ngf*4) x 8 x 8
#            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf * 2),
#            nn.ReLU(True),
#            # state size. (ngf*2) x 16 x 16
#            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf),
#            nn.ReLU(True),
#            # state size. (ngf) x 32 x 32
#            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
#            nn.Tanh()
#            # state size. (nc) x 64 x 64
#        )
#
#    def forward(self, x):
#        return self.net(x)
#
#class Discriminator(nn.Module):
#    def __init__(self, input_size):
#        super(Discriminator, self).__init__()
#        self.net = nn.Sequential(
#            nn.Conv2d(input_size, 134, kernel_size=4, stride=4, padding=1),
#            nn.BatchNorm2d(134),
#            nn.LeakyReLU(True),
#            nn.Conv2d(134, 68, kernel_size=4, stride=4, padding=1),
#            nn.BatchNorm2d(68),
#            nn.LeakyReLU(True),
#            nn.Conv2d(68, 1, kernel_size=4, stride=1, padding=0),
#            nn.Sigmoid()
#        )
#
#    def forward(self, x):
#        return self.net(x)


# Paper-based GAN

class Generator(nn.Module):
    def __init__(self, num_features=200):
        super(Generator, self).__init__()
        LAYERS = 3
        layer_1_channels = num_features - ceil((num_features - 3) / LAYERS) + 1
        layer_2_channels = layer_1_channels - ceil((num_features - 3) / LAYERS)
        layer_3_channels = layer_2_channels - ceil((num_features - 3) / LAYERS)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(num_features, layer_1_channels, 3),
            nn.BatchNorm2d(layer_1_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_1_channels, layer_2_channels, 3),
            nn.BatchNorm2d(layer_2_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_2_channels, layer_3_channels, 3),
            nn.BatchNorm2d(layer_3_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_3_channels, layer_3_channels, 3),
            nn.BatchNorm2d(layer_3_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_3_channels, layer_3_channels, 3),
            nn.BatchNorm2d(layer_3_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_3_channels, layer_3_channels, 3),
            nn.BatchNorm2d(layer_3_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_3_channels, layer_3_channels, 3),
            nn.BatchNorm2d(layer_3_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_3_channels, layer_3_channels, 3),
            nn.BatchNorm2d(layer_3_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(layer_3_channels, layer_3_channels, 3),
            nn.BatchNorm2d(layer_3_channels),
            nn.ReLU(True),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_size, 100, kernel_size=3, stride=2),
            nn.LeakyReLU(True),
            nn.Conv2d(100, 200, kernel_size=3, stride=2),
            nn.LeakyReLU(True)
        )
        
        self.out = nn.Conv2d(400, 1, kernel_size=4, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, text_embedding):

        x = self.net(x)

        # Concatenate replicated text embedding
        x = torch.cat((x, text_embedding), 1)
        x = self.out(x)

        return self.sig(x)


    
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
    img = images.iloc[rand_index,:]["Index"]
    
    return img


# Generate a batch of bag-of-words embeddings and images
def gen_batch(bird_classifier, master_word_list, image_mapping, image_classes, data_split, subset):
    img_size = 19

    image_dir_path = "CUB_200_2011/CUB_200_2011/images/"

    # Choose index of one random image from each class
    num_classes = 200
    image_ind = np.zeros((num_classes,), dtype="int")
    image_names = []
    images = np.zeros((num_classes,3,img_size,img_size))
    for bird_class in range(num_classes):
        # Select a random image from this class
        image_ind[bird_class] = sample_class(image_mapping, image_classes, data_split, bird_class+1, subset)
        image_names.append( image_mapping["Image"][image_mapping["Index"] == image_ind[bird_class]].item()[:-4] )
        
        # Read in image, resize
        img = Image.open( image_dir_path + image_names[bird_class] + ".jpg" ).resize((img_size,img_size)).convert('RGB')
        # Move channel dimension to the first dimension
        images[bird_class] = np.array(img).transpose((2,0,1))
    
    
    # Generate the text embeddings and style vectors, then concatenate them
    batch_txt = tef.gen_embed( bird_classifier, image_names, master_word_list )
    style_vec = torch.from_numpy( svf.gen_style_vec( image_ind ) )
    embedding = torch.cat((batch_txt, style_vec), dim=1)
    
    return torch.from_numpy( images ), embedding

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

#        batch_txt = torch.rand(batch_txt.shape)
        
        batch_txt = batch_txt[shuffle_ind].float() # shuffle batch
        batch_txt = batch_txt.unsqueeze(-1).unsqueeze(-1) # reshape text embedding so to work with 2D convolution
        batch_txt_D = batch_txt[:,:200,:,:].repeat([1,1,4,4])
        
        
        ''' Train discriminator on real images '''
        discriminator.zero_grad()
        # Format batch
#        real_cpu = data[0].to(device)
#        b_size = real_cpu.size(0)
        label = torch.full((batch_img.shape[0],), 1, dtype=torch.float)
        # Forward pass real batch through D
        pred = discriminator(batch_img, batch_txt_D).view(-1)
        
        # Calculate loss on all-real batch
        lossD_real = loss_func(pred, label)
        # Calculate gradients for D in backward pass
        lossD_real.backward(retain_graph=True)
#        D_x = output.mean().item()
        
        
        ''' Train discriminator on fake images '''
#        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake_img = generator(batch_txt)
        
        label.fill_(0)
        # Classify all fake batch with D
        pred = discriminator(fake_img.detach(), batch_txt_D).view(-1)
        print("pred", pred, "type", type(pred))
        
        # Calculate D's loss on the all-fake batch
        lossD_fake = loss_func(pred, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        lossD_fake.backward(retain_graph=True)
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
        pred = discriminator(fake_img, batch_txt_D).view(-1)
        # Calculate G's loss based on this output
        lossG = loss_func(pred, label)
        loss_chart[0,i] = lossG.item() # Store the loss of the discriminator on the current batch
        # Calculate gradients for G
        lossG.backward()
#        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

    return


# Create an embedding for each of the images in "image_ind"
def gen_embed( generator, bird_classifier, image_ind, image_mapping, master_word_list ):

    image_dir_path = "CUB_200_2011/CUB_200_2011/images"
    caption_dir_path="birds/text"
    
    
    image_names = []
    for ind in image_ind:
        image_names.append( image_mapping["Image"][image_mapping["Index"] == ind].item()[:-4] )

    # Generate text embedding and style vector
    text = tef.gen_embed( bird_classifier, image_names, master_word_list )
    style_vec = torch.from_numpy( svf.gen_style_vec( image_ind ) ).float()
    
    # Concatenate inputs
    embedding = torch.cat((text, style_vec), dim=1).float().unsqueeze(-1).unsqueeze(-1)
    
    generator.eval() # tell model we are evaluating, affects dropout layer behavior
    
    # Generate images
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
            ax[x,y].imshow( img )

            # turn off numbered axes
            ax[x,y].set_xticks([])
            ax[x,y].set_yticks([])

    # Title and save image to jpg
    plt.suptitle("Generated Images")
    plt.savefig(filename)

    return
