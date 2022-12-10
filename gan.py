#
# Phil Butler
# gan.py
# Fall 2022
# CS 7180
# Resources:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# https://github.com/soumith/dcgan.torch/blob/master/main.lua
# https://github.com/aelnouby/Text-to-Image-Synthesis/blob/master/models/gan.py

import torch
from torch import nn
from bow_training import bird_classify
from math import ceil


class Generator(nn.Module):
    def __init__(self, num_features=200):
        super(Generator, self).__init__()
        LAYERS = 3
        layer_1_channels = num_features - ceil((num_features - 3) / LAYERS) + 1
        layer_2_channels = layer_1_channels - ceil((num_features - 3) / LAYERS)
        layer_3_channels = layer_2_channels - ceil((num_features - 3) / LAYERS)
        print(layer_3_channels)

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
            nn.Tanh()
        )

    def forward(self, x):
        print(x)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_size, 134, 1),
            nn.BatchNorm2d(134),
            nn.LeakyReLU(True),
            nn.Conv2d(134, 68, 1),
            nn.BatchNorm2d(),
            nn.LeakyReLU(True),
            nn.Conv2d(3, 3, 1),
            nn.BatchNorm2d(),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        return self.net(x)


def main():
    # Load text_embedder
    PATH = 'bow_weights.pth'
    text_embedder = bird_classify(500)
    text_embedder.load_state_dict(torch.load(PATH))  # it takes the loaded dictionary, not the path file itself
    text_embedder.eval()

    # Put a sample text into an embedding
    text_embedding = text_embedder(torch.rand(500))
    text_embedding = torch.reshape(text_embedding, (1, 200, 1, 1))

    # Doing a dry run of the untrained generator on one image
    gen = Generator()
    output = gen(text_embedding)
    print(output.shape)


if __name__ == '__main__':
    main()
