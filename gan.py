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
import torch.nn.functional as F
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
        print(x)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=3, stride=2)
        #self.conv1_bn = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 200, kernel_size=3, stride=2)
        #self.conv2_bn = nn.BatchNorm2d(200)
        self.out = nn.Conv2d(400, 1, kernel_size=4, stride=1)

    def forward(self, x, text_embedding):
        print('input shape:', x.shape)
        #x = self.conv1_bn(self.conv1(x))
        x = self.conv1(x)
        x = F.leaky_relu(x)
        print('conv1 shape:', x.shape)
        #x = self.conv2_bn(self.conv2(x))
        x = self.conv2(x)
        x = F.leaky_relu(x)
        print('conv2 shape:', x.shape)

        # Concatenate replicated text embedding
        x = torch.cat((x, text_embedding), 1)

        return self.out(x)


def main():
    # Load text_embedder
    PATH = 'bow_weights.pth'
    text_embedder = bird_classify(500)
    text_embedder.load_state_dict(torch.load(PATH))  # it takes the loaded dictionary, not the path file itself
    text_embedder.eval()

    # Put a sample text into an embedding
    text_embedding = text_embedder(torch.rand(500))
    text_embedding = torch.reshape(text_embedding, (1, 200, 1, 1))

    # Dry run of the untrained generator on one image
    gen = Generator()
    output = gen(text_embedding)
    print(output.shape)

    # Dry run the discriminator
    dis = Discriminator()

    out = dis(output, text_embedding.repeat(1, 1, 4, 4))
    print('d score:', out)

if __name__ == '__main__':
    main()
