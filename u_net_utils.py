import torch
import torch.nn as nn

"""
Implementation of U-net convolutional architecture
"""

class U_net(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        features = out_channels

        # we use the braai architecture for the encoder path
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= kernel_size),
            nn.Dropout(0.25)
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64* 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )


    def encoder(self, x):
        x = self.convolutional_block(x)
        x = self.classifier(x)
        return logits


    def decoder(self, x):

