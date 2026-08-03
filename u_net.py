import torch
import torch.nn as nn

"""
Implementation of U-net convolutional architecture (by Ronnerberger et al.)

This architecture consist of a contracting path (adding continuos convolutional 
layers and pooling in each step) to capture context and a symmetric expanding 
path (retrieving the original shape from the convolutional layers by upsampling
operators and combining with features obtained in the contracting path) 
that enables precise localization.
"""

class U_net(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()


        self.features = nn.Sequential(
            nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= kernel_size - 1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64* 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return logits

