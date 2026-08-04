import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of U-net convolutional architecture
"""


class Feature(nn.Module):
    '''
    Define the convolutional block to extract features
    in each 
    '''

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        features = out_channels

        # we use the braai architecture 
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(features, features * 2, kernel_size=kernel_size),
            nn.ReLU()
        )
        self.num_filters = features * 2
        self.kernel_size = kernel_size
        self.max_pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = self.convolutional_block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, feature_block, features_list): # feature block is a instance of Feature
        super().__init__()

        # we use the braai architecture for the encoder path
        self.current_level = feature_block
        self.map_features = features_list

    def forward(self, x):
        x = self.current_level.convolutional_block(x)
        self.map_features.add(x)
        
        x = self.current_level.max_pool(x)
        x = self.current_level.dropout(x)


        return x


class Decoder(nn.Module):

    def __init__(self, feature_block, features_list): # feature block is a instance of Feature
        super().__init__()

        # we use the braai architecture for the encoder path
        n_features = self.current_level.num_filters
        kernel_size = self.current_level.kernel_size

        self.current_level = feature_block
        self.map_features = features_list
        self.resize = nn.ConvTranspose2d(in_channels= n_features, out_channels= n_features / 2, 
                                         kernel_size= kernel_size, stride= kernel_size)
        

    def forward(self, x):
        x = self.resize(x)
        current_feature_skip = self.map_features.pop()
        skip_cropped = F.interpolate(current_feature_skip, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip_cropped], dim= 1)
        
        x = self.feature_block.forward(x)
        x = self.current_level.dropout(x)

        return x



class U_net(nn.Module):
    '''
    Define the convolutional block to extract features
    in each 
    '''
    def __init__(self, in_channels, out_channels, kernel_size, num_layers):
        super().__init__()

        self.map_features_list = []
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.convolutional_block = Feature(in_channels, out_channels, kernel_size)
        self.encoder = Encoder(self.convolutional_block, self.map_features_list)
        self.decoder = Decoder(self.convolutional_block, self.map_features_list)
        self.output_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):

        for _ in (self.num_layers + 1):
            x = Encoder(x)

        for _ in (self.num_layers):
            x = Decoder(x)

        logits = self.output_layer(x)
        return logits
        








