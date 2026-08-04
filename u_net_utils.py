import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Implementation of U-net architecture

class Feature(nn.Module):
    '''
    Define the convolutional block to extract features
    in each level of encode and decode layers.
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
    
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= 3),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, kernel_size= 3),
            nn.ReLU(inplace= True)
        )

        self.num_filters = out_channels * 2
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = self.convolutional_block(x)
        return x


class Encoder(nn.Module):
    '''
    Define the encode level of the architecture; apply the convolutional block of Feature
    (save the current features) and max_pool and dropout to the layer.
    '''
    def __init__(self, feature_block, features_list): # feature block is a instance of Feature
        super().__init__()

        self.current_level = feature_block
        self.map_features = features_list
        self.max_pool = nn.MaxPool2d(kernel_size= 2, stride= 2)


    def forward(self, x):
        x = self.current_level(x)
        self.map_features.append(x)
        x = self.max_pool(x)
        x = self.current_level.dropout(x)

        return x


class Decoder(nn.Module):
    '''
    Define the decode level of the architecture; apply upsampling (via ConvTranspose2d), load and crop
    the skip connections saved in features_list (Encoder path) and finally apply the convolutional block 
    of Feature and dropout to the layer.
    '''

    def __init__(self, feature_block, features_list): # feature block is a instance of Feature
        super().__init__()

        self.current_level = feature_block
        self.map_features = features_list
        self.resize = nn.ConvTranspose2d(in_channels= self.current_level.in_channels, 
                                         out_channels= self.current_level.in_channels // 2, 
                                         kernel_size= 2, stride= 2)
        
    def forward(self, x):
        x = self.resize(x)

        current_feature_skip = self.map_features.pop()
        skip_cropped = TF.center_crop(current_feature_skip, output_size= x.shape[2:])

        x = torch.cat([skip_cropped, x], dim= 1)
        x = self.current_level(x)
        x = self.current_level.dropout(x)

        return x









