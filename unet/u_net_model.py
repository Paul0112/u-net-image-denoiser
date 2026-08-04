from unet.u_net_utils import *

# Implementation of U_net with the architecture defined in utils

class U_net(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels):
        super().__init__()

        self.map_features_list = []
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        curr_in = in_channels
        curr_out = out_channels

        # Instance the encoders levels
        for _ in range(num_levels):
            feat = Feature(curr_in, curr_out)
            self.encoders.append(Encoder(feat, self.map_features_list))
            curr_in = curr_out
            curr_out = curr_out * 2

        # U shape
        self.bottom = Feature(curr_in, curr_out)

        # Instance the decoders levels
        for _ in range(num_levels):
            feat = Feature(curr_out, curr_in)
            self.decoders.append(Decoder(feat, self.map_features_list))
            curr_out = curr_in
            curr_in = curr_in // 2

        self.output_layer = nn.Conv2d(out_channels, 1, kernel_size=1)


    def forward(self, x):
        self.map_features_list.clear()

        for encoder in self.encoders: # down
            x = encoder(x)

        x = self.bottom(x) # bottom

        for decoder in self.decoders: # up
            x = decoder(x)

        logits = self.output_layer(x)
        return logits

