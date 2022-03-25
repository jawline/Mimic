import math

# Sample constants we need to size the model
from sample import MAX_WINDOW_SIZE,BYTES_PER_ENTRY

# Torch stuff
import torch
import torch.nn.functional as F
from torch import nn, optim
from local_attention import LocalAttention
from functools import reduce

# Transformers go crazy and start outputting NaN if the LR is too high early for the first few epochs, but
# the LR can be increased after some initial weights are adjusted. We use this gradual warmup scheduler
# to automate that process.
from adaptive_warmup import Scheduler as AdaptiveWarmup

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len = 1024 * 32):
        super().__init__()

        assert(MAX_WINDOW_SIZE <= max_len)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, **kwargs):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size , dilation=dilation, **kwargs)

    def forward(self, x):
        #pad here to only add to the left side
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class Pointwise(nn.Module):
    def __init__(self, dim, hfactor, kernel_size):
        super(Pointwise, self).__init__()

        expanded_dim = dim * hfactor

        layers = [
            nn.Conv1d(dim, expanded_dim, 1),
            nn.Sigmoid(),
            nn.Conv1d(expanded_dim, dim, 1),
        ]

        self.transform = nn.Sequential(*layers)

    def forward(self, x):
        return self.transform(x) 

"""
A residual block trains a layer to predict the residual of a previous
output (i.e, how much do we need to nudge the output by the get the
correct result).
"""
class ResidualBlock(nn.Module):
    def __init__(self, layer):
        super(ResidualBlock, self).__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)

"""
This layer combines a causal convolution layer and a residual layer together, optionally
batch normalizing the output. A stack of these blocks forms our CausalConv model.
"""
class CausalConvModelLayer(nn.Module):
    def __init__(self, dim, hfactor, kernel_size, batch_norm, dilation):
        super(CausalConvModelLayer, self).__init__()

        causal = CausalConv1d(dim, dim, kernel_size, dilation)
        residual = ResidualBlock(Pointwise(dim, hfactor, batch_norm))

        layers = [causal, residual]
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

"""
A model that uses convolutions as a network to predict the next byte in a variable length sequence
with a fixed receptive window (the number of previous data points considered when predicting the
next data point). Formed from a series of stacked CausalConvLMLayers.

hfactor marks the number factor by which the dimensions should be expanded when doing the pointwise
layers (e.g, Conv(256, 256 * [hfactor]))

When dilations is true num_blocks sets the number of stacked blocks of [layers], if [dilations] is not true then
num_blocks should be one.
"""
class GameboyNet(nn.Module):
    def __init__(self, dim=256, num_blocks=1, num_layers=10, hfactor=4, kernel_size=BYTES_PER_ENTRY*8, dilations=False, batch_norm=True): 
        super(GameboyNet, self).__init__()

        if not dilations:
            assert(num_blocks == 1)

        # First we embed and then add positional encodings to our input
        self.prepare_input = nn.Sequential(*[nn.Embedding(dim, dim), PositionalEncoding(dim)])
       
        def dilation(layer_index):
            if dilations:
                return 2**i
            else:
                return 1

        # Build the core of our model by stacking [layers] CausalConvModelLayer instances on top of each other.
        layers = [CausalConvModelLayer(dim, hfactor, kernel_size, batch_norm, dilation(layer_idx)) for layer_idx in range(num_layers) for _block in range(num_blocks)]
        self.layers = nn.Sequential(*layers)

        # Combine all the channels and then activate as a final step
        self.finalize = nn.Sequential(*[nn.Conv1d(dim, dim, 1), nn.ReLU()])

        # When using dilations the effective lookback is KERNEL_SIZE^num_layers otherwise it is KERNEL_SIZE*num_layers
        if dilations:
            # The size of the receptive field is kernel size exponentially increased by the number
            # of layers because we are using dilations
            self.receptive_field_size=kernel_size**num_layers
        else:
            # If we don't use dilations then the receptive field size is linear in the number of layers
            self.receptive_field_size=(kernel_size*num_layers*num_blocks)

    def receptive_field(self):
        return self.receptive_field_size

    def forward(self, x):
        x = self.prepare_input(x) 
        # Permute the input so that the embeddings are at dim 1 and the inputs for each embedding are at dim 2
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        x = self.finalize(x)
        return x

    """
    Calls forward and then permutes the result back to (batch_size, input_size, embedding_dim)
    from (batch_size, embedding_dim, input_size).
    """
    def predict(self, x):
        return self.forward(x).permute(0, 2, 1)
        

def lr_criterion(epoch, last_lr, last_loss, current_lr, current_loss):
    if epoch > 2:
        if last_loss < current_loss:
            return last_lr
        else:
            return None
    else:
        return None

# Load a command net model, either initialized with random values (if path is None) otherwise from an existing network saved on disk.
def load_model(model, path, device):

    optimizer = optim.AdamW(
        model.parameters(),
        lr = 0.0001,
    )

    # optimizer = optim.SGD ( model.parameters(), lr = 0.001 )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, min_lr=0.0000000001, patience=1)

    model = model.to(device)

    # This needs to be after to because the optimizer decides what device to send the tensors to based on the
    # device of the model.
    if path != None:
        print("Loading from " + path)
        model.load_state_dict(torch.load(path + ".model"))
        optimizer.load_state_dict(torch.load(path + ".optimizer"))
        #scheduler = torch.load(path + ".scheduler")
    else:
        # Fresh model so start with some adaptive warmup
        scheduler = AdaptiveWarmup(optimizer, start_lr=0.00000001, end_lr=0.0001, num_steps=5, criterion=lr_criterion, underlying_scheduler=scheduler, pass_through_loss_to_underlying=True)

    return model, optimizer, scheduler

def load_gameboy_net(path, device):
    return load_model(GameboyNet(), path, device)
