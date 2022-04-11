import math

# Sample constants we need to size the model
from sample import MAX_WINDOW_SIZE,BYTES_PER_ENTRY

# Torch stuff
import torch
import torch.nn.functional as F
from torch import nn, optim
from local_attention import LocalAttention

# Transformers go crazy and start outputting NaN if the LR is too high early for the first few epochs, but
# the LR can be increased after some initial weights are adjusted. We use this gradual warmup scheduler
# to automate that process.
from adaptive_warmup import Scheduler as AdaptiveWarmup

"""
Positional encoding steps encode information about where we are in a sequence of data into the data
allowing a model to change how it responds to an input value based on where it occurs in a sequence
of inputs.

TODO: Currently there is no clear identifier for where we are in a piece of music so I don't think
that PositionalEncoding should add anything to the model, but maybe I am misunderstanding it's usage
"""
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

"""
Causal convolutions prevent the convolution calculation for x[i] considering data from
any input[j] where j > i by padding the input with [kernel_size] - 1 zeros.

For example, while a standard convolution with kernel size 5 on an input [1, 2, 3, 4, 5]
may look like:
    x[0] = kernel([0, 0, 1, 2, 3])
    x[1] = kernel([0, 1, 2, 3, 4])
    x[2] = kernel([1, 2, 3, 4, 5])
    x[3] = kernel([2, 3, 4, 5, 0])
    x[3] = kernel([3, 4, 5, 0, 0])
a causal convolution padding on the same input would look like:
    x[0] = kernel([0, 0, 0, 0, 1])
    x[1] = kernel([0, 0, 0, 1, 2])
    x[2] = kernel([0, 0, 1, 2, 3])
    x[3] = kernel([0, 1, 2, 3, 4])
    x[4] = kernel([1, 2, 3, 4, 5])

If the convolution has a dilation > 1 then we will multiply the amount of padding by [dilation]
because dilated convolutions consider every [dilation * kernel_size] inputs around a target output.
"""
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, **kwargs):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size , dilation=dilation, **kwargs)

    def forward(self, x):
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

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn = LocalAttention(
            dim = dim,
            window_size = 512,
            causal = True,
            look_backward = 1,
            look_forward = 0,
            dropout = 0.1,
            autopad = True,
            exact_windowsize = False)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        x = self.attn(q, k, v)
        return x

"""
When used with the ConvLM we need to permute the dimensions
before using a self attention layer since the conv representation is (batch_sz, dim, seq_size) but LocalAttention expects (batch_sz, seq_size, dim).
"""
class PermutedResidualAttentionBlock(nn.Module):
    def __init__(self, dim):
        super(PermutedResidualAttentionBlock, self).__init__()
        self.attn = ResidualBlock(AttentionBlock(dim))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.attn(x)
        x = x.permute(0, 2, 1)
        return x

class ModelLayer(nn.Module):
    def __init__(self, causal, dim, hfactor, batch_norm, layer_dropout):
        super(ModelLayer, self).__init__()
        residual = ResidualBlock(Pointwise(dim, hfactor, batch_norm))
        layers = [causal, residual]
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim))

        if layer_dropout is not None:
            layers.append(nn.Dropout(p=layer_dropout))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

"""
This layer combines a causal convolution layer and a residual layer together, optionally
batch normalizing the output. A stack of these blocks forms our CausalConv model.
"""
def CausalConvModelLayer(dim, hfactor, kernel_size, batch_norm, dilation, layer_dropout):
    return ModelLayer(CausalConv1d(dim, dim, kernel_size, dilation), dim, hfactor, batch_norm, layer_dropout)


"""
This layer combines an attention block and a residual layer together, optionally
batch normalizing the output.
"""
def AttentionModelLayer(dim, hfactor, batch_norm, layer_dropout):
    return ModelLayer(PermutedResidualAttentionBlock(dim), dim, hfactor, batch_norm, layer_dropout)

"""
A model that combines layers of either convolutions or local attention to predict the next byte in
a variable length sequence.

hfactor marks the number factor by which the dimensions should be expanded when doing the pointwise
layers (e.g, Conv(256, 256 * [hfactor]))

When [dilations] is true [num_blocks] sets the number of stacked blocks of [layers], if [dilations] is not true then
num_blocks should be one.
"""
class GameboyNet(nn.Module):
    def __init__(self,
            dim=256,
            num_blocks=1,
            layer_spec=["attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention", "attention"],
            hfactor=4,
            layer_dropout=0.0,
            kernel_size=BYTES_PER_ENTRY*30,
            dilations=False,
            batch_norm=True): 
        super(GameboyNet, self).__init__()

        if not dilations:
            assert(num_blocks == 1)
       
        def dilation(i):
            if dilations:
                return 2**i
            else:
                return 1

        def make_layer(i):
            spec = layer_spec[i]
            if spec == "attention":
                return AttentionModelLayer(
                        dim=dim,
                        hfactor=hfactor,
                        batch_norm=batch_norm,
                        layer_dropout=layer_dropout)
            elif spec == "convolution":
                return CausalConvModelLayer(
                        dim=dim,
                        hfactor=hfactor,
                        kernel_size=kernel_size,
                        batch_norm=batch_norm,
                        dilation=dilation(i),
                        layer_dropout=layer_dropout)

        num_layers = len(layer_spec)

        # First we embed and then add positional encodings to our input
        self.prepare_input = nn.Sequential(*[nn.Embedding(dim, dim)])

        # Build the core of our model by stacking [layers] CausalConvModelLayer instances on top of each other.
        layers = [make_layer(layer_idx) for layer_idx in range(num_layers) for _block in range(num_blocks)]
        self.layers = nn.Sequential(*layers)

        # Combine all the channels and then activate as a final step
        self.finalize = nn.Sequential(*[nn.Conv1d(dim, dim, 1), nn.ReLU()])

        # TODO: This is wrong if we are using attention
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
        x = self.forward(x)
        return x.permute(0, 2, 1)
        

"""
Stop warming up if loss starts increasing
"""
def lr_criterion(epoch, last_lr, last_loss, current_lr, current_loss):
    if epoch > 2:
        if last_loss < current_loss:
            return last_lr
        else:
            return None
    else:
        return None

"""
Load a model, either initialized with random values if [path] is None or from an existing network
saved on disk if [path] is a string.
"""
def load_model(model, path, device):

    optimizer = optim.AdamW(
        model.parameters(),
        lr = 0.0001,
    )

    # optimizer = optim.SGD ( model.parameters(), lr = 0.001 )
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, min_lr=0.0000000001, patience=1)
    scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = optim.lr_scheduler.ChainedScheduler([scheduler_plateau, scheduler_step])

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
