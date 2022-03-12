import math

# Sample constants we need to size the model
from sample import MAX_WINDOW_SIZE,BYTES_PER_ENTRY

# Torch stuff
import torch
import torch.nn.functional as F
from torch import nn, optim
from local_attention import LocalAttention
from functools import reduce

KERNEL_SIZE=7*4
NUM_LAYERS=12

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 1024 * 32):
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
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size , dilation=dilation, **kwargs)

    def forward(self, x):
        #pad here to only add to the left side
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, skip_channels, dilation=1):
        super(ResidualBlock, self).__init__()

        self.conv_sig = CausalConv1d(input_channels, output_channels, kernel_size, dilation)
        self.sig = nn.Sigmoid()

        self.conv_tan = CausalConv1d(input_channels, output_channels, kernel_size, dilation)
        self.tanh = nn.Tanh()

        #separate weights for residual and skip channels
        self.conv_r = nn.Conv1d(output_channels, output_channels, 1)
        self.conv_s = nn.Conv1d(output_channels, skip_channels, 1)

    def forward(self, x):
        o = self.sig(self.conv_sig(x)) * self.tanh(self.conv_tan(x))
        skip = self.conv_s(o)
        residual = self.conv_r(o)
        return residual, skip

# A model that uses convolutions as a network to predict the next byte in a variable length sequence
# with a fixed receptive window (the number of previous data points considered when predicting the
# next data point).
class CommandNet(nn.Module):
    def __init__(self, skip_channels=256, num_blocks=4, num_layers=NUM_LAYERS, num_hidden=256, kernel_size=KERNEL_SIZE, dilations=False): 
        super(CommandNet, self).__init__()

        self.embed = nn.Embedding(skip_channels, skip_channels)
        self.positional_embedding = PositionalEncoding(skip_channels)
        self.causal_conv = CausalConv1d(skip_channels, num_hidden, kernel_size)
        self.res_stack = nn.ModuleList()

        for b in range(num_blocks):
            for i in range(num_layers):
                # dilation=2**i
                if dilations:
                    self.res_stack.append(ResidualBlock(num_hidden, num_hidden, kernel_size, skip_channels=skip_channels, dilation=2**i))
                else:
                    self.res_stack.append(ResidualBlock(num_hidden, num_hidden, kernel_size, skip_channels=skip_channels))

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(skip_channels, skip_channels, 1)

        # When using dilations the effective lookback is KERNEL_SIZE^num_layers otherwise it is KERNEL_SIZE*num_layers
        if dilations:
            # The size of the receptive field is kernel size exponentially increased by the number
            # of layers because we are using dilations
            self.receptive_field_size=kernel_size**num_layers
        else:
            # If we don't use dilations then the receptive field size is linear in the number of layers
            self.receptive_field_size=(kernel_size*num_layers) * 4

    def receptive_field(self):
        return self.receptive_field_size

    def forward(self, x):

        o = self.embed(x)
        o = self.positional_embedding(o)
        o = o.permute(0,2,1)
        o = self.causal_conv(o)

        skip_vals = []

        #run res blocks
        for i, layer in enumerate(self.res_stack):
            o, s = layer(o)
            skip_vals.append(s)

        #sum skip values and pass to last portion of network
        o = o + reduce((lambda a,b: a+b), skip_vals)

        o = self.relu1(o)
        o = self.conv1(o)
        o = self.relu2(o)
        o = self.conv2(o)

        return o

class AttentionNet(nn.Module):

    def __init__(self, skip_channels=256, num_blocks=4, num_layers=5, num_hidden=256, kernel_size=KERNEL_SIZE):
        super(AttentionNet, self).__init__()

        self.embed = nn.Embedding(skip_channels, skip_channels)
        self.positional_embed = PositionalEncoding(skip_channels)

        self.transformer = nn.Transformer(d_model=256, batch_first=True)

        self.final = nn.Linear(256, 256)

        self.receptive_field_size = KERNEL_SIZE*NUM_LAYERS

    # Generate a input mask mask matrix of [(len, len)] where each row 0 < i <= len has i zeros and then (len - i) -infinities.
    # This permits all of the data up until the current datapoint to be considered but masks out any after the current datapoint.
    # Eg: 
    #
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]
    #
    # (Based on code from  https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1)
    def get_tgt_mask(self, size):
        # Lower triangular matrix makes a matrix (size, size) where column[i] has i False rows followed by size - i True rows
        # e.g, column[0] is all True and the final column is all False except for the final element which is True.
        mask = torch.tril(torch.ones(size, size) == 1) 
        # Create a float tensor of size mask where 
        mask = mask.float().masked_fill(mask == True, 0.).masked_fill(mask == False, float('-inf')) 
        return mask

    # First we convert our inputs to byte-embedding (vectors of 256 values)
    # next we positionally encode our embeddings
    # after that, we pass or sequence into a transformer model
    # finally, we pass the output of the transformer into a final linear layer
    # and softmax the output (normalize so everything is close to either zero
    # or one).
    def forward(self, x, tgt_mask):
        x = self.embed(x)
        x = self.positional_embed(x)
        x = self.transformer(x, x, tgt_mask)
        x = self.final(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def receptive_field(self):
        return self.receptive_field_size

# Load a command net model, either initialized with random values (if path is None) otherwise from an existing network saved on disk.
def load_model(model, path, device):

    lr = 0.01
    momentum=0.8

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, min_lr=0.0001)
    model = model.to(device)

    # This needs to be after to because the optimizer decides what device to send the tensors to based on the
    # device of the model.
    if path != None:
        print("Loading from " + path)
        model.load_state_dict(torch.load(path + ".model"))
        optimizer.load_state_dict(torch.load(path + ".optimizer"))
        #scheduler = torch.load(path + ".scheduler")

    return model, optimizer, scheduler

def load_command_net(path, device):
    return load_model(CommandNet(), path, device)

def load_attention_net(path, device):
    return load_model(AttentionNet(), path, device)
