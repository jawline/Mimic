import math

# Sample constants we need to size the model
from sample import MAX_WINDOW_SIZE,BYTES_PER_ENTRY

# Torch stuff
import torch
import torch.nn.functional as F
from torch import nn, optim
from local_attention import LocalAttention
from functools import reduce

KERNEL_SIZE_SAMPLES=4
KERNEL_SIZE=BYTES_PER_ENTRY * KERNEL_SIZE_SAMPLES
NUM_LAYERS=4

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 1024 * 10):
        super().__init__()

        assert(MAX_WINDOW_SIZE <= max_len)

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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

# When using dilations the effective lookback is KERNEL_SIZE^num_layers
class CommandNet(nn.Module):
    def __init__(self, skip_channels=256, num_blocks=8, num_layers=NUM_LAYERS, num_hidden=256, kernel_size=KERNEL_SIZE): 
        super(CommandNet, self).__init__()

        self.embed = nn.Embedding(skip_channels, skip_channels)
        self.positional_embedding = PositionalEncoding(skip_channels)
        self.causal_conv = CausalConv1d(skip_channels, num_hidden, kernel_size)
        self.res_stack = nn.ModuleList()

        for b in range(num_blocks):
            for i in range(num_layers):
                self.res_stack.append(ResidualBlock(num_hidden, num_hidden, kernel_size, skip_channels=skip_channels, dilation=2**i))

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(skip_channels, skip_channels, 1)

        # The size of the receptive field is kernel size exponentially increased by the number
        # of layers because we are using dilations
        self.receptive_field_size=kernel_size**num_layers

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
        o = reduce((lambda a,b: a+b), skip_vals)

        o = self.relu1(o)
        o = self.conv1(o)
        o = self.relu2(o)
        o = self.conv2(o)

        return o

# WIP Replacement for the CausalConv that uses local attention
class AttentionResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, skip_channels, dilation=1):
        super(AttentionResBlock, self).__init__()

        self.attn_sig = LocalAttention(window_size=512, dim=input_channels, causal=True)
        self.sig = nn.Sigmoid()

        self.attn_tan = LocalAttention(window_size=512, dim=input_channels, causal=True)
        self.tanh = nn.Tanh()

        #separate weights for residual and skip channels
        self.conv_r = nn.Conv1d(output_channels, output_channels, 1)
        self.conv_s = nn.Conv1d(output_channels, skip_channels, 1)

    def forward(self, x):
        o = self.sig(self.attn_sig(x)) * self.tanh(self.attn_tan(x))
        skip = self.conv_s(o)
        residual = self.conv_r(o)
        return residual, skip

class AttentionNet(nn.Module):
    def __init__(self, skip_channels=256, num_blocks=4, num_layers=5, num_hidden=256, kernel_size=KERNEL_SIZE):
        super(AttentionNet, self).__init__()

        self.embed = nn.Embedding(skip_channels, skip_channels)
        self.positional_embedding = PositionalEncoding(skip_channels)
        self.causal_conv = CausalConv1d(skip_channels, num_hidden, kernel_size)
        self.res_stack = nn.ModuleList()

        for b in range(num_blocks):
            for i in range(num_layers):
                self.res_stack.append(AttentionResBlock(num_hidden, num_hidden, kernel_size, skip_channels=skip_channels, dilation=2**i))

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(skip_channels, skip_channels, 1)

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
        o = reduce((lambda a,b: a+b), skip_vals)

        o = self.relu1(o)
        o = self.conv1(o)
        o = self.relu2(o)
        o = self.conv2(o)

        return o

# Load a command net model, either initialized with random values (if path is None) otherwise from an existing network saved on disk.
def load_command_net(path, device):

    lr = 0.01
    momentum=0.8

    command_generator = CommandNet()
    optimizer = optim.SGD(
        command_generator.parameters(),
        lr=lr,
        momentum=momentum
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.97, min_lr=0.0001)
    command_generator = command_generator.to(device)

    # This needs to be after to because the optimizer decides what device to send the tensors to based on the
    # device of the model.
    if path != None:
        command_generator.load_state_dict(torch.load(path + ".model"))
        optimizer.load_state_dict(torch.load(path + ".optimizer"))
        #scheduler = torch.load(path + ".scheduler")

    return command_generator, optimizer, scheduler
