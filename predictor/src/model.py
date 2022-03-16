import math

# Sample constants we need to size the model
from sample import MAX_WINDOW_SIZE,BYTES_PER_ENTRY

# Torch stuff
import torch
import torch.nn.functional as F
from torch import nn, optim
from local_attention import LocalAttention
from functools import reduce

NUM_LAYERS=4

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

        # self.conv_tan = CausalConv1d(input_channels, output_channels, kernel_size, dilation)
        # self.tanh = nn.Tanh()

        #separate weights for residual and skip channels
        self.conv_r = nn.Conv1d(output_channels, output_channels, 1)
        self.conv_s = nn.Conv1d(output_channels, skip_channels, 1)

        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        o = self.sig(self.conv_sig(x)) # * self.tanh(self.conv_tan(x))
        skip = self.conv_s(o)
        residual = self.conv_r(o)
        return self.norm(residual), skip

# A model that uses convolutions as a network to predict the next byte in a variable length sequence
# with a fixed receptive window (the number of previous data points considered when predicting the
# next data point).
class CommandNet(nn.Module):
    def __init__(self, skip_channels=256, num_blocks=4, num_layers=NUM_LAYERS, num_hidden=256, kernel_size=7*8, dilations=False): 
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
            self.receptive_field_size=(kernel_size*num_layers)

    def receptive_field(self):
        return self.receptive_field_size

    def forward(self, x):

        x = self.embed(x)
        x = self.positional_embedding(x)
        x = x.permute(0,2,1)
        x = self.causal_conv(x)

        skip_vals = []

        #run res blocks
        for i, layer in enumerate(self.res_stack):
            x, s = layer(x)
            skip_vals.append(s)

        #sum skip values and pass to last portion of network
        x = x + reduce((lambda a,b: a+b), skip_vals)

        x = self.relu1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)

        return x

class AttentionBlock(nn.Module):

    def __init__(self, dim, kernel_size, window_size):
        super(AttentionBlock, self).__init__() 

        self.conv_q = CausalConv1d(dim, dim, kernel_size)
        self.conv_k = CausalConv1d(dim, dim, kernel_size)
        self.conv_v = CausalConv1d(dim, dim, kernel_size)

        self.attention = LocalAttention(dim=dim, window_size=window_size, causal=True, autopad=True)
        self.conv = nn.Conv1d(dim, dim, 1)

        self.sig = nn.Sigmoid()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        x = x.permute(0, 2, 1)
        #print(x.shape)
        x = self.attention(q.permute(0, 2, 1), k.permute(0, 2, 1), v.permute(0, 2, 1)).permute(0, 2, 1)
        #print(x)
        #x = self.sig(x)
        x = self.conv(x)

        x = self.norm(x)
        return x 

class AttentionNet(nn.Module):

    def __init__(self, num_layers=3):
        super(AttentionNet, self).__init__()

        dim = 256
        self.embed = nn.Embedding(dim, dim)
        self.positional_embed = PositionalEncoding(dim)

        kernel_size=7*4

        self.causal_conv = CausalConv1d(dim, dim, kernel_size=kernel_size)
        
        self.res_stack = nn.ModuleList()

        for i in range(num_layers):
            self.res_stack.append(AttentionBlock(dim, kernel_size, 2048))

        # Bring it all back
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(dim, dim, 1)

        self.final = nn.Linear(256, 256)

        self.receptive_field_size = 4096

    def forward(self, x):
        x = self.embed(x)
        x = self.positional_embed(x)
        key = x
        x = x.permute(0,2,1)
        x = self.causal_conv(x)

        for i, layer in enumerate(self.res_stack):
            #print("Pre-res:", x.shape)
            x = layer(x)
            #print("Post-res:", x.shape)

        x = self.conv(x)
        x = self.relu(x)
        # x = F.log_softmax(x, dim=-1)
        return x

    def receptive_field(self):
        return self.receptive_field_size

# Load a command net model, either initialized with random values (if path is None) otherwise from an existing network saved on disk.
def load_model(model, path, device):

    lr = 0.0001
    momentum=0.8

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, min_lr=0.0001)
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
