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

        self.norm = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        o = self.sig(self.conv_sig(x)) # * self.tanh(self.conv_tan(x))
        skip = self.conv_s(o)
        residual = self.conv_r(o)
        return self.norm(residual), skip

# A model that uses convolutions as a network to predict the next byte in a variable length sequence
# with a fixed receptive window (the number of previous data points considered when predicting the
# next data point).
class CommandNet(nn.Module):
    def __init__(self, skip_channels=256, num_blocks=4, num_layers=10, num_hidden=256, kernel_size=7*8, dilations=False): 
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

        self.conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.relu2 = nn.ReLU()

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

    def __init__(self, dim, heads):
        super(AttentionBlock, self).__init__() 

        self.dim = dim
        self.heads = heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        # We run this linear layer on every input after computing attention
        self.final = nn.Linear(dim, dim)

        self.attention = LocalAttention(causal=True, dim=dim, window_size=101, dropout=0.1)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):

        # Input shape is (batch_size, dim, inp-window)
        batch_size = x.size(0)

        # Bring the shape back to (bs, inp-window, byte probabilities)
        x = x.permute(0, 2, 1)

        # Get q k and v by a run through out linear layer
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Reshape tensors so that we have (batch_size, head, inp-size / head, dim)
        q = q.view(batch_size, -1, self.heads, self.dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.heads, self.dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.heads, self.dim).permute(0, 2, 1, 3)

        # print(q.shape, mask.shape)

        # this will yield self.heads attention scores 
        x = self.attention(q, k, v)

        # Reverse the previous permute so we are back to (batch_size, inp-size / head, head, dim)
        x = x.contiguous().permute(0, 2, 1, 3).contiguous()

        # Reshape the tensor to (batch_size, inp_size, dim)
        x = x.view(batch_size, -1, self.dim) 

        # Execute a final layer on each input
        x = self.final(x)
         
        # Batch normalize the result
        # x = self.norm(x)

        return x.permute(0, 2, 1)

class AttentionNet(nn.Module):

    def __init__(self, num_layers=6):
        super(AttentionNet, self).__init__()

        dim = 256
        self.embed = nn.Embedding(dim, dim)
        self.positional_embed = PositionalEncoding(dim)

        kernel_size=7*4

        self.causal_conv = CausalConv1d(dim, dim, kernel_size=kernel_size)
       
        self.attn_stack = nn.ModuleList()
        self.res_stack = nn.ModuleList()

        for i in range(num_layers):
            self.attn_stack.append(AttentionBlock(dim, 9))
            self.res_stack.append(ResidualBlock(256, 256, kernel_size, skip_channels=256))

        # Bring it all back
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(dim, dim, 1)

        self.receptive_field_size = 512 * 7

    def forward(self, x):
        x = self.embed(x)
        x = self.positional_embed(x)

        x = x.permute(0,2,1)
        x = self.causal_conv(x)

        skip_vals = []

        #run res blocks
        for i in range(0, len(self.res_stack)):
            x = self.attn_stack[i](x)
            x, s = self.res_stack[i](x)
            skip_vals.append(s)

        # x = x + reduce((lambda a,b: a+b), skip_vals)

        x = self.conv(x)
        x = self.relu(x)
        # x = F.log_softmax(x, dim=-1)
        return x

    def receptive_field(self):
        return self.receptive_field_size

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.embed = nn.Embedding(256, 256)
        self.pos = PositionalEncoding(256)
        self.transformer = nn.Transformer(d_model = 256, nhead = 16, num_encoder_layers = 12, batch_first=True)

    def forward(self, x):
        device = x.device

        # Mask is a square matrix of [ true | false ] where each [row, column] is true if the input at step row can
        # be based on the data at step column
        # i.e, [[True, False]] means that the first input can use the first input but not the second
        # mask = torch.tril(torch.ones(3590, 3590)).bool().to(device)
        # TODO: This won't change much so precompute a maximum mask to speed stuff up
        mask = generate_square_subsequent_mask(x.size(1), device)

        x = self.embed(x)
        x = self.pos(x)
        x = self.transformer(src = x, tgt = x, src_mask = mask, tgt_mask = mask)
        x = F.log_softmax(x, dim = -1)
        return x

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

    optimizer = optim.Adam(
        model.parameters(),
        lr = 0.000001,
    )

    # optimizer = optim.SGD ( model.parameters(), lr = 0.001 )
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, min_lr=0.0000000001, patience=2)

    # This scheduler expects step to be called with step(epoch, loss)
    scheduler = AdaptiveWarmup(optimizer, start_lr=0.000001, end_lr=0.001, num_steps=10, criterion=lr_criterion, underlying_scheduler=scheduler_lr)

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

def load_transformer_net(path, device):
    return load_model(TransformerNet(), path, device)
