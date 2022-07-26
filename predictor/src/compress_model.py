import math
import torch
import torch.nn.functional as F
from torch import nn, optim
from sample import BYTES_PER_ENTRY

COMPRESSED_WINDOW_SIZE = 2
EXPANSION_SIZE=128

class CompressionBlock(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "encode":
            self.layer = nn.Sequential(*[nn.Linear(BYTES_PER_ENTRY, EXPANSION_SIZE), nn.ReLU(), nn.Linear(EXPANSION_SIZE, COMPRESSED_WINDOW_SIZE), nn.ReLU()])
        elif mode == "decode":
            self.layer = nn.Sequential(*[nn.Linear(COMPRESSED_WINDOW_SIZE, EXPANSION_SIZE), nn.ReLU(), nn.Linear(EXPANSION_SIZE, BYTES_PER_ENTRY), nn.ReLU()])
        else:
            raise Exception('expected encode / decode')

    def forward(self, x): 

        batch_size = x.shape[0]

        if self.mode == "encode":
            x = x.reshape(batch_size, -1, BYTES_PER_ENTRY)
        elif self.mode == "decode":
            x = x.reshape(batch_size, -1, COMPRESSED_WINDOW_SIZE)
        else:
            raise Exception('expected encode / decode')

        x = self.layer(x)
        x = x.reshape(batch_size, -1)

        return x


class CompressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.encode = CompressionBlock("encode")
        self.decode = CompressionBlock("decode")

    def forward(self, x):
        return self.decode(self.encode(x))

"""
TODO: This was stolen from model.py almost verbatim, make generic.
"""
def load_model(model, path, device):

    optimizer = optim.AdamW(
        model.parameters(),
        lr = 0.0001,
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    model = model.to(device)

    if path != None:
        print("Loading from " + path)
        model.load_state_dict(torch.load(path + ".model"))
        optimizer.load_state_dict(torch.load(path + ".optimizer"))
        pass

    return model, optimizer, scheduler

def load(path, device):
    return load_model(CompressionModel(), path, device)
