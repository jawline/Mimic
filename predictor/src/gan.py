import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sample

NUM_EVENTS_PER_ROUND = sample.WINDOW_SIZE
DIM = sample.NUM_INP * NUM_EVENTS_PER_ROUND

class TimeNet(nn.Module):
    def __init__(self):
        super(TimeNet, self).__init__()

        self.main = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(sample.WINDOW_SIZE, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1)
        )

    def forward(self, noise):
        output = self.main(noise)
        return output

class CommandNet(nn.Module):

    def __init__(self):
        super(CommandNet, self).__init__()

        self.main = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, sample.NUM_CMD),
            nn.Softmax(dim=1)
        )

    def forward(self, noise):
        output = self.main(noise)
        return output

class ChannelNet(nn.Module):

    def __init__(self):
        super(ChannelNet, self).__init__()

        self.main = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, noise):
        output = self.main(noise)
        return output

class FreqNet(nn.Module):

    def __init__(self):
        super(FreqNet, self).__init__()

        self.main = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(sample.WINDOW_SIZE, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1)
        )

    def forward(self, noise):
        output = self.main(noise)
        return output
