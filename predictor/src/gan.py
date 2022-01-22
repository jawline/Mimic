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
            nn.Linear(sample.WINDOW_SIZE, 1)
        )

    def forward(self, noise):
        output = self.main(noise)
        return output

class CommandNet(nn.Module):

    def __init__(self):
        super(CommandNet, self).__init__()

        self.main = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(DIM, 64),
            nn.ReLU(),
            nn.Linear(64, sample.NUM_CMD),
            nn.ReLU(),
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
            nn.Linear(sample.WINDOW_SIZE, 1)
        )

    def forward(self, noise):
        output = self.main(noise)
        return output
