# Based on https://github.com/caogang/wgan-gp/blob/master/gan_toy.py

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
            nn.Dropout(),
            nn.Linear(512, sample.WINDOW_SIZE),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=4),
            nn.ReLU(),
            nn.Linear(29, 1),
        )

    def forward(self, noise):
        output = self.main(noise)
        return output

class CommandNet(nn.Module):

    def __init__(self):
        super(CommandNet, self).__init__()

        self.main = nn.Sequential(
            nn.Dropout(),
            nn.Linear(DIM, sample.NUM_CMD),
            nn.Softmax()
        )

    def forward(self, noise):
        output = self.main(noise)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1),
        )

        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)
