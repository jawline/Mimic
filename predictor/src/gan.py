# Based on https://github.com/caogang/wgan-gp/blob/master/gan_toy.py

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sample

NUM_EVENTS_PER_ROUND = sample.WINDOW_SIZE
DIM = sample.NUM_INP * NUM_EVENTS_PER_ROUND

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
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
