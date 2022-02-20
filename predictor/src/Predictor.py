#!/usr/bin/env python
# coding: utf-8

# Core python includes
import gc
import sys
import logging
import os
import math

# Data preperation
import numpy as np
import pescador

# Pytorch bits
import torch

# Print human-interpretable outputs
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

# Set device to GPU if it is available, otherwise CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from sample import SampleDataset,MAX_WINDOW_SIZE
from model import load_command_net
from trainer import train
from music_generator import generate_a_song

print("Would you like to [train] or [generate]:")
line = sys.stdin.readline().strip()

if line == "train":
    loader = torch.utils.data.DataLoader(SampleDataset("../../training_data/",window_size=MAX_WINDOW_SIZE))
    train(loader, load_command_net, "./last.checkpoint", device)
elif line == "generate":

    # This loader is used as a seed to the NN and needs to
    # start on a complete sample (starrt_at_sample=True)
    # because we need to know which byte we are in within
    # the current sample when we generate new samples byte
    # by byte
    # We do not always train exactly on a sample when
    # training, which is why this is a flag.
    out_of_sample_loader = torch.utils.data.DataLoader(SampleDataset("../../out_of_sample/", window_size=MAX_WINDOW_SIZE, start_at_sample=True))

    # Generate a song using the out of sample loader
    generate_a_song(out_of_sample_loader, load_command_net, "./last.checkpoint", device)
