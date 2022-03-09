#!/usr/bin/env python
# coding: utf-8

# Core python includes
import gc
import sys
import logging
import os
import math

# Data preparation
import numpy as np
import pescador

from sample import SampleDataset,MAX_WINDOW_SIZE
from model import load_command_net, load_attention_net
from trainer import train
from music_generator import generate_a_song

# Pytorch setup
import torch

# Print human-interpretable outputs
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

# Set device to GPU if it is available, otherwise CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Would you like a [fresh] model, to [train] and existing model or [generate] music:")
line = sys.stdin.readline().strip()

model = load_attention_net

if line == "fresh":

    # Create a standard data loader from our samples
    loader = torch.utils.data.DataLoader(SampleDataset("../../training_data/", window_size=MAX_WINDOW_SIZE), num_workers=4, prefetch_factor=32)

    # Train a model with the data loader
    train(loader, model, None, device)
elif line == "train":

    # Create a standard data loader from our samples
    loader = torch.utils.data.DataLoader(SampleDataset("../../training_data/", window_size=MAX_WINDOW_SIZE), num_workers=4, prefetch_factor=32)

    # Train a model with the data loader
    train(loader, model, "./last.checkpoint", device)

elif line == "generate":

    # This loader is used as a seed to the NN and needs to
    # start on a complete sample (start_at_sample=True)
    # because we need to know which byte we are in within
    # the current sample when we generate new samples byte
    # by byte
    # We do not always train exactly on a sample when
    # training, which is why this is a flag.
    out_of_sample_loader = torch.utils.data.DataLoader(SampleDataset("../../out_of_sample/", window_size=MAX_WINDOW_SIZE, start_at_sample=True))

    # Generate a song using the out of sample loader
    generate_a_song(out_of_sample_loader, model, "./last.checkpoint", device)
