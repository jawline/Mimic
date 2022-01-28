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
            nn.Dropout(p=0.5),
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

        self.cmd_embedding = nn.Embedding(sample.CMD_COUNT, 1)

        self.main = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Conv1d(1, 1, kernel_size=sample.NUM_INP),
            nn.Conv1d(1, 1, kernel_size=sample.NUM_INP),
            nn.Conv1d(1, 1, kernel_size=sample.NUM_INP),
            nn.ReLU(),
            nn.Linear(985, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, sample.CMD_COUNT),
            nn.Softmax(dim=1)
        )

    def forward(self, sequence):
        #print([sequence[0][i + sample.CMD_OFFSET].item() for i in range(0, DIM, sample.NUM_INP)])
        #print("cmd", sequence[:,0 + sample.CMD_OFFSET])
        sequence = [self.cmd_embedding(sequence[:,i + sample.CMD_OFFSET].type(torch.IntTensor).cuda()) for i in range(0, DIM, sample.NUM_INP)]
        sequence = torch.cat(sequence, 1)
        output = self.main(sequence)
        return output

class ChannelNet(nn.Module):

    def __init__(self):
        super(ChannelNet, self).__init__()

        self.main = nn.Sequential(
            nn.Dropout(p=0.6),
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
