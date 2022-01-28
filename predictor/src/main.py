import sample
import gan

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
EPOCHS = 20
ROUND_SZ = 1000
CUDA = torch.cuda.is_available()

def extract_and_pad(data, cmd, field_offset):
    new_data = np.array([data[(i * sample.NUM_INP) + field_offset] for i in range(sample.WINDOW_SIZE) if data[(i * sample.NUM_INP) + sample.CMD_OFFSET] == cmd])
    new_data = np.pad(new_data, pad_width=(0, sample.WINDOW_SIZE - len(new_data)))
    new_data = torch.Tensor(np.array([new_data]))
    return new_data

def prepare_data_train(data):

    data_train_cmd= torch.Tensor(np.array([data]))
    data_train_time = torch.Tensor(np.array([[data[(i * sample.NUM_INP) + sample.TIME_OFFSET] for i in range(sample.WINDOW_SIZE)]]))
    data_train_freq_lsb = extract_and_pad(data, sample.CMD_LSB, sample.FREQLSB_OFFSET)
    data_train_freq_msb = extract_and_pad(data, sample.CMD_MSB, sample.FREQMSB_OFFSET)

    if CUDA:
        data_train_cmd = data_train_cmd.cuda()
        data_train_time = data_train_time.cuda()
        data_train_freq_lsb = data_train_freq_lsb.cuda()
        data_train_freq_msb = data_train_freq_msb.cuda()

    return data_train_cmd, data_train_time, data_train_freq_lsb, data_train_freq_msb


def train():

    time_generator = gan.TimeNet()

    lr=0.0001
    time_optimizer= optim.Adam(time_generator.parameters(), lr=lr, weight_decay=1e-5)
    time_criterion = nn.MSELoss()

    channel_generator = gan.ChannelNet()
    channel_optimizer= optim.Adam(channel_generator.parameters(), lr=lr, weight_decay=1e-5)
    channel_criterion = nn.CrossEntropyLoss()

    command_generator = gan.CommandNet()
    command_optimizer= optim.Adam(command_generator.parameters(), lr=lr, weight_decay=1e-5)
    command_criterion = nn.CrossEntropyLoss()

    # The two frequency generators are trained when we are predicting a
    # frequency command only
    freq_msb_generator = gan.FreqNet()
    freq_msb_optimizer = optim.Adam(command_generator.parameters(), lr=lr, weight_decay=1e-5)
    freq_msb_criterion = nn.MSELoss()

    freq_lsb_generator = gan.FreqNet()
    freq_lsb_optimizer = optim.Adam(command_generator.parameters(), lr=lr, weight_decay=1e-5)
    freq_lsb_criterion = nn.MSELoss()

    if CUDA:
        time_generator = time_generator.cuda()
        channel_generator = channel_generator.cuda()
        command_generator = command_generator.cuda()
        freq_msb_generator = freq_msb_generator.cuda()
        freq_lsb_generator = freq_lsb_generator.cuda()

    print("Collecting training data")
    train_gen = sample.create_data_split(sample.training_files("./training_data/"), 1)
    print("Collected")

    for iteration in range(EPOCHS):
        print(f"Round {iteration}")

        for i in range(ROUND_SZ):
          data = next(train_gen)

          time_optimizer.zero_grad()
          command_optimizer.zero_grad()
          freq_lsb_optimizer.zero_grad()
          freq_msb_optimizer.zero_grad()

          data_train_cmd, data_train_time, data_train_freq_lsb, data_train_freq_msb = prepare_data_train(data['X'][0])

          data_test = data['Y'][0]
          data_test_time = torch.Tensor(np.array([[data_test[0]]])).to(torch.float)
          data_test_channel = torch.Tensor(np.array([data_test[1:3]]))
          data_test_command= torch.Tensor(np.array([sample.onehot_cmd(data_test)]))

          data_test_freq_lsb = torch.Tensor(np.array([[data_test[sample.FREQLSB_OFFSET]]]))
          data_test_freq_msb = torch.Tensor(np.array([[data_test[sample.FREQMSB_OFFSET]]]))

          count = 0
          for x in data_test_command[0]:
              if x.item() == 1.:
                  count += 1
          if count > 1:
              raise "count?!?!?"

          if CUDA:
              data_test_time = data_test_time.cuda()
              data_test_channel = data_test_channel.cuda()
              data_test_command = data_test_command.cuda()
              data_test_freq_lsb = data_test_freq_lsb.cuda()
              data_test_freq_msb = data_test_freq_msb.cuda()

          prediction_time = time_generator(data_train_time)

          #print("TRAIN", data_train)
          #print("TEST", prediction_time, data_test_time)
          time_loss = time_criterion(prediction_time, data_test_time)
          time_loss.backward()
          time_optimizer.step()

          time_ltgt = data_test_time
          time_lpred = prediction_time

          prediction_command = command_generator(data_train_cmd)
          #print(prediction_command.flatten(), data_test_command.flatten())
          command_loss = command_criterion(prediction_command, data_test_command)
          command_loss.backward()
          command_optimizer.step()

          command_ltgt = data_test_command
          command_lpred = prediction_command

          if data_test[sample.CMD_LSB] == 1.:
              #print("Train LSB")
              prediction_freq_lsb = freq_lsb_generator(data_train_freq_lsb)
              freq_lsb_loss = freq_lsb_criterion(prediction_freq_lsb, data_test_freq_lsb)
              freq_lsb_loss.backward()
              freq_lsb_optimizer.step()

          if data_test[sample.CMD_MSB] == 1.:
              #print("Train MSB")
              prediction_freq_msb = freq_msb_generator(data_train_freq_msb)
              freq_msb_loss = freq_msb_criterion(prediction_freq_msb, data_test_freq_msb)
              freq_msb_loss.backward()
              freq_msb_optimizer.step()

        print("Time batch loss:", time_loss.item())
        print("Last prediction:", time_ltgt, time_lpred)
        print("Command batch loss:", command_loss.item())
        print("Last prediction:", command_ltgt, command_lpred)
        print("Freq LSB loss:", freq_lsb_loss.item())
        print("Freq LSB last pred:", prediction_freq_lsb, data_test_freq_lsb)
        print("Freq MSB loss:", freq_msb_loss.item())
        print("Freq MSB last pred:", prediction_freq_msb, data_test_freq_msb)

    return data['X'][0], time_generator.eval(), channel_generator.eval(), command_generator.eval(), freq_lsb_generator.eval(), freq_msb_generator.eval()

seed, time_generator, channel_generator, command_generator, freq_lsb_generator, freq_msb_generator = train()

def pred_cmd(offset, cmd_pred):
    return offset == cmd_pred.argmax().item()

for i in range(10000):

    data_train_cmd, data_train_time, data_train_freq_lsb, data_train_freq_msb = prepare_data_train(seed)

    next_channel = channel_generator(data_train_cmd)

    if pred_cmd(sample.CH_1_OFFSET - 1, next_channel):
        next_channel = 1
    elif pred_cmd(sample.CH_2_OFFSET - 1, next_channel):
        next_channel = 2

    next_time = sample.unnorm(time_generator(data_train_time)[0].item(), sample.NORMALIZE_TIME_BY)
    next_cmd = command_generator(data_train_cmd)

    #print(next_time)
    #print(next_cmd)

    print("Next cmd", next_cmd)

    if pred_cmd(sample.NOP_CMD_OFFSET - 3, next_cmd):
        fresh = sample.nop()
        print("NOP - WHY DID I PREDICT A NO-OP?")
    elif pred_cmd(sample.DUTY_LL_CMD_OFFSET - 3, next_cmd):
        fresh = sample.duty_ll(1, [0, 0, 0, 0, 0, 0, 0], next_time)
        print("DUTY LL TODO")
    elif pred_cmd(sample.VOLENVPER_CMD_OFFSET - 3, next_cmd):
        fresh = sample.volenvper(1, [0, 0, 0, 0, 0, 0, 0], next_time)
        print("VOLENVPER TODO")
    elif pred_cmd(sample.FREQMSB_CMD_OFFSET - 3, next_cmd):
        pred = freq_msb_generator(data_train_freq_msb)[0].item()
        fresh = sample.freqmsb(1, [0, 0, 0, sample.unnorm(pred, 7.), 0, 0], next_time)
        print("FREQMSB AT ", next_time, sample.unnorm(pred, 7.))
    elif pred_cmd(sample.FREQLSB_CMD_OFFSET - 3, next_cmd):
        pred = freq_lsb_generator(data_train_freq_msb)[0].item()
        fresh = sample.freqlsb(1, [0, 0, 0, sample.unnorm(pred, 255.), 0, 0], next_time)
        print(f"FREQLSB {sample.unnorm(pred, 255.)} AT {next_time}")
    else:
        print("??")

    seed = np.concatenate((seed[sample.NUM_INP:], fresh))
    #print("new seed", seed)
    #print(fresh)
