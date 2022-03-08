import sys
import numpy as np
import os

import pescador
import torch

# These commands enumerate the different kind of instruction we can send to each channel.
# Note: not every command is legal for every channel, invalid commands will be ignored.
CMD_VOLENVPER = 0
CMD_DUTYLL = 1
CMD_MSB = 2
CMD_LSB = 3
CMD_COUNT = 4

# These offsets mark the type of data index in sample form (not model form)
TIME_OFFSET = 0
CH_OFFSET = 1
CMD_OFFSET = 2
PARAM1_OFFSET = 3
PARAM2_OFFSET = 4
PARAM3_OFFSET = 5
SIZE_OF_INPUT_FIELDS = 6

# The maximum number of samples we will send to the model in a single iteration
MAX_WINDOW_SIZE = 1024 * 32

# The Gameboy cycles this many times per second. This is the
# measurement of time we use in our TIME_OFFSET values
M_CYCLES_PER_SECOND = 4194304.

# This is the number of bytes in every individual sample given
# to the model after it is converted to bytes
BYTES_PER_ENTRY = 7

def fresh_input(command, channel, time):
    newd = np.zeros(shape=SIZE_OF_INPUT_FIELDS, dtype=int)
    newd[TIME_OFFSET] = time
    newd[CH_OFFSET] = channel
    newd[CMD_OFFSET] = command
    return newd

def parse_bool(v):
    if v == "true":
        return 1
    elif v == "false":
        return 0
    else:
        return int(v)

def command_of_parts(command, channel, parts, time):
    inp = fresh_input(command, channel, time)

    if command == CMD_DUTYLL:
        inp[PARAM1_OFFSET] = int(parts[3])
        inp[PARAM2_OFFSET] = int(parts[4])
    elif command == CMD_VOLENVPER:
        inp[PARAM1_OFFSET] = int(parts[3])
        inp[PARAM2_OFFSET] = parse_bool(parts[4])
        inp[PARAM3_OFFSET] = int(parts[4])
    elif command == CMD_LSB:
        inp[PARAM1_OFFSET] = int(parts[3])
        inp[PARAM2_OFFSET] = 0
        inp[PARAM3_OFFSET] = 0
    elif command == CMD_MSB:
        inp[PARAM1_OFFSET] = int(parts[3])
        inp[PARAM2_OFFSET] = parse_bool(parts[4])
        inp[PARAM3_OFFSET] = parse_bool(parts[5])
    else:
        raise "this should not happen"
    return inp

def int32_as_bytes(ival):
    return np.frombuffer(ival.item().to_bytes(4, byteorder = 'big'), dtype=np.uint8)

def int32_of_bytes(np):
    return int.from_bytes(np, byteorder = 'big')

def int8_as_bytes(ival):
    return np.frombuffer(ival.item().to_bytes(1, byteorder='big'), dtype=np.uint8)

def int8_of_bytes(np):
    return int.from_bytes(np, byteorder = 'big')

def merge_params(data):
    command = data[CMD_OFFSET]
    if command == CMD_DUTYLL:
        return (data[PARAM1_OFFSET] << 6) | data[PARAM2_OFFSET]
    elif command == CMD_VOLENVPER:
        return (data[PARAM1_OFFSET] << 4) | (data[PARAM2_OFFSET] << 3) | data[PARAM3_OFFSET]
    elif command == CMD_LSB:
        return data[PARAM1_OFFSET]
    elif command == CMD_MSB:
        return data[PARAM1_OFFSET]  | (data[PARAM2_OFFSET] << 6) | (data[PARAM3_OFFSET] << 7)
    else:
        raise "this should not happen"

def unmerge_params(command, data,v ):
    if command == CMD_DUTYLL:
        data[PARAM1_OFFSET] = v >> 6;
        data[PARAM2_OFFSET] = v & 0b0011_1111
    elif command == CMD_VOLENVPER:
        data[PARAM1_OFFSET] = v >> 4
        data[PARAM2_OFFSET] = (v & 0b0000_1000) >> 3
        data[PARAM3_OFFSET] = (v & 0b0000_0111)
    elif command == CMD_LSB:
        data[PARAM1_OFFSET] = v
    elif command == CMD_MSB:
        data[PARAM1_OFFSET] = v & 0b0011_1111
        data[PARAM2_OFFSET] = (v & 0b0100_0000) >> 6
        data[PARAM3_OFFSET] = (v & 0b1000_0000) >> 7
    else:
        raise Exception("this should not happen")

def command_to_bytes(command):
    new_arr = np.concatenate([
                    int32_as_bytes(command[TIME_OFFSET]),
                    int8_as_bytes(command[CH_OFFSET]),
                    int8_as_bytes(command[CMD_OFFSET]),
                    int8_as_bytes(merge_params(command)),]).flatten()
    return new_arr

def command_of_bytes(byte_arr):
    d = fresh_input(0, 0, 0)
    d[TIME_OFFSET] = int32_of_bytes(byte_arr[0:4])
    d[CH_OFFSET] = int8_of_bytes(byte_arr[4:5])
    if d[CH_OFFSET] != 1 and d[CH_OFFSET] != 2:
        raise Exception("bad channel prediction")
    d[CMD_OFFSET] = int8_of_bytes(byte_arr[5:6])
    unmerge_params(d[CMD_OFFSET], d, byte_arr[6])
    return d

def print_feature(data, file=sys.stdout):
    command = data[CMD_OFFSET]
    if command == CMD_DUTYLL:
        print(f"CH {data[CH_OFFSET]} DUTYLL {data[PARAM1_OFFSET]} {data[PARAM2_OFFSET]} AT {data[TIME_OFFSET]}", file=file, flush=True)
    elif command == CMD_VOLENVPER:
        print(f"CH {data[CH_OFFSET]} VOLENVPER {data[PARAM1_OFFSET]} {data[PARAM2_OFFSET]} {data[PARAM3_OFFSET]} AT {data[TIME_OFFSET]}", file=file, flush=True)
    elif command == CMD_LSB:
        print(f"CH {data[CH_OFFSET]} FREQLSB {data[PARAM1_OFFSET]} AT {data[TIME_OFFSET]}", file=file, flush=True)
    elif command == CMD_MSB:
        print(f"CH {data[CH_OFFSET]} FREQMSB {data[PARAM1_OFFSET]} {data[PARAM2_OFFSET]} {data[PARAM3_OFFSET]} AT {data[TIME_OFFSET]}", file=file, flush=True)
    else:
        print(f"Bad prediction", file=file, flush=True)

def load_training_data(src):
    data = []
    file = open(src, 'r')
    for line in file:
        parts = line.split()
        if len(parts) > 0 and parts[0] == "CH":
            #print(parts)
            channel = int(parts[1])
            command = parts[2]
            time = int(parts[-1])
            if command == "DUTYLL":
                new_item = command_of_parts(CMD_DUTYLL, channel, parts, time)
            elif command == "VOLENVPER":
                new_item = command_of_parts(CMD_VOLENVPER, channel, parts, time)
            elif command == "FREQLSB":
                new_item = command_of_parts(CMD_LSB, channel, parts, time)
            elif command == "FREQMSB":
                new_item = command_of_parts(CMD_MSB, channel, parts, time)
            else:
                # Otherwise unknown data we ignore. We currently don't handle some commands (e.g, sweep) so erroring here is spurious
                print("Unknown", command)
            data.append(new_item)
    return data

@pescador.streamable
def samples_from_training_data(src, window_size, start_at_sample):

    # Scale the window size by the bytes per entry
    window_size = window_size * BYTES_PER_ENTRY

    sample_data = None
    try:
        sample_data = load_training_data(src)
    except Exception as e:
        LOGGER.error('Could not load {}: {}'.format(src, str(e)))
        raise StopIteration()
        
    sample_data = np.array([command_to_bytes(x) for x in sample_data]).flatten()

    while True:
        if len(sample_data) < window_size:
            sample = sample_data
        else:
            # Sample a random window from the audio file
            start_idx = np.random.randint(0, len(sample_data) - window_size)
            
            # If we should start on a sample boundary then round to the nearest multiple of sample boundary from the start
            if start_at_sample:
                start_idx = BYTES_PER_ENTRY * round(start_idx / BYTES_PER_ENTRY)

            sample = sample_data[start_idx:(start_idx + window_size)]

        yield sample

def create_batch_generator(paths, window_size, start_at_sample):
    streamers = []
    for path in paths:
        streamers.append(samples_from_training_data(path, window_size, start_at_sample))
    mux = pescador.StochasticMux(streamers, n_active=1, rate=1).iterate()
    return mux

def training_files(dirp):
    return [
      os.path.join(root, fname)
      for (root, dir_names, file_names) in os.walk(dirp, followlinks=True)
      for fname in file_names
    ]

def create_data_split(paths, window_size=MAX_WINDOW_SIZE, start_at_sample=True):
    train_gen = create_batch_generator(paths, window_size, start_at_sample)
    return train_gen

class SampleDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, window_size, start_at_sample=False):
        super(SampleDataset).__init__()
        files = training_files(path)

        print("Training files: ")
        for filename in files:
            print(filename)

        # Add one to window_size so that we have window size labels and inputs
        self.loader = create_data_split(files, window_size=MAX_WINDOW_SIZE + 1, start_at_sample=start_at_sample)

    def __iter__(self):
        while True:
             yield next(self.loader)
