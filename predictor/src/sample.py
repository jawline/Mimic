import numpy as np
import pescador
import logging
import os

LOGGER = logging.getLogger('gbsd')
LOGGER.setLevel(logging.DEBUG)

TIME_OFFSET = 0
NOP_COMMAND_OFFSET = 1
DUTY_LL_CMD_OFFSET = 2
VOLENVPER_CMD_OFFSET = 3
FREQLSB_CMD_OFFSET = 4
FREQMSB_CMD_OFFSET= 5
CHANNEL_OFFSET = 6
DUTY_OFFSET = 7
LENGTH_OFFSET = 8
VOL_OFFSET = 9
ADD_MODE_OFFSET = 10
PERIOD_OFFSET = 11
FREQLSB_OFFSET = 12
FREQMSB_OFFSET = 13
LENGTH_ENABLE_OFFSET = 14
TRIGGER_OFFSET = 15
NUM_INP = 16

WINDOW_SIZE = 32

def fresh_input(command, channel, time):
    newd = np.array([time, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    newd[command] = 1.
    return newd

def nop():
    return fresh_input(NOP_CMD_OFFSET, 0, 0)

def duty_ll(channel, parts, time):
    inp = fresh_input(DUTY_LL_CMD_OFFSET, channel, time)
    inp[DUTY_OFFSET] = float(parts[3]) / 2.
    inp[LENGTH_OFFSET] = float(parts[4]) / 64.
    return inp

def volenvper(channel, parts, time):
    inp = fresh_input(VOLENVPER_CMD_OFFSET, channel, time)
    inp[VOL_OFFSET] = float(parts[3]) / 16.
    inp[ADD_MODE_OFFSET] = float(parts[4])
    inp[PERIOD_OFFSET] = float(parts[4]) / 7.
    return inp

def freqlsb(channel, parts, time):
    inp = fresh_input(FREQLSB_CMD_OFFSET, channel, time)
    inp[FREQLSB_OFFSET] = float(parts[3]) / 255.
    return inp

def freqmsb(channel, parts, time):
    inp = fresh_input(FREQMSB_CMD_OFFSET, channel, time)
    inp[FREQMSB_OFFSET] = float(parts[3]) / 7.
    inp[LENGTH_ENABLE_OFFSET] = float(bool(parts[4]))
    inp[TRIGGER_OFFSET] = float(bool(parts[5]))
    return inp

def load_training_data(src):
    data = []
    file = open(src, 'r')
    for line in file:
        parts = line.split()
        if len(parts) > 0 and parts[0] == "CH":
           channel = int(parts[1])
           command = parts[2]
           time = int(parts[-1])
           if command == "DUTYLL":
               data.append(duty_ll(channel, parts, time))
           if command == "VOLENVPER":
               data.append(volenvper(channel, parts, time))
           if command == "FREQLSB":
               data.append(freqlsb(channel, parts, time))
           if command == "FREQMSB":
               data.append(freqmsb(channel, parts, time))
    return data

def samples_from_training_data(src, window_size=WINDOW_SIZE):
    sample_data = None

    try:
        sample_data = load_training_data(src)
    except Exception as e:
        LOGGER.error('Could not load {}: {}'.format(src, str(e)))
        raise StopIteration()

    # Pad small samples with nop
    while len(sample_data) < window_size:
        sample_data.append(nop())

    while True:
        if len(sample_data) == window_size:
            sample = audio_data
        else:
            # Sample a random window from the audio file
            start_idx = np.random.randint(0, len(sample_data) - window_size)
            end_idx = start_idx + window_size
            sample = sample_data[start_idx:end_idx]
        sample = np.array(sample).flatten().astype(np.float32)
        yield { 'X':sample }

def create_batch_generator(paths, batch_size):
    streamers = []
    for path in paths:
        s = pescador.Streamer(samples_from_training_data, path)
        streamers.append(s)
    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)
    return batch_gen

def training_files(dirp):
    return [
      os.path.join(root, fname)
      for (root, dir_names, file_names) in os.walk(dirp, followlinks=True)
      for fname in file_names
    ]

def create_data_split(paths, valid_ratio, test_ratio,
                      train_batch_size, valid_size, test_size):
    num_files = len(paths)
    num_valid = int(np.ceil(num_files * valid_ratio))
    num_test = int(np.ceil(num_files * test_ratio))
    num_train = num_files - num_valid - num_test

    assert num_valid > 0
    assert num_test > 0
    assert num_train > 0

    valid_files = paths[:num_valid]
    test_files = paths[num_valid:num_valid + num_test]
    train_files = paths[num_valid + num_test:]

    train_gen = create_batch_generator(train_files, train_batch_size)
    valid_data = next(iter(create_batch_generator(valid_files, valid_size)))
    test_data = next(iter(create_batch_generator(test_files, test_size)))

    return train_gen, valid_data, test_data

