import numpy as np
import pescador
import logging
import os

LOGGER = logging.getLogger('gbsd')
LOGGER.setLevel(logging.DEBUG)

CMD_VOLENVPER = 0
CMD_DUTYLL = 1
CMD_MSB = 2
CMD_LSB = 3
CMD_COUNT = 4

def onehot_cmd(data):
    cmd = data[CMD_OFFSET]
    nd = [ 0, 0, 0, 0 ]
    nd[int(cmd)] = 1
    return nd

TIME_OFFSET = 0
CH_1_OFFSET = 1
CH_2_OFFSET = 2
CMD_OFFSET = 3
CHANNEL_OFFSET = 4
DUTY_OFFSET = 5
LENGTH_OFFSET = 6
VOL_OFFSET = 7
ADD_MODE_OFFSET = 8
PERIOD_OFFSET = 9
FREQLSB_OFFSET = 10
FREQMSB_OFFSET = 11
LENGTH_ENABLE_OFFSET = 12
TRIGGER_OFFSET = 13
NUM_INP = 14

WINDOW_SIZE = 1024

NORMALIZE_TIME_BY = float(4194304 * 3) # 1 second is 4194304 cycles so this is 10s

def norm(val, max_val):
    return ((val / max_val) * 2.) - 1.

def unnorm(val, max_val):
    return ((val + 1.) / 2.) * max_val

def fresh_input(command, channel, time):
    newd = np.zeros(shape=NUM_INP, dtype=float)
    newd[TIME_OFFSET] = norm(time, NORMALIZE_TIME_BY)
    #print(channel)
    if int(channel) == 1:
        newd[CH_1_OFFSET] = 1.
    elif int(channel) == 2:
        newd[CH_2_OFFSET] = 1.
    else:
        raise "I didn't expect this"
    newd[CMD_OFFSET] = channel
    return newd

def nop():
    return fresh_input(NOP_CMD_OFFSET, 1, 0)

def duty_ll(channel, parts, time):
    inp = fresh_input(CMD_DUTYLL, channel, time)
    inp[DUTY_OFFSET] = norm(float(parts[3]), 2.)
    inp[LENGTH_OFFSET] = norm(float(parts[4]), 64.)
    return inp

def volenvper(channel, parts, time):
    inp = fresh_input(CMD_VOLENVPER, channel, time)
    inp[VOL_OFFSET] = float(parts[3]) / 16.
    inp[ADD_MODE_OFFSET] = float(parts[4])
    inp[PERIOD_OFFSET] = float(parts[4]) / 7.
    return inp

def freqlsb(channel, parts, time):
    inp = fresh_input(CMD_LSB, channel, time)
    inp[FREQLSB_OFFSET] = norm(float(parts[3]), 255.)
    return inp

def freqmsb(channel, parts, time):
    inp = fresh_input(CMD_MSB, channel, time)
    inp[FREQMSB_OFFSET] = norm(float(parts[3]), 7.)
    inp[LENGTH_ENABLE_OFFSET] = float(bool(parts[4]))
    inp[TRIGGER_OFFSET] = float(bool(parts[5]))
    return inp

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
               data.append(duty_ll(channel, parts, time))
           if command == "VOLENVPER":
               data.append(volenvper(channel, parts, time))
           if command == "FREQLSB":
               data.append(freqlsb(channel, parts, time))
           if command == "FREQMSB":
               data.append(freqmsb(channel, parts, time))
           #print("NEXTCMD", data[-1])
    return data

@pescador.streamable
def samples_from_training_data(src, window_size=WINDOW_SIZE):
    sample_data = None

    try:
        sample_data = load_training_data(src)
    except Exception as e:
        LOGGER.error('Could not load {}: {}'.format(src, str(e)))
        raise StopIteration()

    true_window_size = window_size + 1

    # Pad small samples with nop
    while len(sample_data) < true_window_size:
        sample_data.append(nop())

    while True:

        if len(sample_data) == true_window_size:
            sample = sample_data
        else:
            # Sample a random window from the audio file
            start_idx = np.random.randint(0, len(sample_data) - true_window_size)
            end_idx = start_idx + true_window_size
            sample = sample_data[start_idx:end_idx]

        sample_input = sample[0:window_size]
        sample_output = sample[window_size:window_size+1]

        sample_input = np.array(sample_input).astype(np.float32)
        sample_output = np.array(sample_output).astype(np.float32)

        yield { 'X':sample_input, 'Y': sample_output }

def create_batch_generator(paths, batch_size):
    streamers = []
    for path in paths:
        print("Creating a batch generator")
        streamers.append(samples_from_training_data(path))
        print("Done creating batch generator")
    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)
    return batch_gen

def training_files(dirp):
    return [
      os.path.join(root, fname)
      for (root, dir_names, file_names) in os.walk(dirp, followlinks=True)
      for fname in file_names
    ]

def create_data_split(paths, batch_size):
    train_gen = create_batch_generator(paths, batch_size)
    return train_gen

