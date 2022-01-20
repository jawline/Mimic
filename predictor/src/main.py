import numpy as np
import pescador
import logging
import os

LOGGER = logging.getLogger('gbsd')
LOGGER.setLevel(logging.DEBUG)

NOP_CMD = 0
DUTY_LL_CMD = 1
VOLENVPER_CMD = 2
FREQLSB_CMD = 3
FREQMSB_CMD = 4

TIME_OFFSET = 0
COMMAND_OFFSET = 1
CHANNEL_OFFSET = 2
DUTY_OFFSET = 3
LENGTH_OFFSET = 4
VOL_OFFSET = 5
ADD_MODE_OFFSET = 6
PERIOD_OFFSET = 7
FREQLSB_OFFSET = 8
FREQMSB_OFFSET = 9
LENGTH_ENABLE_OFFSET = 10
TRIGGER_OFFSET = 11
NUM_INP = 12

WINDOW_SIZE = 32

def fresh_input(command, channel, time):
    return np.array([time, command, channel, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def nop():
    return fresh_input(NOP_CMD, 0, 0)

def duty_ll(channel, parts, time):
    inp = fresh_input(DUTY_LL_CMD, channel, time)
    inp[DUTY_OFFSET] = float(parts[3])
    inp[LENGTH_OFFSET] = float(parts[4])
    return inp

def volenvper(channel, parts, time):
    inp = fresh_input(VOLENVPER_CMD, channel, time)
    inp[VOL_OFFSET] = float(parts[3])
    inp[ADD_MODE_OFFSET] = float(parts[4])
    inp[PERIOD_OFFSET] = float(parts[4])
    return inp

def freqlsb(channel, parts, time):
    inp = fresh_input(FREQLSB_CMD, channel, time)
    inp[FREQLSB_OFFSET] = float(parts[3])
    return inp

def freqmsb(channel, parts, time):
    inp = fresh_input(FREQMSB_CMD, channel, time)
    inp[FREQMSB_OFFSET] = float(parts[3])
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
        yield sample

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

gen = create_batch_generator(training_files("./training_data/"), 8096)

for batch in gen:
    for item in batch:
        print(batch)
