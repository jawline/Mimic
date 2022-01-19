use crate::cpu::Cpu;
use crate::memory::{isset8, GameboyState};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::result::Result;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

// TODO: Refactor all the sound code to take a multiple of T-cycles rather than stepping by
// four cycles every call.

const GAMEBOY_FREQUENCY: usize = 4194304;
const FRAME_SEQUENCER_CLOCK_IN_T_CYCLES: usize = 8192;
const WAVE_PATTERN_RAM: u16 = 0xFF30;
const CHANNEL_ONE_SWEEP_ADDRESS: u16 = 0xFF10;

/**
 * Constan for the lower six bits of a byte for bitwise logic.
 */

const LOWER_SIX: u8 = 0b00111111;
/**
 * Constant for the upper two bits of a byte for bitwise logic.
 */
const UPPER_TWO: u8 = 0b11000000;

/**
 * The most significant bit in a byte.
 */
const UPPER_BIT: u8 = 0b10000000;

/**
 * The lower nibble of a byte
 */
const LOWER_NIBBLE: u8 = 0b00001111;

/**
 * The number of bits to shift to extract the upper nibble from a byte as a value.
 */
const UPPER_NIBBLE_SHIFT: usize = 4;

/**
 * The gameboy square wave generators use one of four pre-programmed duty functions to produce
 * their wave.
 * These are hard coded here and referenced by the step function in Square.
 */
const DUTY_FUNCTIONS: [[bool; 8]; 4] = [
  [false, false, false, false, true, false, false, false],
  [false, false, false, false, true, true, false, false],
  [false, false, false, false, true, true, true, true],
  [false, true, true, true, true, true, true, false],
];

/**
 * This method returns the sweep period, negate, and shift for channel one.
 */
fn sweep_parameters(mem: &mut GameboyState) -> SweepState {
  let sweep_data = mem.core_read(CHANNEL_ONE_SWEEP_ADDRESS);
  let period = (sweep_data & 0b01110000) >> 4;
  let negate = isset8(sweep_data, 0b00001000);
  let shift = sweep_data & 0b00000111;
  SweepState {
    period,
    negate,
    shift,
  }
}

/**
 * Decrements whatever the current sweep period is by one.
 */
fn dec_sweep_period(mem: &mut GameboyState) {
  let existing = mem.core_read(CHANNEL_ONE_SWEEP_ADDRESS);
  let sweep = (existing & 0b0111_0000) >> 4;
  let sweep = sweep - 1;
  mem.core_write(
    CHANNEL_ONE_SWEEP_ADDRESS,
    ((sweep << 4) & 0b01110000) | (existing & 0b1000_1111),
  );
}

#[derive(Serialize, Deserialize)]
struct Wave {
  dac_address: u16,
  frequency_lsb: u16,
  frequency_msb: u16,
  volume_address: u16,
  length_address: u16,
  frequency_timer: u16,
  frequency_clock: u16,
  amplitude: f32,
  volume: f32,
  enabled: bool,
}

impl Wave {
  /**
   * Returns true if the channel has been triggered and should update it's shadow state. This
   * function also resets the triggered value back to zero.
   */
  fn triggered(&self, mem: &mut GameboyState) -> bool {
    let trigger = mem.core_read(self.frequency_msb);
    mem.core_write(self.frequency_msb, trigger & 0b0111_1111);
    isset8(trigger, 0b1000_0000)
  }

  fn dac_power(&self, mem: &mut GameboyState) -> bool {
    isset8(mem.core_read(self.dac_address), UPPER_BIT)
  }

  fn frequency(&self, mem: &mut GameboyState) -> u16 {
    // The 11 bit frequency is stored in the lowest three bits in frequency msb and the whole
    // of frequency lsb.
    let upper = ((mem.core_read(self.frequency_msb) & 0b00000111) as u16) << 8;

    let lower = mem.core_read(self.frequency_lsb) as u16;
    upper + lower
  }

  fn length(&self, mem: &GameboyState) -> u8 {
    mem.core_read(self.length_address)
  }

  /**
   * Preserve the first two bits of the second channel register
   * and write the remaining six bits as new_val
   */
  fn set_length(&self, mem: &mut GameboyState, new_val: u8) {
    mem.core_write(self.length_address, new_val)
  }

  /**
   * Returns true if the length enable bit is set.
   */
  fn length_enabled(&self, mem: &mut GameboyState) -> bool {
    const LENGTH_ENABLE_BIT: u8 = 0b01000000;
    isset8(mem.core_read(self.frequency_msb), LENGTH_ENABLE_BIT)
  }

  fn amplitude(&self, mem: &mut GameboyState) -> f32 {
    if self.enabled {
      if self.length_enabled(mem) {
        if self.length(mem) > 0 {
          self.amplitude
        } else {
          0.0
        }
      } else {
        self.amplitude
      }
    } else {
      0.0
    }
  }

  /**
   * The wave generator has a single 2-bit volume register that scale a sound to 0%, 100%, 50% and
   * 25% for values of 0, 1, 2, 3 respectively
   */
  fn reload_volume(&mut self, mem: &mut GameboyState) {
    const WAVE_VOLUME_BITS: u8 = 0b01100000;
    const WAVE_VOLUME_SHIFT: usize = 5;
    self.volume = match (mem.core_read(self.volume_address) & WAVE_VOLUME_BITS) >> WAVE_VOLUME_SHIFT
    {
      0 => 0.0,
      1 => 1.0,
      2 => 0.5,
      3 => 0.25,
      _ => panic!("this should not be possible because we have sanitized the byte"),
    };
  }

  /**
   * Get the amplitude (4 bit value) of the current sample of the wave function.
   */
  fn sample(&self, mem: &mut GameboyState) -> u8 {
    // 4 bits per sample, higher bits contain the first element in each byte.
    let sample = mem.core_read(WAVE_PATTERN_RAM + (self.frequency_clock / 2));
    if self.frequency_clock % 2 == 0 {
      sample >> UPPER_NIBBLE_SHIFT
    } else {
      sample & LOWER_NIBBLE
    }
  }

  /**
   * Gameboy wave sample generators step through 32 4-bit samples in Wave RAM. Once per frequency
   * tick they emit the volume of the next sample as sound.
   *
   * We expect this function to be called once per 4 cycles by Sound, not just once per instruction.
   */
  fn step(&mut self, mem: &mut GameboyState) {
    if !self.dac_power(mem) {
      self.enabled = false;
    }

    if self.enabled {
      if self.frequency_timer == 0 {
        // There are 32 4-bit entries in the wave tale
        self.frequency_clock = (self.frequency_clock + 1) % 32;

        self.amplitude = ((self.sample(mem) as f32) / 16.) * self.volume;

        // reload frequency timer
        self.frequency_timer = (2048 - self.frequency(mem)) * 4;
      } else {
        self.frequency_timer -= 4;
      }
    } else {
      self.amplitude = 0.;
    }
  }
}

/**
 * This stores the value of the sweep register for the first channel.
 */
#[derive(Serialize, Deserialize)]
struct SweepState {
  period: u8,
  negate: bool,
  shift: u8,
}

#[derive(Serialize, Deserialize)]
struct Square {
  frequency_lsb: u16,
  frequency_msb: u16,
  volume_address: u16,
  duty_address: u16,
  wave_duty: u8,
  frequency_timer: u16,
  frequency_clock: u16,
  amplitude: f32,
  enabled: bool,
}

enum AddMode {
  Add,
  Subtract,
}

impl Square {
  /**
   * Returns true if the channel has been triggered and should update it's shadow state. This
   * function also resets the triggered value back to zero.
   */
  fn triggered(&self, mem: &mut GameboyState) -> bool {
    let trigger = mem.core_read(self.frequency_msb);
    mem.core_write(self.frequency_msb, trigger & 0b0111_1111);
    isset8(trigger, 0b1000_0000)
  }

  /// Read the desired frequency of the square wave from mem
  fn frequency(&self, mem: &mut GameboyState) -> u16 {
    // The 11 bit frequency is stored in the lowest three bits in frequency msb and the whole
    // of frequency lsb.
    let upper = ((mem.core_read(self.frequency_msb) & 0x7) as u16) << 8;

    let lower = mem.core_read(self.frequency_lsb) as u16;
    upper + lower
  }

  /**
   * Set the channel frequency by writing to the lsb and msb locations in
   * memory.
   */
  fn set_frequency(&self, mem: &mut GameboyState, val: u16) {
    mem.core_write(self.frequency_lsb, val as u8);
    let existing_msb = mem.core_read(self.frequency_msb);
    let new_msb = (existing_msb & 0b1111_1000) | ((val >> 8) as u8);
    mem.core_write(self.frequency_msb, new_msb);
  }

  /**
   * Returns the add mode of the envelope function for the square wave (if the envelope should add
   * or subtract the volume when the frame sequencer steps it).
   */
  fn envelope_add_mode(&self, mem: &GameboyState) -> AddMode {
    const ADD_MODE_BIT: u8 = 0b00001000;

    match isset8(mem.core_read(self.volume_address), ADD_MODE_BIT) {
      true => AddMode::Add,
      false => AddMode::Subtract,
    }
  }

  /**
   * Get the number of frame sequencer tickers the
   * envelope function should continue stepping for.
   */
  fn envelope_period(&self, mem: &GameboyState) -> u8 {
    mem.core_read(self.volume_address) & 0b00000111
  }

  /**
   * Set the number of frame sequencer tickers the
   * envelope function should continue stepping for.
   */
  fn set_envelope_period(&self, mem: &mut GameboyState, val: u8) {
    let current = mem.core_read(self.volume_address) & 0b11111000;
    mem.core_write(self.volume_address, current | (val & 0b00000111));
  }

  /**
   * The wave function this sound should play (index into the 4
   * wave duty functions preprogrammed into the hardware) is
   * stored in the final 2 bits of the sound sound register.
   */
  fn duty(&self, mem: &GameboyState) -> u8 {
    mem.core_read(self.duty_address) >> 6
  }

  /**
   * The length of the sound (the duration the channel
   * should keep playing for) is stored in the first 6
   * bits of the second sound register for the square
   * channel.
   */
  fn length(&self, mem: &GameboyState) -> u8 {
    mem.core_read(self.duty_address) & LOWER_SIX
  }

  /**
   * Preserve the first two bits of the second channel register
   * and write the remaining six bits as new_val
   */
  fn set_length(&self, mem: &mut GameboyState, new_val: u8) {
    mem.core_write(
      self.duty_address,
      (mem.core_read(self.duty_address) & UPPER_TWO) | (new_val & LOWER_SIX),
    )
  }

  /**
   * Returns true if the length enable bit is set.
   */
  fn length_enabled(&self, mem: &mut GameboyState) -> bool {
    const LENGTH_ENABLE_BIT: u8 = 0b01000000;
    isset8(mem.core_read(self.frequency_msb), LENGTH_ENABLE_BIT)
  }

  fn volume(&self, mem: &mut GameboyState) -> u8 {
    mem.core_read(self.volume_address) >> 4
  }

  fn set_volume(&self, mem: &mut GameboyState, val: u8) {
    let current = mem.core_read(self.volume_address) & 0b00001111;
    mem.core_write(self.volume_address, current | (val << 4));
  }

  fn amplitude(&self, mem: &mut GameboyState) -> f32 {
    if self.length_enabled(mem) {
      if self.length(mem) > 0 {
        self.amplitude
      } else {
        0.0
      }
    } else {
      self.amplitude
    }
  }

  /// Reload the frequency period
  fn reload_frequency(&mut self, mem: &mut GameboyState) {
    self.frequency_timer = (2048 - self.frequency(mem)) * 4;
  }

  /**
   * Process a trigger event, setting the wave generator to enabled and reloading registers.
   */
  fn trigger(&mut self, mem: &mut GameboyState) {
    self.enabled = true;
    self.set_length(mem, 64);
    self.reload_frequency(mem);
  }

  /**
   * Gameboy square wave generators produce output through 4 duty cycles.
   *
   * We expect this function to be called once per 4 cycles by Sound, not just once per
   * instruction.
   */
  fn step(&mut self, mem: &mut GameboyState) {
    if self.enabled {
      if self.frequency_timer == 0 {
        let volume = self.volume(mem);

        self.amplitude = if DUTY_FUNCTIONS[self.duty(mem) as usize][self.frequency_clock as usize] {
          (volume as f32) / 15.
        } else {
          -0.
        };

        self.reload_frequency(mem);
        self.frequency_clock = (self.frequency_clock + 1) % 8;
      } else {
        self.frequency_timer -= 4;
      }
    } else {
      self.amplitude = 0.;
    }
  }
}

/**
 * We wrap all of the channels in a structure so they can be passed around more easily.
 */
#[derive(Serialize, Deserialize)]
struct Channels {
  channel_one: Square,
  channel_two: Square,
  wave: Wave,
}

impl Channels {
  /**
   * Step all channels by 4 T-cycles.
   */
  fn step(&mut self, mem: &mut GameboyState) {
    self.channel_one.step(mem);
    self.channel_two.step(mem);
    //self.wave.step(mem);
  }

  /**
   * Get the mixed amplitude of all channels now.
   */
  fn amplitude(&self, mem: &mut GameboyState) -> f32 {
    self.channel_one.amplitude(mem) + self.channel_two.amplitude(mem) + self.wave.amplitude(mem)
  }
}

/**
 * The frame sequencer clocks the length, envelope, and sweep components of the channels.
 * The frame sequencer has a 512Hz clock.
 *
 * The length of each channel is decremented every 2 frame sequencer cycles.
 */
#[derive(Serialize, Deserialize)]
struct FrameSequencer {
  pub cycles: usize,
  pub frame_sequencer_cycles: usize,

  /**
   * True if sweep is currently enabled.
   */
  pub sweep: bool,

  /**
   * The last frequency seen for a channel when sweep was triggered.
   */
  pub shadow_frequency: u16,
}

impl FrameSequencer {
  fn new() -> Self {
    FrameSequencer {
      cycles: 0,
      frame_sequencer_cycles: 0,
      shadow_frequency: 0,
      sweep: false,
    }
  }

  /**
   * Returns true on every even clock (every 2 frame sequencer cycles, or 16384 T-cycles).
   */
  fn frame_sequencer_clock_channel_lengths(&self) -> bool {
    self.frame_sequencer_cycles % 2 == 0
  }

  /**
   * Returns true if the frame sequencer should decrement envelopes and modify their volumes.
   */
  fn clock_envelopes(&self) -> bool {
    self.frame_sequencer_cycles % 8 == 0
  }

  /**
   * Returns true if the frame sequencer should process channel one sweep this cycle.
   */
  fn clock_sweep(&self) -> bool {
    self.frame_sequencer_cycles % 4 == 0
  }

  /**
   * Calculate the next frequency after a sweep.
   */
  fn next_sweep(&self, mem: &mut GameboyState) -> Option<u16> {
    let sweep = sweep_parameters(mem);
    if sweep.period > 0 {
      let new_frequency = self.shadow_frequency >> sweep.shift;
      let new_frequency = if sweep.negate {
        !new_frequency
      } else {
        new_frequency
      };
      Some(new_frequency)
    } else {
      None
    }
  }

  fn do_sweep_clock(&mut self, mem: &mut GameboyState, square: &mut Square) {
    if self.sweep {
      match self.next_sweep(mem) {
        Some(new_frequency) => {
          dec_sweep_period(mem);
          if new_frequency >= 2048 {
            square.enabled = false;
          } else {
            square.set_frequency(mem, new_frequency);
            self.shadow_frequency = new_frequency;
          }
        }
        None => {}
      }
    }
  }

  fn square_dec_envelope(mem: &mut GameboyState, square: &mut Square) {
    let envelope_period = square.envelope_period(mem);
    if envelope_period > 0 {
      let current_volume = square.volume(mem);
      let mode = square.envelope_add_mode(mem);

      let new_volume = match mode {
        AddMode::Add => {
          if current_volume < 15 {
            current_volume + 1
          } else {
            current_volume
          }
        }
        AddMode::Subtract => {
          if current_volume > 0 {
            current_volume - 1
          } else {
            current_volume
          }
        }
      };

      square.set_volume(mem, new_volume);
      square.set_envelope_period(mem, envelope_period - 1);
    }
  }

  fn dec_length_square(mem: &mut GameboyState, square: &mut Square) {
    let enabled = square.length_enabled(mem);
    let current_length = square.length(mem);
    if enabled && current_length > 0 {
      square.set_length(mem, current_length - 1);
    }
  }

  fn do_cycle(&mut self, mem: &mut GameboyState, channels: &mut Channels) {
    if self.frame_sequencer_clock_channel_lengths() {
      // Channel lengths
      FrameSequencer::dec_length_square(mem, &mut channels.channel_one);
      FrameSequencer::dec_length_square(mem, &mut channels.channel_two);
    }
    if self.clock_sweep() {
      self.do_sweep_clock(mem, &mut channels.channel_one);
    }
    if self.clock_envelopes() {
      FrameSequencer::square_dec_envelope(mem, &mut channels.channel_one);
      FrameSequencer::square_dec_envelope(mem, &mut channels.channel_two);
    }
  }

  fn step(&mut self, mem: &mut GameboyState, channels: &mut Channels) {
    self.cycles += 4;
    if self.cycles >= FRAME_SEQUENCER_CLOCK_IN_T_CYCLES {
      self.frame_sequencer_cycles += 1;
      self.do_cycle(mem, channels);
    }
  }
}

/**
 * The Sound struct carries the current gameboy state of the sound system but not references to the
 * device so that it can be serialized.
 *
 * When stepped, this will add samples (if any are playing) to a channel that is being read by an
 * audio thread (constructed by open_device in this file). The device is recreated when loading a
 * save state so any unplayed samples will be lost.
 */
#[derive(Serialize, Deserialize)]
pub struct Sound {
  channels: Channels,
  sequencer: FrameSequencer,
  t_cycles: usize,
}

impl Sound {
  pub fn new() -> Self {
    Self {
      channels: Channels {
        channel_one: Square {
          frequency_lsb: 0xFF13,
          frequency_msb: 0xFF14,
          volume_address: 0xFF12,
          duty_address: 0xFF11,
          wave_duty: 0,
          frequency_timer: 0,
          frequency_clock: 0,
          amplitude: 0.,
          enabled: false,
        },
        channel_two: Square {
          frequency_lsb: 0xFF18,
          frequency_msb: 0xFF19,
          volume_address: 0xFF17,
          duty_address: 0xFF16,
          wave_duty: 0,
          frequency_timer: 0,
          frequency_clock: 0,
          amplitude: 0.,
          enabled: false,
        },
        wave: Wave {
          dac_address: 0xFF1A,
          amplitude: 0.,
          frequency_lsb: 0xFF1D,
          frequency_msb: 0xFF1E,
          volume_address: 0xFF1C,
          length_address: 0xFF1B,
          frequency_timer: 0,
          frequency_clock: 0,
          volume: 0.,
          enabled: false,
        },
      },
      sequencer: FrameSequencer::new(),
      t_cycles: 0,
    }
  }

  // Step will process the gameboy state and generate new values for sample.
  pub fn step(
    &mut self,
    cpu: &mut Cpu,
    mem: &mut GameboyState,
    sample_rate: usize,
    samples: &Sender<f32>,
  ) {
    // TODO: We handle trigger logic here but it should live in its own place.
    if self.channels.channel_one.triggered(mem) {
      self.channels.channel_one.trigger(mem);
      let params = sweep_parameters(mem);
      if params.period != 0 || params.shift != 0 {
        println!("Enabled sweep");
        self.sequencer.sweep = true;
        self.sequencer.shadow_frequency = self.channels.channel_one.frequency(mem);
      }
    }

    if self.channels.channel_two.triggered(mem) {
      self.channels.channel_two.trigger(mem);
    }

    if self.channels.wave.triggered(mem) {
      println!("Wave enabled");
      self.channels.wave.enabled = true;
      self.channels.wave.set_length(mem, 255);
      self.channels.wave.frequency_clock = 0;
      self.channels.wave.reload_volume(mem);
    }

    // Now process the frame
    let sample_divisor = (GAMEBOY_FREQUENCY / sample_rate) + 1;
    for _ in (0..cpu.registers.last_clock).step_by(4) {
      self.channels.step(mem);
      self.sequencer.step(mem, &mut self.channels);
      self.t_cycles += 4;
      if self.t_cycles >= sample_divisor {
        // We should generate a sample for cpal here
        samples.send(self.channels.amplitude(mem));
        self.t_cycles = 0;
      }
    }
  }
}

/**
 * Run launches an audio thread and begins sending samples to it.
 */
fn run<T: cpal::Sample + std::fmt::Debug>(
  device: &cpal::Device,
  config: &cpal::StreamConfig,
  recv: Receiver<f32>,
) -> Result<cpal::Stream, Box<dyn Error>> {
  let channels = config.channels as usize;

  let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

  // The writer writes any samples written to the channel to the output audio stream. It runs in
  // a thread and is driven by cpal.
  let stream = device.build_output_stream(
    config,
    move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
      let mut written = 0;

      /* TODO: Might need to write an empty frame if we haven't caught up, should test first
       * though.  */

      /*for frame in output.chunks_mut(channels) {
          for channel_fr in &mut *frame {
              *channel_fr = 0.;
          }
      }*/

      for frame in output.chunks_mut(channels) {
        if let Ok(sample) = recv.try_recv() {
          let sample = cpal::Sample::from::<f32>(&sample);
          for channel_fr in &mut *frame {
            *channel_fr = sample;
          }
          written += 1;
        }
      }

      //println!("{}", written);

      if written == 0 {
        for frame in output.chunks_mut(channels) {
          for channel_fr in frame {
            *channel_fr = cpal::Sample::from::<f32>(&0.0);
          }
        }
      }
    },
    err_fn,
  )?;

  stream.play()?;

  println!("Starting to play");

  Ok(stream)
}

/**
 * This function opens a cpal sound output device and returns a channel through which samples can
 * be sent. If the emulator cannot catch up then middle samples (no sound) will be played.
 */
pub fn open_device() -> Result<(cpal::Device, cpal::Stream, usize, Sender<f32>), Box<dyn Error>> {
  // pick the default cpal device and out system sample format as output
  let host = cpal::default_host();
  let device = host.default_output_device().ok_or("no device found")?;
  let config = device.default_output_config().unwrap();

  let sample_rate = config.sample_rate().0 as usize;

  // We send samples from the emulator to the sound thread using channels
  let (sample_tx, sample_rx): (Sender<f32>, Receiver<f32>) = mpsc::channel();

  // Launch the audio thread
  let stream = match config.sample_format() {
    cpal::SampleFormat::F32 => run::<f32>(&device, &config.into(), sample_rx),
    cpal::SampleFormat::I16 => run::<i16>(&device, &config.into(), sample_rx),
    cpal::SampleFormat::U16 => run::<u16>(&device, &config.into(), sample_rx),
  }?;

  Ok((device, stream, sample_rate, sample_tx))
}
