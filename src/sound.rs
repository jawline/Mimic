use crate::cpu::{Cpu, STAT, VBLANK};
use crate::memory::{isset8, GameboyState};
use crate::util::{self, stat};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Sample;
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::marker::PhantomData;
use std::result::Result;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

const DUTY_FUNCTIONS: [[bool; 8]; 4] = [
    [false, false, false, false, false, false, false, true ],
    [true, false, false, false, false, false, false, true ],
    [true, false, false, false, false, true, true, true ],
    [false, true, true, true, true, true, true, false ],
];

#[derive(Serialize, Deserialize)]
pub struct Square {
    pub frequency_lsb: u16,
    pub frequency_msb: u16,
    pub volume_address: u16,
    pub duty_address: u16,
    pub wave_duty : u8,
    pub frequency_timer: u16,
    pub frequency_clock: u16,
}

impl Square {

    /// Read the desired frequency of the square wave from mem
    fn frequency(&self, mem: &mut GameboyState) -> u16 {
        // The 11 bit frequency is stored in the lowest three bits in frequency msb and the whole
        // of frequency lsb.
        let upper = ((mem.core_read(self.frequency_msb) & 0x7) as u16) << 8;

        let lower = mem.core_read(self.frequency_lsb) as u16;
        upper + lower
    }

    fn reload_wave(&self, mem: &mut GameboyState) -> u8 {
        mem.core_read(self.duty_address) >> 6
    }

    fn volume(&self, mem: &mut GameboyState) -> u8 {
        mem.core_read(self.volume_address) >> 4
    }

    fn step(&mut self, cpu: &mut Cpu, mem: &mut GameboyState) -> Option<f32> {

        if self.frequency_timer == 0 {
            let volume = self.volume(mem);

            let result = if DUTY_FUNCTIONS[self.wave_duty as usize][self.frequency_clock as usize] {
                -1. + ((volume as f32) / 8.)
            } else {
                -0.
            };

            // reload frequency timer
            self.frequency_timer = (2048 - self.frequency(mem)) * 4;
            self.frequency_clock = (self.frequency_clock + 1) % 8;
            self.wave_duty = self.reload_wave(mem);

            println!("Stepped");
            Some(result)
        } else {
            self.frequency_timer -= cpu.registers.last_clock;
            None
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
    channel_two: Square,
}

impl Sound {
  pub fn new() -> Self {
    Self {
        channel_two: Square {
          frequency_lsb: 0xFF18,
          frequency_msb: 0xFF19,
          volume_address: 0xFF17,
          duty_address: 0xFF16,
          wave_duty : 0,
          frequency_timer: 0,
          frequency_clock: 0,
        }
    }
  }

  // Step will process the gameboy state and generate new values for sample.
  pub fn step(&mut self, cpu: &mut Cpu, mem: &mut GameboyState, samples: &Sender<f32>) {
      match self.channel_two.step(cpu, mem) {
          Some(sample) => samples.send(sample).unwrap(),
          None => ()
      }
  }
}

fn run<T: cpal::Sample>(
  device: &cpal::Device,
  config: &cpal::StreamConfig,
  recv: Receiver<f32>,
) -> Result<cpal::Stream, Box<dyn Error>> {
  println!("Running");
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
      for frame in output.chunks_mut(channels) {
        while let Ok(sample) = recv.try_recv() {
            let sample = cpal::Sample::from::<f32>(&sample);
            for channel_fr in &mut *frame {
              *channel_fr = sample;
            }
            written += 1;
        }
      }

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
pub fn open_device() -> Result<(cpal::Device, cpal::Stream, Sender<f32>), Box<dyn Error>> {
  // pick the default cpal device and out system sample format as output
  let host = cpal::default_host();
  let device = host.default_output_device().ok_or("no device found")?;
  let config = device.default_output_config().unwrap();

  // We send samples from the emulator to the sound thread using channels
  let (sample_tx, sample_rx): (Sender<f32>, Receiver<f32>) = mpsc::channel();

  // Launch the audio thread
  let stream = match config.sample_format() {
    cpal::SampleFormat::F32 => run::<f32>(&device, &config.into(), sample_rx),
    cpal::SampleFormat::I16 => run::<i16>(&device, &config.into(), sample_rx),
    cpal::SampleFormat::U16 => run::<u16>(&device, &config.into(), sample_rx),
  }?;

  Ok((device, stream, sample_tx))
}
