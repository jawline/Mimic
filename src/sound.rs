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

/**
 * The Sound struct carries the current gameboy state of the sound system but not references to the
 * device so that it can be serialized.
 */
#[derive(Serialize, Deserialize)]
pub struct Sound {}

impl Sound {
  pub fn new() -> Self {
    Self {}
  }

  // Step will process the gameboy state and generate new values for sample.
  pub fn step(&mut self, cpu: &mut Cpu, mem: &mut GameboyState) {}
}

fn run<T: cpal::Sample>(
  device: &cpal::Device,
  config: &cpal::StreamConfig,
  recv: Receiver<f32>,
) -> Result<(), Box<dyn Error>> {
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
        match recv.try_recv() {
          Ok(sample) => {
            let sample = cpal::Sample::from::<f32>(&sample);
            for channel_fr in frame {
              *channel_fr = sample;
            }

            written += 1;
          }
          Err(_) => break,
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

  Ok(())
}

/**
 * This function opens a cpal sound output device and returns a channel through which samples can
 * be sent. If the emulator cannot catch up then middle samples (no sound) will be played.
 */
pub fn open_device() -> Result<Sender<f32>, Box<dyn Error>> {
  // pick the default cpal device and out system sample format as output
  let host = cpal::default_host();
  let device = host.default_output_device().ok_or("no device found")?;
  let config = device.default_output_config().unwrap();

  // We send samples from the emulator to the sound thread using channels
  let (sample_tx, sample_rx): (Sender<f32>, Receiver<f32>) = mpsc::channel();

  // Launch the audio thread
  match config.sample_format() {
    cpal::SampleFormat::F32 => run::<f32>(&device, &config.into(), sample_rx),
    cpal::SampleFormat::I16 => run::<i16>(&device, &config.into(), sample_rx),
    cpal::SampleFormat::U16 => run::<u16>(&device, &config.into(), sample_rx),
  }?;

  Ok(sample_tx)
}
