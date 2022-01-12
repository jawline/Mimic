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

#[derive(Serialize, Deserialize)]
pub struct Sound<T: Sample> {
  _t: PhantomData<T>,
}

impl<T: Sample> Sound<T> {
  pub fn new<R: Sample>() -> Result<Sound<R>, Box<dyn Error>> {
    // pick the default cpal device and out system sample format as output
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no device found")?;
    let config = device.default_output_config().unwrap();
    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    // We send samples from the emulator to the sound thread using channels
    let (sample_tx, sample_rx): (Sender<f32>, Receiver<f32>) = mpsc::channel();

    // We construct the Sound<T> before starting the stream to avoid offending the borrow checker.
    let sound = match config.sample_format() {
      cpal::SampleFormat::F32 => Sound { _t: PhantomData },
      cpal::SampleFormat::I16 => Sound { _t: PhantomData },
      cpal::SampleFormat::U16 => Sound { _t: PhantomData },
    };

    let channels = config.channels() as usize;

    // The writer writes any samples written to the channel to the output audio stream. It runs in
    // a thread and is driven by cpal.
    let stream = device.build_output_stream(
      &config.into(),
      move |output: &mut [R], _: &cpal::OutputCallbackInfo| {
        let mut written = 0;

        /* TODO: Might need to write an empty frame if we haven't caught up, should test first
         * though.  */
        for frame in output.chunks_mut(channels) {
          match sample_rx.try_recv() {
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
      },
      err_fn,
    )?;

    Ok(sound)
  }

  // Step will process the gameboy state and generate new values for sample.
  pub fn step(&mut self, cpu: &mut Cpu, mem: &mut GameboyState) {}
}
