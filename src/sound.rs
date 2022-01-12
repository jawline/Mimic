use crate::cpu::{Cpu, STAT, VBLANK};
use crate::memory::{isset8, GameboyState};
use crate::util::{self, stat};
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::marker::PhantomData;
use cpal::Sample;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::result::Result;
use std::error::Error;

pub struct Sound<T: Sample>{
    _t: PhantomData<T>,
}

impl<T: Sample> Sound<T> {

    pub fn new<R: Sample>() -> Result<Sound<R>, Box<dyn Error>> {
      let host = cpal::default_host();
      let device = host.default_output_device().ok_or("no device found")?;
      let config = device.default_output_config().unwrap();
      Ok(match config.sample_format() {
          cpal::SampleFormat::F32 => Sound { _t: PhantomData },
          cpal::SampleFormat::I16 => Sound { _t: PhantomData },
          cpal::SampleFormat::U16 => Sound { _t: PhantomData }
      })
    }

    pub fn step(&mut self, cpu: &mut Cpu, mem: &mut GameboyState) {

        // We send samples from the emulator to the sound thread using channels
        let (sample_tx, sample_rx): (Sender<f32>, Receiver<f32>) = mpsc::channel();
    }
}
