use crate::cpu::{Cpu, STAT, VBLANK};
use crate::memory::{isset8, GameboyState};
use crate::util::{self, stat};
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::marker::PhantomData;

pub struct Sound<T>{
    _t: PhantomData<T>,
}

impl Sound<T> {

    pub fn new() -> Self<T> {
      let host = cpal::default_host();
      let device = host.default_output_device().ok_or("no device found")?;
      let config = device.default_output_config().unwrap();
      match config.sample_format() {
          cpal::SampleFormat::F32 => Sound { _t: Phanton::new() },
          cpal::SampleFormat::I16 => Sound { _t: Phantom::new() },
          cpal::SampleFormat::U16 => Sound { _t: Phantom::new() }
      }
    }
}

impl <T> Sound<T> {
    pub fn step(&mut self, cpu: &mut Cpu, mem: &mut GameboyState) {

        // We send samples from the emulator to the sound thread using channels
        let (sample_tx, sample_rx): (Sender<f32>, Receiver<f32>) = mpsc::channel();
    }
}
