use std::error::Error;
use std::sync::mpsc;
use std::time::Instant;

use crate::cpu::{Cpu, JOYPAD};
use crate::machine::Machine;
use crate::ppu::{PpuStepState, GB_SCREEN_HEIGHT, GB_SCREEN_WIDTH};
use log::info;

pub fn run(mut gameboy_state: Machine) -> Result<(), Box<dyn Error>> {
  info!("preparing screen");

  // The pixel buffer is ignored (ideally it will be flagged off to avoid drawing costs)
  let mut pixel_buffer = vec![0; GB_SCREEN_WIDTH as usize * GB_SCREEN_HEIGHT as usize * 3];
  let (sound_tx, sound_rx) = mpsc::channel();

  let mut last_frameset = Instant::now();
  let mut frames = 0;

  loop {
    let state = gameboy_state.step(&mut pixel_buffer, 1_000_000_000, &sound_tx);

    // Clear the sound RX (ideally sound will be switched off)
    while let Ok(_) = sound_rx.try_recv() {}

    match state {
      PpuStepState::VBlank => {
        frames = frames + 1;
        if last_frameset.elapsed().as_secs_f64() > 1. {
          println!("Framerate: {}", frames);
          frames = 0;
          last_frameset = Instant::now();
        }
      }
      _ => {}
    }
  }
}
