use std::error::Error;
use std::sync::mpsc;
use std::time::Instant;

use rand::{rngs::ThreadRng, thread_rng, Rng};

use crate::machine::Machine;
use crate::ppu::{PpuStepState, GB_SCREEN_HEIGHT, GB_SCREEN_WIDTH};
use log::info;

fn reset_buttons(machine: &mut Machine) {
  let state = &mut machine.state.memory;
  state.a = false;
  state.b = false;
  state.start = false;
  state.select = false;
  state.up = false;
  state.down = false;
  state.left = false;
  state.right = false;
}

fn random_button(machine: &mut Machine, rng: &mut ThreadRng) {
  let state = &mut machine.state.memory;
  let rand = rng.gen_range(0..8);
  state.a = rand == 0;
  state.b = rand == 1;
  state.start = rand == 2;
  state.select = rand == 3;
  state.up = rand == 4;
  state.down = rand == 5;
  state.left = rand == 6;
  state.right = rand == 7;
}

pub fn run(mut gameboy_state: Machine) -> Result<(), Box<dyn Error>> {
  info!("preparing screen");

  // The pixel buffer is ignored (ideally it will be flagged off to avoid drawing costs)
  let mut pixel_buffer = vec![0; GB_SCREEN_WIDTH as usize * GB_SCREEN_HEIGHT as usize * 3];
  let (sound_tx, sound_rx) = mpsc::channel();

  let mut last_frameset = Instant::now();
  let mut frames = 0;
  let mut seconds = 0;
  let mut rng = thread_rng();

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
          seconds += 1;
          last_frameset = Instant::now();

          // Every 10 seconds for a second press some buttons
          if (seconds / 10) % 10 == 0 {
            print!("Pressing some random buttons");
            random_button(&mut gameboy_state, &mut rng);
          } else {
            print!("Doing nothing");
            reset_buttons(&mut gameboy_state);
          }
        }
      }
      _ => {}
    }
  }
}
