use crate::clock::Clock;
use crate::cpu::Cpu;
use crate::instruction::InstructionSet;
use crate::memory::GameboyState;
use crate::ppu::{Ppu, PpuStepState};
use crate::sound::Sound;
use ciborium::{de, ser};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::sync::mpsc::Sender;

/// Encapsulate the entire running state of the Gameboy
#[derive(Serialize, Deserialize)]
pub struct MachineState {
  pub cpu: Cpu,
  pub ppu: Ppu,
  pub clock: Clock,
  pub sound: Sound,
  pub memory: GameboyState,
}

pub struct Machine {
  pub state: MachineState,
  pub instruction_set: InstructionSet,
  pub disable_sound: bool,
  pub disable_framebuffer: bool,
}

impl Machine {
  pub fn save_state(&self, filename: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(filename)?;
    ser::into_writer(&self.state, file)?;
    Ok(())
  }

  pub fn load_state(
    filename: &str,
    disable_sound: bool,
    disable_framebuffer: bool,
  ) -> Result<Self, Box<dyn Error>> {
    let file = File::open(filename)?;
    let new_state: MachineState = de::from_reader(file)?;
    Ok(Self {
      state: new_state,
      instruction_set: InstructionSet::new(),
      disable_sound,
      disable_framebuffer,
    })
  }

  pub fn step(
    &mut self,
    screen_buffer: &mut [u8],
    sample_rate: usize,
    samples: &Sender<f32>,
  ) -> PpuStepState {
    self
      .state
      .cpu
      .step(&mut self.state.memory, &self.instruction_set);
    self.state.cpu.registers.total_clock +=
      self.state.cpu.registers.cycles_elapsed_during_last_step as usize;

    self.state.clock.step(
      self.state.cpu.registers.cycles_elapsed_during_last_step as u8,
      &mut self.state.memory,
      &self.state.cpu.registers,
    );

    self.state.sound.step(
      &mut self.state.cpu,
      &mut self.state.memory,
      sample_rate,
      samples,
      self.disable_sound,
    );
    self.state.ppu.step(
      &mut self.state.cpu,
      &mut self.state.memory,
      screen_buffer,
      self.disable_framebuffer,
    )
  }
}
