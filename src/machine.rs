use crate::clock::Clock;
use crate::cpu::Cpu;
use crate::instruction::InstructionSet;
use crate::memory::GameboyState;
use crate::ppu::{Ppu, PpuStepState};
use ciborium::{de, ser};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;

/// Encapsulate the entire running state of the Gameboy
#[derive(Serialize, Deserialize)]
pub struct MachineState {
  pub cpu: Cpu,
  pub ppu: Ppu,
  pub clock: Clock,
  pub memory: GameboyState,
}

pub struct Machine {
  pub state: MachineState,
  pub instruction_set: InstructionSet,
}

impl Machine {
  pub fn save_state(&self, filename: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(filename)?;
    ser::into_writer(&self.state, file)?;
    Ok(())
  }

  pub fn load_state(filename: &str) -> Result<Self, Box<dyn Error>> {
    let file = File::open(filename)?;
    let new_state: MachineState = de::from_reader(file)?;
    Ok(Self {
      state: new_state,
      instruction_set: InstructionSet::new(),
    })
  }

  pub fn step(&mut self, screen_buffer: &mut [u8]) -> PpuStepState {
    self
      .state
      .cpu
      .step(&mut self.state.memory, &self.instruction_set);
    self.state.clock.step(
      self.state.cpu.registers.last_clock as u8,
      &mut self.state.memory,
    );
    self
      .state
      .ppu
      .step(&mut self.state.cpu, &mut self.state.memory, screen_buffer)
  }
}
