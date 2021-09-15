use crate::clock::Clock;
use crate::cpu::Cpu;
use crate::memory::GameboyState;
use crate::ppu::{Ppu, PpuStepState};

/// Encapsulate the entire running state of the Gameboy
pub struct Machine {
  pub cpu: Cpu,
  pub ppu: Ppu,
  pub clock: Clock,
  pub memory: GameboyState,
}

impl Machine {
  pub fn step(&mut self, screen_buffer: &mut [u8]) -> PpuStepState {
    self.cpu.step(&mut self.memory);
    self
      .clock
      .step(self.cpu.registers.last_clock as u8, &mut self.memory);
    self
      .ppu
      .step(&mut self.cpu, &mut self.memory, screen_buffer)
  }
}
