use crate::cpu::CPU;
use crate::memory::MemoryChunk;

/// Encapsulate the entire running state of the Gameboy
pub struct Machine {
  pub cpu: CPU,
  pub memory: Box<dyn MemoryChunk>
}

impl Machine {
  pub fn step(&mut self) {
    self.cpu.step(&mut self.memory);
  }
}
