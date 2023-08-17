use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct InstructionClock {
  cycles_to_next_clock: usize,
  cycles_per_clock: usize,
}

impl InstructionClock {
  pub fn new(cycles_per_clock: usize) -> Self {
    Self {
      cycles_to_next_clock: cycles_per_clock,
      cycles_per_clock,
    }
  }

  pub fn step(&mut self, total_instruction_time: usize) -> u16 {
    if total_instruction_time >= self.cycles_to_next_clock {
      self.cycles_to_next_clock += self.cycles_per_clock;
      self.cycles_to_next_clock -= total_instruction_time;
      1
    } else {
      self.cycles_to_next_clock -= total_instruction_time;
      0
    }
  }
}
