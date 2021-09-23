use crate::cpu::{Cpu, TIMER};
use crate::memory::{
  isset8, GameboyState, DIV_REGISTER, MOD_REGISTER, TAC_REGISTER, TIMA_REGISTER,
};

pub struct Clock {
  ticks: usize,
  main: usize,
}

impl Clock {
  pub fn new() -> Self {
    Self { ticks: 0, main: 0 }
  }

  fn tac(&mut self, mem: &mut GameboyState) -> u8 {
    mem.read_u8(TAC_REGISTER)
  }

  fn update_div(&mut self, instruction_time: usize, mem: &mut GameboyState) {
    // TODO: Optimize the loop away
    self.ticks += instruction_time;
    while self.ticks >= 256 {
      self.ticks -= 256;
      mem.write_special_register(DIV_REGISTER, mem.read_u8(DIV_REGISTER).wrapping_add(1));
    }
  }

  fn update_tima(&mut self, tac: u8, mem: &mut GameboyState) {
    let threshold = match tac & 0x3 {
      0 => 1024,
      1 => 16,
      2 => 64,
      3 => 256,
      _ => panic!("should be impossible"),
    };

    while self.main >= threshold {
      self.main -= threshold;
      let tima = mem.read_u8(TIMA_REGISTER);

      let (new_tima, carried) = match tima.checked_add(1) {
        Some(n) => (n, false),
        None => (mem.read_u8(MOD_REGISTER), true),
      };

      mem.write_u8(TIMA_REGISTER, new_tima);

      if carried {
        Cpu::set_interrupt_happened(mem, TIMER);
      }
    }
  }

  pub fn step(&mut self, instruction_time: u8, mem: &mut GameboyState) {
    let instruction_time = usize::from(instruction_time);
    self.update_div(instruction_time, mem);

    let tac = self.tac(mem);

    if isset8(tac, 0x4) {
      self.main += instruction_time;
      self.update_tima(tac, mem);
    }
  }
}
