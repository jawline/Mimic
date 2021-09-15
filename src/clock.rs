use crate::cpu::{Cpu, TIMER};
use crate::memory::{isset8, MemoryPtr};

const DIV_REGISTER: u16 = 0xFF04;
const TIMA_REGISTER: u16 = 0xFF05;
const MOD_REGISTER: u16 = 0xFF06;
const TAC_REGISTER: u16 = 0xFF07;

pub struct Clock {
  div: usize,
  main: usize,
}

impl Clock {
  pub fn new() -> Self {
    Self { div: 0, main: 0 }
  }

  fn tac(&mut self, mem: &mut MemoryPtr) -> u8 {
    mem.read_u8(TAC_REGISTER)
  }

  fn update_div(&mut self, instruction_time: usize, mem: &mut MemoryPtr) {
    self.div += instruction_time;
    while self.div >= 64 {
      mem.write_u8(DIV_REGISTER, mem.read_u8(DIV_REGISTER).wrapping_add(1));
      self.div -= 64;
    }
  }

  fn update_tima(&mut self, tac: u8, instruction_time: usize, mem: &mut MemoryPtr) {
    self.main += instruction_time;

    let threshold = match tac & 0x3 {
      0 => 256,
      1 => 4,
      2 => 16,
      3 => 64,
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

  pub fn step(&mut self, instruction_time: u8, mem: &mut MemoryPtr) {
    let instruction_time = usize::from(instruction_time);
    self.update_div(instruction_time, mem);

    let tac = self.tac(mem);

    if isset8(tac, 0x4) {
      self.update_tima(tac, instruction_time, mem);
    }
  }
}
