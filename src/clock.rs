use crate::cpu::{CPU, TIMER};
use crate::memory::{isset8, MemoryPtr};

const DIV_REGISTER: u16 = 0xFF04;
const TIMA_REGISTER: u16 = 0xFF05;
const MOD_REGISTER: u16 = 0xFF06;
const TAC_REGISTER: u16 = 0xFF07;

pub struct CLOCK {
  div: u8,
  main: u8,
  sub: u8,
}

impl CLOCK {
  pub fn new() -> CLOCK {
    CLOCK {
      div: 0,
      main: 0,
      sub: 0,
    }
  }

  pub fn step(&mut self, instruction_time: u8, mem: &mut MemoryPtr) {
    self.sub += instruction_time;
    if self.sub >= 4 {
      self.main += 1;
      self.sub -= 4;
      self.div += 1;
      if self.div == 16 {
        mem.write_u8(DIV_REGISTER, mem.read_u8(DIV_REGISTER) + 1);
      }
    }

    let tac = mem.read_u8(TAC_REGISTER);

    if isset8(tac, 0x4) {
      let threshold = match tac & 0x3 {
        0 => 64,
        1 => 1,
        2 => 4,
        3 => 16,
        _ => panic!("should be impossible"),
      };

      if self.main > threshold {
        self.main = 0;
        let tima = mem.read_u8(TIMA_REGISTER);
        let new_tima = tima + 1;

        // Overflow, fire the TIMER interrupt
        if new_tima < tima {
          mem.write_u8(TIMA_REGISTER, mem.read_u8(MOD_REGISTER));
          CPU::set_interrupt_happened(mem, TIMER);
        } else {
          mem.write_u8(TIMA_REGISTER, new_tima);
        }
      }
    }
  }
}
