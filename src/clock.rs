use crate::cpu::Registers;
use crate::cpu::{Cpu, TIMER};
use crate::instruction_clock::InstructionClock;
use crate::memory::{isset16, isset8, GameboyState, MOD_REGISTER, TAC_REGISTER, TIMA_REGISTER};
use log::trace;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Clock {
  div_clock: InstructionClock,
}

impl Clock {
  pub fn new() -> Self {
    Self {
      div_clock: InstructionClock::new(256),
    }
  }

  fn tac(&mut self, mem: &mut GameboyState) -> u8 {
    mem.core_read(TAC_REGISTER)
  }

  fn update_div(&mut self, instruction_time: u16, mem: &mut GameboyState) -> u16 {
    let result = mem.div.wrapping_add(instruction_time);
    let carries = result ^ mem.div ^ instruction_time;
    trace!(
      "DIV: {} (origin {}, incremented by {}, carries={:16b})",
      result,
      mem.div,
      instruction_time,
      carries
    );
    mem.div = result;
    carries
  }

  fn increment_tima(&mut self, mem: &mut GameboyState, registers: &Registers) {
    let tima = mem.core_read(TIMA_REGISTER);

    let (new_tima, carried) = match tima.checked_add(1) {
      Some(n) => (n, false),
      None => (mem.core_read(MOD_REGISTER), true),
    };

    trace!("NEW TIMA: {}", new_tima);
    mem.write_special_register(TIMA_REGISTER, new_tima);

    if carried {
      Cpu::set_interrupt_happened(mem, TIMER, registers);
    }
  }

  pub fn step(
    &mut self,
    mut total_instruction_time: u8,
    mem: &mut GameboyState,
    registers: &Registers,
  ) {
    // If we wrote to the DIV register on the last instruction then stepping is incorrect because
    // the stepping would increment prior to the write.
    if !registers.wrote_div_on_last_instruction {
      // We split this into 4 cycle instructions so that the carry is correctly respected from div
      // changes
      while total_instruction_time > 0 {
        total_instruction_time -= 4;
        let div_carries = self.update_div(4, mem);

        let tac = self.tac(mem);

        if isset8(tac, 0x4) {
          let tima_should_increment = match tac & 3 {
            0 => isset16(div_carries, 1 << 10),
            1 => isset16(div_carries, 1 << 4),
            2 => isset16(div_carries, 1 << 6),
            3 => isset16(div_carries, 1 << 8),
            _ => panic!("tac & 3 should not ever be > 3"),
          };
          trace!(
            "TAC: {} carries {:b} triggered {}",
            tac & 3,
            div_carries,
            tima_should_increment,
          );
          if (!registers.wrote_tima_on_last_instruction) & tima_should_increment {
            self.increment_tima(mem, registers);
          }
        }
      }
    }
  }
}
