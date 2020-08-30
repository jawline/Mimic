use crate::cpu::{Registers, SmallWidthRegister, WideRegister};
use crate::memory::MemoryChunk;

use log::trace;

/// We re-use some instruction functions for multiple register implementations
/// This struct carries data for the single-implementation for many opcode instruction methods
pub struct InstructionData {
  pub code: u8,

  pub small_reg_one: SmallWidthRegister,
  pub small_reg_two: SmallWidthRegister,
  pub small_reg_dst: SmallWidthRegister,

  pub wide_reg_one: WideRegister,
  pub wide_reg_two: WideRegister,
  pub wide_reg_dst: WideRegister,
}

impl Default for InstructionData {
  fn default() -> InstructionData {
    InstructionData {
      code: 0,

      small_reg_one: SmallWidthRegister::B,
      small_reg_two: SmallWidthRegister::B,
      small_reg_dst: SmallWidthRegister::B,

      wide_reg_one: WideRegister::BC,
      wide_reg_two: WideRegister::BC,
      wide_reg_dst: WideRegister::BC,
    }
  }
}

impl InstructionData {
  pub fn wide_dst(r: WideRegister) -> InstructionData {
    let mut a = InstructionData::default();
    a.wide_reg_dst = r;
    a
  }
}

/// The instruction struct contains the implementation of and metadata on an instruction
pub struct Instruction {
  pub execute: fn(registers: &mut Registers, memory: &mut Box<dyn MemoryChunk>, additional: &InstructionData),
  pub timings: (u16, u16),
  pub text: String,
  pub data: InstructionData
}

/// No-op just increments the stack pointer
pub fn no_op(registers: &mut Registers, memory: &mut Box<dyn MemoryChunk>, additional: &InstructionData) {
  trace!("NoOp!");
  registers.write_r16(WideRegister::PC, registers.pc() + 1);
}

/// Load immediate loads a 16 bit value from memory and places it in a register
pub fn ld_imm_reg(registers: &mut Registers, memory: &mut Box<dyn MemoryChunk>, additional: &InstructionData) {
  let imm_val = memory.read_u16(registers.pc() + 1);
  registers.write_r16(additional.wide_reg_dst, imm_val);
  registers.write_r16(WideRegister::PC, registers.pc() + 1);
}

pub fn instruction_set() -> Vec<Instruction> {

  let no_op = Instruction {
    execute: crate::instruction::no_op,
    timings: (1, 4),
    text: "no_op".to_string(),
    data: InstructionData::default()
  };

  let load_imm_bc = Instruction {
    execute: crate::instruction::ld_imm_reg,
    timings: (3, 12),
    text: "ld_imm_bc".to_string(),
    data: InstructionData::wide_dst(WideRegister::BC)
  };

  vec![no_op, load_imm_bc]
}
