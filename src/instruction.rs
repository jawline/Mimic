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

  pub fn wide_dst_small_in(r: WideRegister, l: SmallWidthRegister) -> InstructionData {
    let mut a = InstructionData::wide_dst(r);
    a.small_reg_one = l;
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
pub fn ld_imm_reg16(registers: &mut Registers, memory: &mut Box<dyn MemoryChunk>, additional: &InstructionData) {
  //Load the 16 bit value after the opcode and store it to the dst register
  let imm_val = memory.read_u16(registers.pc() + 1);
  registers.write_r16(additional.wide_reg_dst, imm_val);

  //Increment the PC by three once finished
  registers.write_r16(WideRegister::PC, registers.pc() + 3);
}

/// Write the value of small register one to the address pointed to by wide_reg_dst
pub fn ld_reg8_mem_reg16(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  // Store the provided 8-bit register to the location pointed to by the 16-bit dst register
  let reg_val = registers.read_r8(additional.small_reg_one);
  let mem_dst = registers.read_r16(additional.wide_reg_dst);
  memory.write_u16(mem_dst, reg_val);

  // Increment the PC by three once finished
  registers.write_r16(WideRegister::PC, registers.pc() + 1);
}

/// Increment the value of a wide-register by one
pub fn inc_wr(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  // Increment the destination wide register by one
  registers.write_r16(
    additional.wide_reg_dst,
    registers.read_r16(additional.wide_reg_dst) + 1
  );

  // Increment the PC by one once finished
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
    execute: crate::instruction::ld_imm_reg16,
    timings: (3, 12),
    text: "ld_imm_bc".to_string(),
    data: InstructionData::wide_dst(WideRegister::BC)
  };

  let load_bc_a = Instruction {
    execute: crate::instruction::ld_reg8_mem_reg16,
    timings: (1, 8),
    text: "ld_(bc)_a".to_string(),
    data: InstructionData::wide_dst_small_in(WideRegister::BC, SmallWidthRegister::A)
  };

  let inc_bc = Instruction {
    execute: crate::instruction::inc_wr,
    timings: (1, 8),
    text: "inc_bc".to_string(),
    data: InstructionData::wide_dst(WideRegister::BC),
  };

  vec![no_op, load_imm_bc, load_bc_a, inc_bc]
}
