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

  pub fn small_dst(r: SmallWidthRegister) -> InstructionData {
    let mut a = InstructionData::default();
    a.small_reg_dst = r;
    a
  }

  pub fn wide_dst(r: WideRegister) -> InstructionData {
    let mut a = InstructionData::default();
    a.wide_reg_dst = r;
    a
  }

  pub fn wide_src(r: WideRegister) -> InstructionData {
    let mut a = InstructionData::default();
    a.wide_reg_one = r;
    a
  }

  pub fn wide_dst_small_in(r: WideRegister, l: SmallWidthRegister) -> InstructionData {
    let mut a = InstructionData::wide_dst(r);
    a.small_reg_one = l;
    a
  }

  pub fn wide_dst_wide_src(r: WideRegister,
    l: WideRegister) -> InstructionData {
    let mut a = InstructionData::wide_dst(r);
    a.wide_reg_one = l;
    a
  }

  pub fn small_dst_wide_src(r: SmallWidthRegister,
    l: WideRegister) -> InstructionData {
    let mut a = InstructionData::small_dst(r);
    a.wide_reg_one = l;
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

/// Load immediate loads a 16 bit value following this instruction places it in a register
pub fn ld_imm_r16(registers: &mut Registers, memory: &mut Box<dyn MemoryChunk>, additional: &InstructionData) {
  //Load the 16 bit value after the opcode and store it to the dst register
  let imm_val = memory.read_u16(registers.pc() + 1);
  registers.write_r16(additional.wide_reg_dst, imm_val);

  //Increment the PC by three once finished
  registers.inc_pc(3);
}

/// Load immediate loads a 8 bit value following this instruction places it in a small register
pub fn ld_imm_r8(registers: &mut Registers, memory: &mut Box<dyn MemoryChunk>, additional: &InstructionData) {
  //Load the 8 bit value after the opcode and store it to the dst register
  let imm_val = memory.read_u8(registers.pc() + 1);
  registers.write_r8(additional.small_reg_dst, imm_val);

  //Increment the PC by three once finished
  registers.inc_pc(2);
}

/// Write the value of small register one to the address pointed to by wide_reg_dst
pub fn ld_reg8_mem_reg16(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  // Store the provided 8-bit register to the location pointed to by the 16-bit dst register
  let reg_val = registers.read_r8(additional.small_reg_one);
  let mem_dst = registers.read_r16(additional.wide_reg_dst);
  memory.write_u8(mem_dst, reg_val);

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Increment the value of a wide-register by one
pub fn inc_wide_register(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  // Increment the destination wide register by one
  registers.write_r16(
    additional.wide_reg_dst,
    registers.read_r16(additional.wide_reg_dst) + 1
  );

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Increment the value of a small register by one
pub fn inc_small_register(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  // Increment the destination register by one
  registers.write_r8(
    additional.small_reg_dst,
    registers.read_r8(additional.small_reg_dst) + 1
  );

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Rotate 8-bit register left, placing whatever is in bit 7 in the carry bit before
fn rotate_left_with_carry(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  let mut a = registers.read_r8(additional.small_reg_dst);
  let carry = a & (1 << 7) != 0;
  a = a.rotate_left(1);
  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(a == 0, false, false, carry);

  // Increment PC by one
  registers.inc_pc(1);
}

/// Rotate 8-bit register right, placing whatever is in bit 0 in the carry bit before
fn rotate_right_with_carry(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  let mut a = registers.read_r8(additional.small_reg_dst);
  let carry = a & (1) != 0;
  a = a.rotate_right(1);
  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(a == 0, false, false, carry);

  // Increment PC by one
  registers.inc_pc(1);
}

/// Rotate 8-bit register left using the carry bit as an additional bit.
/// In practice, store the current carry bit, set carry bit to value of bit 7
/// then shift left everything by one bit and then replace bit 0 with the origin
/// carry bit
fn rotate_r8_left_through_carry(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  let mut a = registers.read_r8(additional.small_reg_dst);
  let origin_carry = registers.carry();
  let carry = a & (1 << 7) != 0;

  if (origin_carry) {
    a |= 1;
  } else {
    a &= !1;
  }

  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(a == 0, false, false, carry);

  // Increment PC by one
  registers.inc_pc(1);
}

/// Rotate 8-bit register right using the carry bit as an additional bit.
/// In practice, store the current carry bit, set carry bit to value of bit 0
/// then shift right everything by one bit and then replace bit 7 with the origin
/// carry bit
fn rotate_r8_right_through_carry(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  let mut a = registers.read_r8(additional.small_reg_dst);
  let origin_carry = registers.carry();
  let carry = a & 1 != 0;

  if (origin_carry) {
    a |= (1 << 7);
  } else {
    a &= !(1 << 7);
  }

  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(a == 0, false, false, carry);

  // Increment PC by one
  registers.inc_pc(1);
}

/// Decrement the value of a small register by one
fn dec_wide_register(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  // Increment the destination register by one
  registers.write_r16(
    additional.wide_reg_dst,
    registers.read_r16(additional.wide_reg_dst) - 1
  );

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Decrement the value of a small register by one
fn dec_small_register(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  // Increment the destination register by one
  registers.write_r8(
    additional.small_reg_dst,
    registers.read_r8(additional.small_reg_dst) - 1
  );

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Write the contents of a wide register to an address in memory
fn load_immediate_wide_register(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  // Find the address we are writing to
  let load_address = memory.read_u16(registers.pc() + 1);

  // Write the contents of wide register one to that location 
  memory.write_u16(load_address, registers.read_r16(additional.wide_reg_one));

  // Increment the PC by one once finished
  registers.inc_pc(3);
}

/// Add a wide register to a wide register
fn add_r16_r16(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  let l = registers.read_r16(additional.wide_reg_dst);
  let r = registers.read_r16(additional.wide_reg_one);
  let result = l + r;

  registers.write_r16(
    additional.wide_reg_dst,
    result
  );

  // Did the 11th bits carry?
  let half_carried = (l & 0xFFF) + (r & 0xFFF) & (1 << 12);
  let carried = (l as u32 + r as u32) & (1 << 16);

  registers.set_flags(
    registers.zero(),
    false,
    half_carried != 0, 
    carried != 0
  );

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Load a value from memory to a small register
fn load_r16_mem_to_r8(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  let address = registers.read_r16(additional.wide_reg_one);
  let value = memory.read_u8(address);
  registers.write_r8(additional.small_reg_dst, value);

  registers.inc_pc(1);
}

/// Stop the processor & screen until button press
fn stop(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {
  registers.inc_pc(1);
  unimplemented!();
}

fn jump_relative_signed_immediate(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {
  let byte = memory.read_u8(registers.pc() + 1) as i8;
  registers.inc_pc(2);
  registers.jump_relative(byte);
}

pub fn instruction_set() -> Vec<Instruction> {

  let no_op = Instruction {
    execute: no_op,
    timings: (1, 4),
    text: "NOP".to_string(),
    data: InstructionData::default()
  };

  let load_imm_bc = Instruction {
    execute: ld_imm_r16,
    timings: (3, 12),
    text: "ld BC, nn".to_string(),
    data: InstructionData::wide_dst(WideRegister::BC)
  };

  let load_bc_a = Instruction {
    execute: ld_reg8_mem_reg16,
    timings: (1, 8),
    text: "ld (BC) A".to_string(),
    data: InstructionData::wide_dst_small_in(WideRegister::BC, SmallWidthRegister::A)
  };

  let inc_bc = Instruction {
    execute: inc_wide_register,
    timings: (1, 8),
    text: "inc BC".to_string(),
    data: InstructionData::wide_dst(WideRegister::BC),
  };

  let inc_b = Instruction {
    execute: inc_small_register,
    timings: (1, 4),
    text: format!("inc B"),
    data: InstructionData::small_dst(SmallWidthRegister::B),
  };

  let dec_b = Instruction {
    execute: dec_small_register,
    timings: (1, 4),
    text: "dec B".to_string(),
    data: InstructionData::small_dst(SmallWidthRegister::B),
  };

  let load_imm_b = Instruction {
    execute: ld_imm_r8,
    timings: (2, 8),
    text: "ld B, n".to_string(),
    data: InstructionData::small_dst(SmallWidthRegister::B)
  };

  let rlca = Instruction {
    execute: rotate_left_with_carry,
    timings: (1, 1),
    text: format!("RLCA"),
    data: InstructionData::small_dst(SmallWidthRegister::A)
  };

  let ld_nn_sp = Instruction {
    execute: load_immediate_wide_register,
    timings: (3, 20),
    text: format!("ld (NN), SP"),
    data: InstructionData::wide_src(WideRegister::SP)
  };

  let add_hl_bc = Instruction {
    execute: add_r16_r16,
    timings: (1, 8),
    text: format!("add HL, BC"),
    data: InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::BC)
  };

  let ld_a_bc = Instruction {
    execute: load_r16_mem_to_r8,
    timings: (1, 8),
    text: format!("ld A, (BC)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::BC)
  };

  let dec_bc = Instruction {
    execute: dec_wide_register,
    timings: (1, 8),
    text: format!("dec BC"),
    data: InstructionData::wide_dst(WideRegister::BC)
  };

  let inc_c = Instruction {
    execute: inc_small_register,
    timings: (1, 4),
    text: format!("inc C"),
    data: InstructionData::small_dst(SmallWidthRegister::C)
  };

  let dec_c = Instruction {
    execute: dec_small_register,
    timings: (1, 4),
    text: format!("dec C"),
    data: InstructionData::small_dst(SmallWidthRegister::C)
  };

  let ld_c_n = Instruction {
    execute: ld_imm_r8,
    timings: (2, 8),
    text: format!("ld C, n"),
    data: InstructionData::small_dst(SmallWidthRegister::C)
  };

  let rrca = Instruction {
    execute: rotate_right_with_carry,
    timings: (1, 4),
    text: format!("RRCA"),
    data: InstructionData::small_dst(SmallWidthRegister::A)
  };

  let stop = Instruction {
    execute: stop,
    timings: (1, 4),
    text: format!("STOP"),
    data: InstructionData::default()
  };

  let load_imm_de = Instruction {
    execute: ld_imm_r16,
    timings: (3, 12),
    text: format!("ld DE, nn"),
    data: InstructionData::wide_dst(WideRegister::DE)
  };

  let load_mem_de_a = Instruction {
    execute: ld_reg8_mem_reg16,
    timings: (1, 8),
    text: format!("ld (DE), A"),
    data: InstructionData::wide_dst_small_in(WideRegister::DE, SmallWidthRegister::A)
  };

  let inc_de = Instruction {
    execute: inc_wide_register,
    timings: (1, 8),
    text: "inc DE".to_string(),
    data: InstructionData::wide_dst(WideRegister::DE),
  };

  let inc_d = Instruction {
    execute: inc_small_register,
    timings: (1, 4),
    text: format!("inc D"),
    data: InstructionData::small_dst(SmallWidthRegister::D),
  };

  let dec_d = Instruction {
    execute: dec_small_register,
    timings: (1, 4),
    text: format!("dec D"),
    data: InstructionData::small_dst(SmallWidthRegister::D),
  };

  let ld_d_n = Instruction {
    execute: ld_imm_r8,
    timings: (2, 8),
    text: format!("ld D, n"),
    data: InstructionData::small_dst(SmallWidthRegister::D)
  };

  let rla = Instruction {
    execute:  rotate_r8_left_through_carry,
    timings: (1, 4),
    text: format!("RLA"),
    data: InstructionData::small_dst(SmallWidthRegister::A)
  };

  let jr_n = Instruction {
    execute: jump_relative_signed_immediate,
    timings: (2, 12),
    text: format!("JR n"),
    data: InstructionData::default()
  };

  let add_hl_de = Instruction {
    execute: add_r16_r16,
    timings: (1, 8),
    text: format!("add HL, DE"),
    data: InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::DE)
  };

  let ld_a_de = Instruction {
    execute: load_r16_mem_to_r8,
    timings: (1, 8),
    text: format!("ld A, (DE)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::DE)
  };

  let dec_de = Instruction {
    execute: dec_wide_register,
    timings: (1, 8),
    text: format!("dec DE"),
    data: InstructionData::wide_dst(WideRegister::DE)
  };

  let inc_e = Instruction {
    execute: inc_small_register,
    timings: (1, 4),
    text: format!("inc E"),
    data: InstructionData::small_dst(SmallWidthRegister::E),
  };

  let dec_e = Instruction {
    execute: dec_small_register,
    timings: (1, 4),
    text: format!("dec E"),
    data: InstructionData::small_dst(SmallWidthRegister::E),
  };

  let ld_e_n = Instruction {
    execute: ld_imm_r8,
    timings: (2, 8),
    text: format!("ld E, n"),
    data: InstructionData::small_dst(SmallWidthRegister::E)
  };

  vec![
    no_op, load_imm_bc, load_bc_a, inc_bc, inc_b, rlca, ld_nn_sp, add_hl_bc, ld_a_bc,
    dec_bc, inc_c, dec_c, ld_c_n, rrca, stop, load_imm_de, load_mem_de_a, inc_de,
    dec_d, ld_d_n, rla, jr_n, add_hl_de, ld_a_de, dec_de, inc_e, dec_e, ld_e_n
  ]
}
