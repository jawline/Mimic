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

  let a = registers.read_r8(additional.small_reg_dst);
  registers.set_carry(a & (1 << 7) != 0);
  a.rotate_left(1);
  registers.write_r8(additional.small_reg_dst, a);

  // Increment PC by one
  registers.inc_pc(1);
}

/// Rotate 8-bit register right, placing whatever is in bit 0 in the carry bit before
fn rotate_right_with_carry(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  let a = registers.read_r8(additional.small_reg_dst);
  registers.set_carry(a & (1) != 0);
  a.rotate_right(1);
  registers.write_r8(additional.small_reg_dst, a);

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

  registers.write_r16(
    additional.wide_reg_dst,
    registers.read_r16(additional.wide_reg_dst) + registers.read_r16(additional.wide_reg_one)
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
    text: "inc B".to_string(),
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
    data: InstructionData::small_dst(SmallWidthRegister::A) //Doesn't use
  };

  let load_imm_de = Instruction {
    execute: ld_imm_r16,
    timings: (3, 12),
    text: format!("ld DE, nn"),
    data: InstructionData::wide_dst(WideRegister::DE)
  };

  vec![
    no_op, load_imm_bc, load_bc_a, inc_bc, inc_b, rlca, ld_nn_sp, add_hl_bc, ld_a_bc,
    dec_bc, inc_c, dec_c, ld_c_n, rrca, stop, load_imm_de,
  ]
}
