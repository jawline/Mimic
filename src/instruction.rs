use crate::cpu::{Registers, SmallWidthRegister, WideRegister, ZERO_FLAG, CARRY_FLAG};
use crate::memory::MemoryChunk;

use log::trace;

/// We re-use some instruction functions for multiple register implementations
/// This struct carries data for the single-implementation for many opcode instruction methods
#[derive(Clone)]
pub struct InstructionData {
  pub code: u8,
  pub flag_mask: u8,
  pub flag_expected: u8,

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

      flag_mask: 0,
      flag_expected: 0,

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

  pub fn with_flag(&self, mask: u8, expected: u8) -> InstructionData {
    let mut m = self.clone();
    m.flag_mask = mask;
    m.flag_expected = expected;
    return m;
  }

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

/// Loads an immediate 8-bit value into the address pointed to by the wide dest register
pub fn ld_mem_r16_immediate(registers: &mut Registers, memory: &mut Box<dyn MemoryChunk>, additional: &InstructionData) {
  let val = memory.read_u8(registers.pc() + 1);
  registers.inc_pc(2);
  let addr = registers.read_r16(additional.wide_reg_dst);
  memory.write_u8(addr, val);
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

  registers.inc_pc(1);

  let l = registers.read_r8(additional.small_reg_dst);
  let result = l + 1;

  // Increment the destination register by one
  registers.write_r8(
    additional.small_reg_dst,
    result 
  );

  registers.set_flags(
    result == 0,
    false,
    ((l & 0xF) + 1) & (0x10) != 0,
    registers.carry()
  );
}

pub fn inc_mem_r16(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  registers.inc_pc(1);
  let addr = registers.read_r16(additional.wide_reg_dst);
  let l = memory.read_u8(addr);
  let result = l + 1;

  // Increment by one and modify memory
  memory.write_u8(addr, result);

  registers.set_flags(
    result == 0,
    false,
    ((l & 0xF) + 1) & (0x10) != 0,
    registers.carry()
  );
}

/// Decrement the value of memory pointed by a wide register by one
/// and write it back to the same location in memory
pub fn dec_mem_r16(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {

  registers.inc_pc(1);
  let addr = registers.read_r16(additional.wide_reg_dst);
  let l = memory.read_u8(addr);
  let result = l - 1;

  // Increment by one and modify memory
  memory.write_u8(addr, result);

  registers.set_flags(
    result == 0,
    false,
    ((l & 0xF0) - 1) & (0x0F) != 0,
    registers.carry()
  );
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

  if origin_carry {
    a |= 1 << 7;
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

  let l = registers.read_r8(additional.small_reg_dst);
  let result = l - 1;

  // Increment the destination register by one
  registers.write_r8(
    additional.small_reg_dst,
    result 
  );

  registers.set_flags(
    result == 0,
    false,
    ((l & 0xF0) - 1) & (0x0F) != 0,
    registers.carry()
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

/// Place the value of a small register into the
/// memory address pointed to by the wide destination
/// register and then increment the wide destination register
/// Example, ld HL, 0 ld A, 5 ldi (HL), A will leave [0] = 5, A = 5, HL = 1
fn ldi_mem_r16_val_r8(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {
  registers.inc_pc(1);
  let wide_reg = registers.read_r16(additional.wide_reg_dst);
  let target = registers.read_r8(additional.small_reg_one);
  memory.write_u8(wide_reg, target);
  registers.write_r16(additional.wide_reg_dst, wide_reg + 1);
}

/// Place memory pointed to by the wide register into the small dst register
/// then increment the wide register
fn ldi_r8_mem_r16(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {
  registers.inc_pc(1);
  let wide_reg = registers.read_r16(additional.wide_reg_one);
  let mem = memory.read_u8(wide_reg);
  let target = registers.write_r8(additional.small_reg_dst, mem);
  registers.write_r16(additional.wide_reg_dst, wide_reg + 1);
}

/// Like ldi but decrement
fn ldd_mem_r16_val_r8(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {
  registers.inc_pc(1);
  let wide_reg = registers.read_r16(additional.wide_reg_dst);
  let target = registers.read_r8(additional.small_reg_one);
  memory.write_u8(wide_reg, target);
  registers.write_r16(additional.wide_reg_dst, wide_reg - 1);
}

/// Like ldi but decrement
fn ldd_r8_mem_r16(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {
  registers.inc_pc(1);
  let wide_reg = registers.read_r16(additional.wide_reg_one);
  let mem = memory.read_u8(wide_reg);
  let target = registers.write_r8(additional.small_reg_dst, mem);
  registers.write_r16(additional.wide_reg_dst, wide_reg - 1);
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

/// Jump relative by a signed 8-bit value following the opcode
fn jump_relative_signed_immediate(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {
  let byte = memory.read_u8(registers.pc() + 1) as i8;
  registers.inc_pc(2);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected { 
    registers.jump_relative(byte);
  }
}

/// DAA takes the result of an arithmetic operation and makes it binary coded
/// retrospectively
fn daa(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {
  // https://forums.nesdev.com/viewtopic.php?t=15944
  unimplemented!();
}

/// Flip all bits in an r8
fn cpl_r8(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {
  registers.inc_pc(1);
  registers.write_r8(
    additional.small_reg_dst,
    !registers.read_r8(additional.small_reg_dst)
  );
  registers.set_flags(registers.zero(), true, true, registers.carry());  
}

/// Sets the carry flag, resets negative and half carry flags, zero unaffected
fn scf(registers: &mut Registers,
  memory: &mut Box<dyn MemoryChunk>,
  additional: &InstructionData) {
  registers.inc_pc(1);
  registers.set_flags(registers.zero(), false, false, true);
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
    data: InstructionData::default().with_flag(0, 0)
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

  let rra = Instruction {
    execute:  rotate_r8_left_through_carry,
    timings: (1, 4),
    text: format!("RRA"),
    data: InstructionData::small_dst(SmallWidthRegister::A)
  };

  let jr_nz_n = Instruction {
    execute: jump_relative_signed_immediate,
    timings: (2, 8),
    text: format!("JRNZ n"),
    data: InstructionData::default().with_flag(ZERO_FLAG, 0)
  };

  let load_imm_hl = Instruction {
    execute: ld_imm_r16,
    timings: (3, 12),
    text: format!("ld HL, nn"),
    data: InstructionData::wide_dst(WideRegister::HL)
  };

  let ldi_hl_a = Instruction {
    execute: ldi_mem_r16_val_r8,
    timings: (1, 8),
    text: format!("ldi (HL), A"),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::A)
  };

  let inc_hl = Instruction {
    execute: inc_wide_register,
    timings: (1, 8),
    text: format!("inc HL"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let inc_h = Instruction {
    execute: inc_small_register,
    timings: (1, 4),
    text: format!("inc H"),
    data: InstructionData::small_dst(SmallWidthRegister::H),
  };

  let dec_h = Instruction {
    execute: dec_small_register,
    timings: (1, 4),
    text: format!("dec H"),
    data: InstructionData::small_dst(SmallWidthRegister::H),
  };

  let ld_h_n = Instruction {
    execute: ld_imm_r8,
    timings: (2, 8),
    text: format!("ld H, n"),
    data: InstructionData::small_dst(SmallWidthRegister::H)
  };

  let daa = Instruction {
    execute: daa,
    timings: (1, 4),
    text: format!("daa"),
    data: InstructionData::small_dst(SmallWidthRegister::A)
  };

  let jr_z_n = Instruction {
    execute: jump_relative_signed_immediate,
    timings: (2, 8),
    text: format!("JRZ n"),
    data: InstructionData::default().with_flag(ZERO_FLAG, ZERO_FLAG)
  };

  let add_hl_hl = Instruction {
    execute: add_r16_r16,
    timings: (1, 8),
    text: format!("add HL, HL"),
    data: InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::HL)
  };

  let ldi_a_hl = Instruction {
    execute: ldi_mem_r16_val_r8,
    timings: (1, 8),
    text: format!("ldi A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  };

  let dec_hl = Instruction {
    execute: dec_wide_register,
    timings: (1, 8),
    text: format!("dec HL"),
    data: InstructionData::wide_dst(WideRegister::HL)
  };

  let inc_l = Instruction {
    execute: inc_small_register,
    timings: (1, 4),
    text: format!("inc L"),
    data: InstructionData::small_dst(SmallWidthRegister::L),
  };

  let dec_l = Instruction {
    execute: dec_small_register,
    timings: (1, 4),
    text: format!("dec L"),
    data: InstructionData::small_dst(SmallWidthRegister::L),
  };

  let ld_l_n = Instruction {
    execute: ld_imm_r8,
    timings: (2, 8),
    text: format!("ld L, n"),
    data: InstructionData::small_dst(SmallWidthRegister::L)
  };

  let cpl = Instruction {
    execute: ld_imm_r8,
    timings: (1, 4),
    text: format!("cpl"),
    data: InstructionData::small_dst(SmallWidthRegister::A)
  };

  let jr_nc_n = Instruction {
    execute: jump_relative_signed_immediate,
    timings: (2, 8),
    text: format!("JRNC n"),
    data: InstructionData::default().with_flag(CARRY_FLAG, 0)
  };

  let load_imm_sp = Instruction {
    execute: ld_imm_r16,
    timings: (3, 12),
    text: format!("ld SP, nn"),
    data: InstructionData::wide_dst(WideRegister::SP)
  };

  let ldd_hl_a = Instruction {
    execute: ldd_mem_r16_val_r8,
    timings: (1, 8),
    text: format!("ldd (HL), A"),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::A)
  };

  let inc_sp = Instruction {
    execute: inc_wide_register,
    timings: (1, 8),
    text: format!("inc SP"),
    data: InstructionData::wide_dst(WideRegister::SP),
  };

  let inc_mem_hl = Instruction {
    execute: inc_mem_r16,
    timings: (1, 12),
    text: format!("inc (HL)"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let dec_mem_hl = Instruction {
    execute: inc_mem_r16,
    timings: (1, 12),
    text: format!("dec (HL)"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let ld_mem_hl_n = Instruction {
    execute: ld_mem_r16_immediate,
    timings: (2, 12),
    text: format!("ld (HL), n"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let scf = Instruction {
    execute: scf,
    timings: (1, 4),
    text: format!("SCF"),
    data: InstructionData::default(),
  };

  vec![
    no_op, load_imm_bc, load_bc_a, inc_bc, inc_b, rlca, ld_nn_sp, add_hl_bc, ld_a_bc,
    dec_bc, inc_c, dec_c, ld_c_n, rrca, stop, load_imm_de, load_mem_de_a, inc_de,
    dec_d, ld_d_n, rla, jr_n, add_hl_de, ld_a_de, dec_de, inc_e, dec_e, ld_e_n, rra,
    jr_nz_n, load_imm_hl, ldi_hl_a, inc_hl, inc_h, dec_h, ld_h_n, daa, jr_z_n,
    add_hl_hl, ldi_a_hl, dec_hl, inc_l, dec_l, ld_l_n, cpl, jr_nc_n, load_imm_sp,
    ldd_hl_a, inc_sp, inc_mem_hl, dec_mem_hl, ld_mem_hl_n, scf,
  ]
}
