use crate::cpu::{Registers, SmallWidthRegister, WideRegister, CARRY_FLAG, ZERO_FLAG};
use crate::memory::{isset16, isset32, isset8, MemoryPtr};
use crate::util::{
  carries_add16_signed_8bit, carries_add8, carries_add8_with_carry, carries_sub16_signed_8bit,
  carries_sub8, carries_sub8_with_carry, half_carry_add8, half_carry_sub8,
};
use log::trace;

/// We re-use some instruction functions for multiple register implementations
/// This struct carries data for the single-implementation for many opcode instruction methods
#[derive(Clone, Debug)]
pub struct InstructionData {
  pub code: u8,
  pub flag_mask: u8,
  pub flag_expected: u8,
  pub bit: u8,

  pub small_reg_one: SmallWidthRegister,
  pub small_reg_dst: SmallWidthRegister,

  pub wide_reg_one: WideRegister,
  pub wide_reg_dst: WideRegister,
}

/// When not in use fields get a default value. This can add some risk, if an instruction data is
/// improperly configured it can be hard to debug, I should revisit this at some point and consider
/// optional values with .unwrap, but it might not be worth it for the clarity.
impl Default for InstructionData {
  fn default() -> InstructionData {
    InstructionData {
      code: 0,
      bit: 0,
      flag_mask: 0,
      flag_expected: 0,
      small_reg_one: SmallWidthRegister::SmallUnset,
      small_reg_dst: SmallWidthRegister::SmallUnset,
      wide_reg_one: WideRegister::WideUnset,
      wide_reg_dst: WideRegister::WideUnset,
    }
  }
}

impl InstructionData {
  pub fn rst_n(code: u8) -> InstructionData {
    let mut m = InstructionData::default();
    m.code = code;
    m
  }

  pub fn with_flag(&self, mask: u8, expected: u8) -> InstructionData {
    let mut m = self.clone();
    m.flag_mask = mask;
    m.flag_expected = expected;
    m
  }

  pub fn with_bit(&self, bit: u8) -> InstructionData {
    let mut m = self.clone();
    m.bit = bit;
    m
  }

  pub fn small_src(r: SmallWidthRegister) -> InstructionData {
    let mut a = InstructionData::default();
    a.small_reg_one = r;
    a
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

  pub fn wide_dst_wide_src(r: WideRegister, l: WideRegister) -> InstructionData {
    let mut a = InstructionData::wide_dst(r);
    a.wide_reg_one = l;
    a
  }

  pub fn small_dst_wide_src(r: SmallWidthRegister, l: WideRegister) -> InstructionData {
    let mut a = InstructionData::small_dst(r);
    a.wide_reg_one = l;
    a
  }

  pub fn small_dst_small_src(r: SmallWidthRegister, l: SmallWidthRegister) -> InstructionData {
    let mut a = InstructionData::small_dst(r);
    a.small_reg_one = l;
    a
  }
}

/// The instruction struct contains the implementation of and metadata on an instruction
#[derive(Clone)]
pub struct Instruction {
  pub execute: fn(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData),
  pub cycles: u16,
  pub text: String,
  pub data: InstructionData,
}

/// No-op just increments the stack pointer
pub fn no_op(registers: &mut Registers, _memory: &mut MemoryPtr, _additional: &InstructionData) {
  registers.inc_pc(1);
}

/// Load immediate loads a 16 bit value following this instruction places it in a register
pub fn ld_imm_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  //Load the 16 bit value after the opcode and store it to the dst register
  let imm_val = memory.read_u16(registers.pc() + 1);

  //Increment the PC by three once arguments are collected
  registers.inc_pc(3);

  // Store the immediate value in the designated register
  registers.write_r16(additional.wide_reg_dst, imm_val);

  trace!(
    "Load Immediate to r16 {:?} {}",
    additional.wide_reg_dst,
    imm_val
  );
}

/// Loads an immediate 8-bit value into the address pointed to by the wide dest register
pub fn ld_mem_r16_immediate(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let val = memory.read_u8(registers.pc() + 1);
  registers.inc_pc(2);
  let addr = registers.read_r16(additional.wide_reg_dst);
  memory.write_u8(addr, val);
}

/// Load immediate loads a 8 bit value following this instruction places it in a small register
pub fn ld_imm_r8(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  //Load the 8 bit value after the opcode and store it to the dst register
  let imm_val = memory.read_u8(registers.pc() + 1);

  registers.write_r8(additional.small_reg_dst, imm_val);

  //Increment the PC by three once finished
  registers.inc_pc(2);
}

/// Write the value of small register one to the address pointed to by wide_reg_dst
pub fn ld_reg8_mem_reg16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  // Store the provided 8-bit register to the location pointed to by the 16-bit dst register
  let reg_val = registers.read_r8(additional.small_reg_one);
  let mem_dst = registers.read_r16(additional.wide_reg_dst);
  memory.write_u8(mem_dst, reg_val);

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Replace the value of small_reg_dst with the value of small_reg_one
pub fn ld_r8_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  registers.write_r8(
    additional.small_reg_dst,
    registers.read_r8(additional.small_reg_one),
  );
}

/// Replace the value of small_reg_dst with whatever is in the memory address
/// pointed to by the provided immediate value
pub fn ld_r8_indirect_imm(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let target_addr = memory.read_u16(registers.pc() + 1);
  registers.inc_pc(3);
  registers.write_r8(additional.small_reg_dst, memory.read_u8(target_addr));
}

/// Increment the value of a wide-register by one
pub fn inc_wide_register(
  registers: &mut Registers,
  _memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  // Increment the destination wide register by one
  registers.write_r16(
    additional.wide_reg_dst,
    registers.read_r16(additional.wide_reg_dst) + 1,
  );

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Increment the value of a small register by one
pub fn inc_small_register(
  registers: &mut Registers,
  _memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);

  let l = registers.read_r8(additional.small_reg_dst);
  let result = l + 1;

  // Increment the destination register by one
  registers.write_r8(additional.small_reg_dst, result);
  registers.set_flags(result == 0, false, half_carry_add8(l, 1), registers.carry());
}

// Increment an 8-bit value at a given memory address by one.
pub fn inc_mem_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);

  let addr = registers.read_r16(additional.wide_reg_dst);
  let l = memory.read_u8(addr);
  let result = l + 1;

  // Increment by one and modify memory
  memory.write_u8(addr, result);
  registers.set_flags(result == 0, false, half_carry_add8(l, 1), registers.carry());
}

/// Decrement the value of memory pointed by a wide register by one
/// and write it back to the same location in memory
pub fn dec_mem_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let addr = registers.read_r16(additional.wide_reg_dst);
  let l = memory.read_u8(addr);
  let result = l - 1;

  // Increment by one and modify memory
  memory.write_u8(addr, result);

  registers.set_flags(result == 0, true, half_carry_sub8(l, 1), registers.carry());
}

/// Add two 8-bit values and set flags
fn add_core(acc: u8, operand: u8, registers: &mut Registers) -> u8 {
  let (half_carry, carry) = carries_add8(acc, operand);
  let result = acc + operand;
  registers.set_flags(result == 0, false, half_carry, carry);
  result
}

/// Add an immediate to a small register
fn add_r8_n(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = memory.read_u8(registers.pc() + 1);

  // We calculate the registers here since its shared with other adds
  let result = add_core(acc, operand, registers);
  registers.write_r8(additional.small_reg_dst, result);

  // Increment the PC by one once finished
  registers.inc_pc(2);
}

/// Add two small registers (small_reg_one to small_reg_dst)
fn add_r8_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = registers.read_r8(additional.small_reg_one);

  // We calculate the registers here since its shared with immediates
  let result = add_core(acc, operand, registers);
  registers.write_r8(additional.small_reg_dst, result);

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Same as sub but subtracts an additional 1 if the carry bit is set and sets the resulting carry
/// and half carries appropriately
fn sub_core_with_carry(acc: u8, operand: u8, registers: &mut Registers, carry: bool) -> u8 {
  let result = acc - operand - if carry { 1 } else { 0 };
  let (half_carry, carry) = carries_sub8_with_carry(acc, operand, carry);
  registers.set_flags(result == 0, true, half_carry, carry);
  result
}

/// This function forms the foundation for all common 8-bit subtractions
fn sub_core(acc: u8, operand: u8, registers: &mut Registers) -> u8 {
  sub_core_with_carry(acc, operand, registers, false)
}

/// Subtract two small registers (small_reg_one to small_reg_dst)
fn sub_r8_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = registers.read_r8(additional.small_reg_one);

  let result = sub_core(acc, operand, registers);
  registers.write_r8(additional.small_reg_dst, result);

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Subtract an immediate from a small dst register
fn sub_r8_n(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = memory.read_u8(registers.pc() + 1);

  let result = sub_core(acc, operand, registers);
  registers.write_r8(additional.small_reg_dst, result);

  // Increment the PC by one once finished
  registers.inc_pc(2);
}

/// The core of an AND operation
fn and_core(acc: u8, operand: u8, registers: &mut Registers) -> u8 {
  let result = acc & operand;
  registers.set_flags(result == 0, false, true, false);
  result
}

/// And of two small registers
fn and_r8_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = registers.read_r8(additional.small_reg_one);
  let result = and_core(acc, operand, registers);

  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// And imm with small reg dst
fn and_r8_n(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = memory.read_u8(registers.pc() + 1);
  let result = and_core(acc, operand, registers);

  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(2);
}

/// The core of an OR operation
fn or_core(acc: u8, operand: u8, registers: &mut Registers) -> u8 {
  let result = acc | operand;
  registers.set_flags(result == 0, false, false, false);
  result
}

/// or two small registers
fn or_r8_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = registers.read_r8(additional.small_reg_one);

  let result = or_core(acc, operand, registers);
  registers.write_r8(additional.small_reg_dst, result);

  registers.inc_pc(1);
}

/// bitwise or small reg dst with immediate
fn or_r8_n(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = memory.read_u8(registers.pc() + 1);
  let result = or_core(acc, operand, registers);

  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(2);
}

/// cp two small registers
fn cp_r8_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = registers.read_r8(additional.small_reg_one);

  // We discard the result but keep the changes to flags
  sub_core(acc, operand, registers);

  registers.inc_pc(1);
}

/// cp small reg dst against an immediate
fn cp_r8_n(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = memory.read_u8(registers.pc() + 1);
  sub_core(acc, operand, registers);

  // Increment the PC by one once finished
  registers.inc_pc(2);
}

/// Implement the core xor logic for 8 bit values
fn xor_core(v1: u8, v2: u8, registers: &mut Registers) -> u8 {
  let result = v1 ^ v2;
  registers.set_flags(result == 0, false, false, false);
  result
}

/// XOR two small registers
fn xor_r8_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let v_src = registers.read_r8(additional.small_reg_one);
  let v_dst = registers.read_r8(additional.small_reg_dst);
  let result = xor_core(v_dst, v_src, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

/// XOR small dst register with immediate
fn xor_r8_n(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let v_src = memory.read_u8(registers.pc() + 1);
  registers.inc_pc(2);
  let v_dst = registers.read_r8(additional.small_reg_dst);
  let result = xor_core(v_dst, v_src, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

/// The core logic for subtraction of 8-bit values through a carry
fn sbc_core(acc: u8, val: u8, registers: &mut Registers) -> u8 {
  let carry = registers.carry();
  sub_core_with_carry(acc, val, registers, carry)
}

/// Subtract through carry using two small registers (small_reg_one to small_reg_dst)
fn sbc_r8_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let v_src = registers.read_r8(additional.small_reg_one);
  let v_dst = registers.read_r8(additional.small_reg_dst);
  let result = sbc_core(v_dst, v_src, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

/// Subtract immediate through carry
fn sbc_r8_n(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let sub_v = memory.read_u8(registers.pc() + 1);
  let result = sbc_core(origin, sub_v, registers);
  registers.write_r8(additional.small_reg_dst, result);

  registers.inc_pc(2);
}

/// Load small reg to 0xFF00 + n
fn ld_ff00_imm_r8(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  // Fetch the immediate offset
  let add_v: u16 = memory.read_u8(registers.pc() + 1).into();

  // Increment the PC by two
  registers.inc_pc(2);

  let rval = registers.read_r8(additional.small_reg_dst);
  memory.write_u8(0xFF00 + add_v, rval);
}

/// Load (0xFF00 + n) to small reg
// TODO: This and its dual function should have reversed names
fn ld_r8_ff00_imm(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  // Fetch
  let add_v: u16 = memory.read_u8(registers.pc() + 1).into();

  // Increment the PC by two
  registers.inc_pc(2);

  let read_value = memory.read_u8(0xFF00 + add_v);
  registers.write_r8(additional.small_reg_dst, read_value);
}

/// Load small reg dst to 0xFF00 + small reg one
fn ld_ff00_r8_r8(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let addr_offset: u16 = registers.read_r8(additional.small_reg_one) as u16;
  let rval = registers.read_r8(additional.small_reg_dst);
  memory.write_u8(0xFF00 + addr_offset, rval);
}

/// Load 0xFF00 + small reg one into small_reg_dst
// TODO: This function and it's dual should have their names reversed
fn ld_r8_ff00_r8(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let addr_offset: u16 = registers.read_r8(additional.small_reg_one).into();
  let rval = memory.read_u8(0xFF00 + addr_offset);
  registers.write_r8(additional.small_reg_dst, rval);
}

/// Generic add with carry, sets flags and returns result
fn adc_generic(acc: u8, r: u8, registers: &mut Registers) -> u8 {
  let result = acc + r + if registers.carry() { 1 } else { 0 };
  let (half_carry, carry) = carries_add8_with_carry(acc, r, registers.carry());
  registers.set_flags(result == 0, false, half_carry, carry);
  result
}

/// Add with carry an immediate to a small register
fn adc_r8_imm(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  // Fetch
  let add_v = memory.read_u8(registers.pc() + 1);

  // Increment the PC by two
  registers.inc_pc(2);

  // Execute
  let origin = registers.read_r8(additional.small_reg_dst);
  let result = adc_generic(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

/// Add two small registers (small_reg_one to small_reg_dst)
/// Also add one if the carry flag is set
fn adc_r8_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  // Increment the PC by one once finished
  registers.inc_pc(1);

  let origin = registers.read_r8(additional.small_reg_dst);
  let add_v = registers.read_r8(additional.small_reg_one);
  let result = adc_generic(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

/// Add value at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn add_r8_mem_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  // Increment the PC by one once finished
  registers.inc_pc(1);

  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);

  let result = add_core(origin, add_v, registers);

  registers.write_r8(additional.small_reg_dst, result);
}

/// Subtract value add wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn sub_r8_mem_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  // Increment the PC by one once finished
  registers.inc_pc(1);

  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);
  let result = sub_core(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

/// Subtract through carry value at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn sbc_r8_mem_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);

  let result = sbc_core(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// And memory at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn and_r8_mem_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);

  let result = and_core(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Cp memory at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn cp_r8_mem_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let target_addr = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(target_addr);
  let origin = registers.read_r8(additional.small_reg_dst);
  sub_core(origin, add_v, registers);
  registers.inc_pc(1);
}

/// or memory at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn or_r8_mem_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);

  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);
  let result = or_core(origin, add_v, registers);

  registers.write_r8(additional.small_reg_dst, result);
}

/// XOR memory at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn xor_r8_mem_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  // Increment the PC by one once finished
  registers.inc_pc(1);

  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);

  let result = xor_core(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

/// Add value add wide_register_one in memory to small_reg_dst
/// Add a further 1 if the carry is set
/// save the result in small_reg_dst
fn adc_r8_mem_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  // Increment the PC by one once finished
  registers.inc_pc(1);

  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);
  let result = adc_generic(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

/// Rotate 8-bit register left, placing whatever is in bit 7 in the carry bit before
fn rotate_left_with_carry(
  registers: &mut Registers,
  _memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let mut a = registers.read_r8(additional.small_reg_dst);
  let carry = a & (1 << 7) != 0;
  a = a.rotate_left(1);
  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(false, false, false, carry);

  // Increment PC by one
  registers.inc_pc(1);
}

/// Rotate 8-bit register right, placing whatever is in bit 0 in the carry bit before
fn rotate_right_with_carry(
  registers: &mut Registers,
  _memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let mut a = registers.read_r8(additional.small_reg_dst);
  let carry = a & (1) != 0;
  a = a.rotate_right(1);
  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(false, false, false, carry);

  // Increment PC by one
  registers.inc_pc(1);
}

fn rotate_r8_left_through_carry(
  registers: &mut Registers,
  _memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let mut a = registers.read_r8(additional.small_reg_dst);

  let will_carry = a & (0x1 << 7) != 0;
  a = a << 1;

  if registers.carry() {
    a += 1;
  }

  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(false, false, false, will_carry);

  // Increment PC by one
  registers.inc_pc(1);
}

/// Rotate 8-bit register right using the carry bit as an additional bit.
/// In practice, store the current carry bit, set carry bit to value of bit 0
/// then shift right everything by one bit and then replace bit 7 with the origin
/// carry bit
fn rotate_r8_right_through_carry(
  registers: &mut Registers,
  _memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let mut a = registers.read_r8(additional.small_reg_dst);

  let will_carry = a & 0x1 != 0;
  a = a >> 1;

  if registers.carry() {
    a += 1 << 7;
  }

  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(false, false, false, will_carry);

  // Increment PC by one
  registers.inc_pc(1);
}

/// Decrement the value of a small register by one
fn dec_wide_register(
  registers: &mut Registers,
  _memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  // Increment the destination register by one
  registers.write_r16(
    additional.wide_reg_dst,
    registers.read_r16(additional.wide_reg_dst) - 1,
  );

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Decrement the value of a small register by one
fn dec_small_register(
  registers: &mut Registers,
  _memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let l = registers.read_r8(additional.small_reg_dst);
  let result = l - 1;

  // Increment the destination register by one
  registers.write_r8(additional.small_reg_dst, result);

  registers.set_flags(result == 0, true, half_carry_sub8(l, 1), registers.carry());

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Write the contents of a wide register to an address in memory
fn load_immediate_wide_register(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  // Find the address we are writing to
  let load_address = memory.read_u16(registers.pc() + 1);

  // Write the contents of wide register one to that location
  memory.write_u16(load_address, registers.read_r16(additional.wide_reg_one));

  // Increment the PC by one once finished
  registers.inc_pc(3);
}

fn load_indirect_nn_small_register(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  // Find the address we are writing to
  let load_address = memory.read_u16(registers.pc() + 1);

  // Write the contents of wide register one to that location
  memory.write_u8(load_address, registers.read_r8(additional.small_reg_one));

  // Increment the PC by one once finished
  registers.inc_pc(3);
}

/// Place the value of a small register into the
/// memory address pointed to by the wide destination
/// register and then increment the wide destination register
/// Example, ld HL, 0 ld A, 5 ldi (HL), A will leave [0] = 5, A = 5, HL = 1
fn ldi_mem_r16_val_r8(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let wide_reg = registers.read_r16(additional.wide_reg_dst);
  let target = registers.read_r8(additional.small_reg_one);
  memory.write_u8(wide_reg, target);
  registers.write_r16(additional.wide_reg_dst, wide_reg + 1);
}

/// Place memory pointed to by the wide register into the small dst register
/// then increment the wide register
fn ldi_r8_mem_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let wide_reg = registers.read_r16(additional.wide_reg_one);
  let mem = memory.read_u8(wide_reg);

  registers.write_r8(additional.small_reg_dst, mem);
  registers.write_r16(additional.wide_reg_one, wide_reg + 1);

  registers.inc_pc(1);
}

/// Like ldi but decrement
fn ldd_mem_r16_val_r8(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let wide_reg = registers.read_r16(additional.wide_reg_dst);
  let target = registers.read_r8(additional.small_reg_one);
  memory.write_u8(wide_reg, target);
  registers.write_r16(additional.wide_reg_dst, wide_reg - 1);
}

/// Like ldi but decrement
fn ldd_r8_mem_r16(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let wide_reg = registers.read_r16(additional.wide_reg_one);
  let mem = memory.read_u8(wide_reg);

  registers.write_r8(additional.small_reg_dst, mem);
  registers.write_r16(additional.wide_reg_one, wide_reg - 1);
}

/// Add a wide register to a wide register
fn add_r16_r16(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  let l = registers.read_r16(additional.wide_reg_dst);
  let r = registers.read_r16(additional.wide_reg_one);
  let result = l + r;

  // Did the 11th bits carry?
  let half_carried = isset16((l & 0xFFF) + (r & 0xFFF), 0x1 << 12);
  let carried = isset32(l as u32 + r as u32, 0x1 << 16);

  registers.write_r16(additional.wide_reg_dst, result);
  registers.set_flags(registers.zero(), false, half_carried, carried);

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Add an immediate byte (signed) to a wide register
fn add_r16_n(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let l = registers.read_r16(additional.wide_reg_dst);
  let add_v = memory.read_u8(registers.pc() + 1);
  let result = register_plus_signed_8_bit_immediate(l, add_v, registers);

  registers.write_r16(additional.wide_reg_dst, result);
  registers.inc_pc(2);
}

/// Load wide_reg_one into wide reg dst  
fn ld_r16_r16(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  registers.write_r16(
    additional.wide_reg_dst,
    registers.read_r16(additional.wide_reg_one),
  );
}

fn register_plus_signed_8_bit_immediate(
  mut acc: u16,
  immediate: u8,
  registers: &mut Registers,
) -> u16 {
  if isset8(immediate, 0x80) {
    // TODO: Broken
    let immediate = (!immediate) + 1;
    let (half_carry, carry) = carries_sub16_signed_8bit(acc, immediate);
    let immediate = immediate as u16;
    let origin = acc;
    acc -= immediate;
    registers.set_flags(false, false, half_carry, carry);
    println!(
      "{} - {} = {} ({} {})",
      origin, immediate, acc, half_carry, carry
    );
  } else {
    let (half_carry, carry) = carries_add16_signed_8bit(acc, immediate);
    acc += immediate as u16;
    registers.set_flags(false, false, half_carry, carry);
  }

  acc
}

/// Add an immediate byte (signed) to wide reg one and then save it to wide reg dst
fn ld_r16_r16_plus_n(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let l = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(registers.pc() + 1);
  let result = register_plus_signed_8_bit_immediate(l, add_v, registers);

  registers.write_r16(additional.wide_reg_dst, result);
  registers.inc_pc(2);
}

/// Load a value from memory to a small register
fn load_r16_mem_to_r8(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let address = registers.read_r16(additional.wide_reg_one);
  let value = memory.read_u8(address);
  registers.write_r8(additional.small_reg_dst, value);

  registers.inc_pc(1);
}

/// Stop the processor & screen until button press
fn stop(registers: &mut Registers, _memory: &mut MemoryPtr, _additional: &InstructionData) {
  registers.inc_pc(2);
}

/// Escape
fn escape(registers: &mut Registers, _memory: &mut MemoryPtr, _additional: &InstructionData) {
  registers.inc_pc(1);
  registers.escaped = true;
}

/// Jump relative by a signed 8-bit value following the opcode
fn jump_relative_signed_immediate(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let byte = memory.read_u8(registers.pc() + 1) as i8;
  registers.inc_pc(2);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected {
    registers.last_clock = 4;
    registers.jump_relative(byte);
  }
}

/// Return if the flags & mask == expected
fn ret(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected {
    registers.last_clock = 12;
    let ret_pc = registers.stack_pop16(memory);
    registers.set_pc(ret_pc);
  }
}

/// Enable interrupts
fn enable_interrupts(
  registers: &mut Registers,
  _memory: &mut MemoryPtr,
  _additional: &InstructionData,
) {
  trace!("Interrupts enabled");
  registers.inc_pc(1);
  registers.ime = true;
}

/// Disable interrupts
fn disable_interrupts(
  registers: &mut Registers,
  _memory: &mut MemoryPtr,
  _additional: &InstructionData,
) {
  trace!("Interrupts disabled");
  registers.inc_pc(1);
  registers.ime = false;
}

/// Return and enable interrupts
fn reti(registers: &mut Registers, memory: &mut MemoryPtr, _additional: &InstructionData) {
  registers.ime = true;
  let ret_pc = registers.stack_pop16(memory);
  registers.set_pc(ret_pc);
}

/// Jump to destination if flags & mask == expected
fn jump_immediate(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let target_address = memory.read_u16(registers.pc() + 1);
  registers.inc_pc(3);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected {
    registers.last_clock = 4;
    registers.set_pc(target_address);
  }
}

/// Jump to the value stored in a wide register
fn jump_wide_reg(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let target_address = registers.read_r16(additional.wide_reg_dst);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected {
    registers.set_pc(target_address);
  }
}

/// Call function if flags & mask == expected
fn call_immediate(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  let target_address = memory.read_u16(registers.pc() + 1);
  registers.inc_pc(3);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected {
    registers.last_clock = 12;
    registers.stack_push16(registers.pc(), memory);
    registers.set_pc(target_address);
  }
}

/// DAA takes the result of an arithmetic operation and makes it binary coded
/// retrospectively
fn daa(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  let target = registers.read_r8(additional.small_reg_dst);

  // Not entirely sure what this does but the general process seems to be
  // if A & 0xF > 0x9 or H then add $06 to A
  // if A & 0xF0 > 0x99 or C then add $60 to A

  let mut t = 0;

  if target & 0xF > 0x9 {
    t += 1;
  }

  let carry = if target & 0xF0 > 0x99 {
    t += 2;
    true
  } else {
    false
  };

  let result = match t {
    0 => target,
    1 => target + if registers.subtract() { 0xFA } else { 0x06 }, // -6 or +6
    2 => target + if registers.subtract() { 0xA0 } else { 0x60 }, // -60 or +60
    3 => target + if registers.subtract() { 0x9A } else { 0x66 }, // -66 or + 66
    _ => panic!("impossible condition for DAA"),
  };

  trace!(
    "DAA {} {} {} {} {}",
    result,
    result == 0,
    registers.subtract(),
    false,
    carry
  );

  // https://forums.nesdev.com/viewtopic.php?t=15944
  // https://stackoverflow.com/questions/8119577/z80-daa-instruction
  registers.write_r8(additional.small_reg_dst, result);
  registers.set_flags(result == 0, registers.subtract(), false, carry);
  registers.inc_pc(1);
}

fn invalid_op(_registers: &mut Registers, _memory: &mut MemoryPtr, _additional: &InstructionData) {
  unimplemented!();
}

/// Flip all bits in an r8
fn cpl_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  registers.write_r8(
    additional.small_reg_dst,
    !registers.read_r8(additional.small_reg_dst),
  );
  registers.set_flags(registers.zero(), true, true, registers.carry());
}

/// Push a 16-bit register to the stack
fn push_wide_register(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let to_push = registers.read_r16(additional.wide_reg_dst);
  registers.stack_push16(to_push, memory);
}

/// Pop two bytes from the stack and store them in specified register
fn pop_wide_register(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let popped_value = registers.stack_pop16(memory);
  registers.write_r16(additional.wide_reg_dst, popped_value);
}

/// Halt the processor
/// Stop doing anything until the next interrupt
fn halt(registers: &mut Registers, _memory: &mut MemoryPtr, _additional: &InstructionData) {
  registers.halted = true;
  registers.inc_pc(1);
}

/// Sets the carry flag, resets negative and half carry flags, zero unaffected
fn scf(registers: &mut Registers, _memory: &mut MemoryPtr, _additional: &InstructionData) {
  registers.inc_pc(1);
  registers.set_flags(registers.zero(), false, false, true);
}

/// Complement the carry flag (flip it)
fn ccf(registers: &mut Registers, _memory: &mut MemoryPtr, _additional: &InstructionData) {
  registers.inc_pc(1);
  registers.set_flags(registers.zero(), false, false, !registers.carry());
}

/// Push current PC to stack then jump to n (8-bit immediate)
fn rst_n(registers: &mut Registers, memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  registers.stack_push16(registers.pc(), memory);
  registers.set_pc(additional.code as u16);
}

pub fn instruction_set() -> Vec<Instruction> {
  let no_op = Instruction {
    execute: no_op,
    cycles: 4,
    text: "NOP".to_string(),
    data: InstructionData::default(),
  };

  let load_imm_bc = Instruction {
    execute: ld_imm_r16,
    cycles: 12,
    text: "ld BC, nn".to_string(),
    data: InstructionData::wide_dst(WideRegister::BC),
  };

  let load_bc_a = Instruction {
    execute: ld_reg8_mem_reg16,
    cycles: 8,
    text: "ld (BC) A".to_string(),
    data: InstructionData::wide_dst_small_in(WideRegister::BC, SmallWidthRegister::A),
  };

  let inc_bc = Instruction {
    execute: inc_wide_register,
    cycles: 8,
    text: "inc BC".to_string(),
    data: InstructionData::wide_dst(WideRegister::BC),
  };

  let inc_b = Instruction {
    execute: inc_small_register,
    cycles: 4,
    text: format!("inc B"),
    data: InstructionData::small_dst(SmallWidthRegister::B),
  };

  let dec_b = Instruction {
    execute: dec_small_register,
    cycles: 4,
    text: "dec B".to_string(),
    data: InstructionData::small_dst(SmallWidthRegister::B),
  };

  let load_imm_b = Instruction {
    execute: ld_imm_r8,
    cycles: 8,
    text: "ld B, n".to_string(),
    data: InstructionData::small_dst(SmallWidthRegister::B),
  };

  let rlca = Instruction {
    execute: rotate_left_with_carry,
    cycles: 4,
    text: format!("RLCA"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let ld_nn_sp = Instruction {
    execute: load_immediate_wide_register,
    cycles: 20,
    text: format!("ld (NN), SP"),
    data: InstructionData::wide_src(WideRegister::SP),
  };

  let add_hl_bc = Instruction {
    execute: add_r16_r16,
    cycles: 8,
    text: format!("add HL, BC"),
    data: InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::BC),
  };

  let ld_a_bc = Instruction {
    execute: load_r16_mem_to_r8,
    cycles: 8,
    text: format!("ld A, (BC)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::BC),
  };

  let dec_bc = Instruction {
    execute: dec_wide_register,
    cycles: 8,
    text: format!("dec BC"),
    data: InstructionData::wide_dst(WideRegister::BC),
  };

  let inc_c = Instruction {
    execute: inc_small_register,
    cycles: 4,
    text: format!("inc C"),
    data: InstructionData::small_dst(SmallWidthRegister::C),
  };

  let dec_c = Instruction {
    execute: dec_small_register,
    cycles: 4,
    text: format!("dec C"),
    data: InstructionData::small_dst(SmallWidthRegister::C),
  };

  let ld_c_n = Instruction {
    execute: ld_imm_r8,
    cycles: 8,
    text: format!("ld C, n"),
    data: InstructionData::small_dst(SmallWidthRegister::C),
  };

  let rrca = Instruction {
    execute: rotate_right_with_carry,
    cycles: 4,
    text: format!("RRCA"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let stop = Instruction {
    execute: stop,
    cycles: 2,
    text: format!("STOP"),
    data: InstructionData::default(),
  };

  let load_imm_de = Instruction {
    execute: ld_imm_r16,
    cycles: 12,
    text: format!("ld DE, nn"),
    data: InstructionData::wide_dst(WideRegister::DE),
  };

  let load_mem_de_a = Instruction {
    execute: ld_reg8_mem_reg16,
    cycles: 8,
    text: format!("ld (DE), A"),
    data: InstructionData::wide_dst_small_in(WideRegister::DE, SmallWidthRegister::A),
  };

  let inc_de = Instruction {
    execute: inc_wide_register,
    cycles: 8,
    text: "inc DE".to_string(),
    data: InstructionData::wide_dst(WideRegister::DE),
  };

  let inc_d = Instruction {
    execute: inc_small_register,
    cycles: 4,
    text: format!("inc D"),
    data: InstructionData::small_dst(SmallWidthRegister::D),
  };

  let dec_d = Instruction {
    execute: dec_small_register,
    cycles: 4,
    text: format!("dec D"),
    data: InstructionData::small_dst(SmallWidthRegister::D),
  };

  let ld_d_n = Instruction {
    execute: ld_imm_r8,
    cycles: 8,
    text: format!("ld D, n"),
    data: InstructionData::small_dst(SmallWidthRegister::D),
  };

  let rla = Instruction {
    execute: rotate_r8_left_through_carry,
    cycles: 4,
    text: format!("RLA"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let jr_n = Instruction {
    execute: jump_relative_signed_immediate,
    cycles: 8, // This is always 12 but the remaining 4 are added by the instr itself
    text: format!("JR n"),
    data: InstructionData::default().with_flag(0, 0),
  };

  let add_hl_de = Instruction {
    execute: add_r16_r16,
    cycles: 8,
    text: format!("add HL, DE"),
    data: InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::DE),
  };

  let ld_a_de = Instruction {
    execute: load_r16_mem_to_r8,
    cycles: 8,
    text: format!("ld A, (DE)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::DE),
  };

  let dec_de = Instruction {
    execute: dec_wide_register,
    cycles: 8,
    text: format!("dec DE"),
    data: InstructionData::wide_dst(WideRegister::DE),
  };

  let inc_e = Instruction {
    execute: inc_small_register,
    cycles: 4,
    text: format!("inc E"),
    data: InstructionData::small_dst(SmallWidthRegister::E),
  };

  let dec_e = Instruction {
    execute: dec_small_register,
    cycles: 4,
    text: format!("dec E"),
    data: InstructionData::small_dst(SmallWidthRegister::E),
  };

  let ld_e_n = Instruction {
    execute: ld_imm_r8,
    cycles: 8,
    text: format!("ld E, n"),
    data: InstructionData::small_dst(SmallWidthRegister::E),
  };

  let rra = Instruction {
    execute: rotate_r8_right_through_carry,
    cycles: 4,
    text: format!("RRA"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let jr_nz_n = Instruction {
    execute: jump_relative_signed_immediate,
    cycles: 8,
    text: format!("JRNZ n"),
    data: InstructionData::default().with_flag(ZERO_FLAG, 0),
  };

  let load_imm_hl = Instruction {
    execute: ld_imm_r16,
    cycles: 12,
    text: format!("ld HL, nn"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let ldi_hl_a = Instruction {
    execute: ldi_mem_r16_val_r8,
    cycles: 8,
    text: format!("ldi (HL), A"),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::A),
  };

  let inc_hl = Instruction {
    execute: inc_wide_register,
    cycles: 8,
    text: format!("inc HL"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let inc_h = Instruction {
    execute: inc_small_register,
    cycles: 4,
    text: format!("inc H"),
    data: InstructionData::small_dst(SmallWidthRegister::H),
  };

  let dec_h = Instruction {
    execute: dec_small_register,
    cycles: 4,
    text: format!("dec H"),
    data: InstructionData::small_dst(SmallWidthRegister::H),
  };

  let ld_h_n = Instruction {
    execute: ld_imm_r8,
    cycles: 8,
    text: format!("ld H, n"),
    data: InstructionData::small_dst(SmallWidthRegister::H),
  };

  let daa = Instruction {
    execute: daa,
    cycles: 4,
    text: format!("daa"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let jr_z_n = Instruction {
    execute: jump_relative_signed_immediate,
    cycles: 8,
    text: format!("JRZ n"),
    data: InstructionData::default().with_flag(ZERO_FLAG, ZERO_FLAG),
  };

  let add_hl_hl = Instruction {
    execute: add_r16_r16,
    cycles: 8,
    text: format!("add HL, HL"),
    data: InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::HL),
  };

  let ldi_a_hl = Instruction {
    execute: ldi_r8_mem_r16,
    cycles: 8,
    text: format!("ldi A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let dec_hl = Instruction {
    execute: dec_wide_register,
    cycles: 8,
    text: format!("dec HL"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let inc_l = Instruction {
    execute: inc_small_register,
    cycles: 4,
    text: format!("inc L"),
    data: InstructionData::small_dst(SmallWidthRegister::L),
  };

  let dec_l = Instruction {
    execute: dec_small_register,
    cycles: 4,
    text: format!("dec L"),
    data: InstructionData::small_dst(SmallWidthRegister::L),
  };

  let ld_l_n = Instruction {
    execute: ld_imm_r8,
    cycles: 8,
    text: format!("ld L, n"),
    data: InstructionData::small_dst(SmallWidthRegister::L),
  };

  let cpl = Instruction {
    execute: cpl_r8,
    cycles: 4,
    text: format!("cpl"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let jr_nc_n = Instruction {
    execute: jump_relative_signed_immediate,
    cycles: 8,
    text: format!("JRNC n"),
    data: InstructionData::default().with_flag(CARRY_FLAG, 0),
  };

  let load_imm_sp = Instruction {
    execute: ld_imm_r16,
    cycles: 12,
    text: format!("ld SP, nn"),
    data: InstructionData::wide_dst(WideRegister::SP),
  };

  let ldd_hl_a = Instruction {
    execute: ldd_mem_r16_val_r8,
    cycles: 8,
    text: format!("ldd (HL), A"),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::A),
  };

  let inc_sp = Instruction {
    execute: inc_wide_register,
    cycles: 8,
    text: format!("inc SP"),
    data: InstructionData::wide_dst(WideRegister::SP),
  };

  let inc_mem_hl = Instruction {
    execute: inc_mem_r16,
    cycles: 12,
    text: format!("inc (HL)"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let dec_mem_hl = Instruction {
    execute: dec_mem_r16,
    cycles: 12,
    text: format!("dec (HL)"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let ld_mem_hl_n = Instruction {
    execute: ld_mem_r16_immediate,
    cycles: 12,
    text: format!("ld (HL), n"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let scf = Instruction {
    execute: scf,
    cycles: 4,
    text: format!("SCF"),
    data: InstructionData::default(),
  };

  let jr_c_n = Instruction {
    execute: jump_relative_signed_immediate,
    cycles: 8,
    text: format!("JRC n"),
    data: InstructionData::default().with_flag(CARRY_FLAG, CARRY_FLAG),
  };

  let add_hl_sp = Instruction {
    execute: add_r16_r16,
    cycles: 8,
    text: format!("add HL, SP"),
    data: InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::SP),
  };

  let ldd_a_hl = Instruction {
    execute: ldd_r8_mem_r16,
    cycles: 8,
    text: format!("ldd A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let dec_sp = Instruction {
    execute: dec_wide_register,
    cycles: 8,
    text: format!("dec SP"),
    data: InstructionData::wide_dst(WideRegister::SP),
  };

  let inc_a = Instruction {
    execute: inc_small_register,
    cycles: 4,
    text: format!("inc A"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let dec_a = Instruction {
    execute: dec_small_register,
    cycles: 4,
    text: format!("dec A"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let ld_a_n = Instruction {
    execute: ld_imm_r8,
    cycles: 8,
    text: format!("ld A, n"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let ccf = Instruction {
    execute: ccf,
    cycles: 4,
    text: format!("CCF"),
    data: InstructionData::default(),
  };

  let ld_b_b = Instruction {
    execute: ld_r8_r8,
    cycles: 8,
    text: format!("ld B, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::B),
  };

  let ld_b_c = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld B, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::C),
  };

  let ld_b_d = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld B, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::D),
  };

  let ld_b_e = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld B, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::E),
  };

  let ld_b_h = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld B, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::H),
  };

  let ld_b_l = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld B, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::L),
  };

  let ld_b_hl = Instruction {
    execute: load_r16_mem_to_r8,
    cycles: 8,
    text: format!("ld B, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::B, WideRegister::HL),
  };

  let ld_b_a = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld B, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::A),
  };

  let ld_c_b = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld C, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::B),
  };

  let ld_c_c = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld C, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::C),
  };

  let ld_c_d = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld C, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::D),
  };

  let ld_c_e = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld C, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::E),
  };

  let ld_c_h = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld C, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::H),
  };

  let ld_c_l = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld C, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::L),
  };

  let ld_c_hl = Instruction {
    execute: load_r16_mem_to_r8,
    cycles: 8,
    text: format!("ld C, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::C, WideRegister::HL),
  };

  let ld_c_a = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld C, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::A),
  };

  let ld_d_b = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld D, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::B),
  };

  let ld_d_c = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld D, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::C),
  };

  let ld_d_d = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld D, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::D),
  };

  let ld_d_e = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld D, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::E),
  };

  let ld_d_h = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld D, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::H),
  };

  let ld_d_l = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld D, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::L),
  };

  let ld_d_hl = Instruction {
    execute: load_r16_mem_to_r8,
    cycles: 8,
    text: format!("ld D, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::D, WideRegister::HL),
  };

  let ld_d_a = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld D, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::A),
  };

  let ld_e_b = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld E, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::B),
  };

  let ld_e_c = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld E, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::C),
  };

  let ld_e_d = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld E, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::D),
  };

  let ld_e_e = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld E, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::E),
  };

  let ld_e_h = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld E, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::H),
  };

  let ld_e_l = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld E, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::L),
  };

  let ld_e_hl = Instruction {
    execute: load_r16_mem_to_r8,
    cycles: 8,
    text: format!("ld E, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::E, WideRegister::HL),
  };

  let ld_e_a = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld E, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::A),
  };

  let ld_h_b = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld H, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::B),
  };

  let ld_h_c = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld H, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::C),
  };

  let ld_h_d = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld H, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::D),
  };

  let ld_h_e = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld H, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::E),
  };

  let ld_h_h = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld H, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::H),
  };

  let ld_h_l = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld H, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::L),
  };

  let ld_h_hl = Instruction {
    execute: load_r16_mem_to_r8,
    cycles: 8,
    text: format!("ld H, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::H, WideRegister::HL),
  };

  let ld_h_a = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld H, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::A),
  };

  let ld_l_b = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld L, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::B),
  };

  let ld_l_c = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld L, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::C),
  };

  let ld_l_d = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld L, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::D),
  };

  let ld_l_e = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld L, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::E),
  };

  let ld_l_h = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld L, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::H),
  };

  let ld_l_l = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld L, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::L),
  };

  let ld_l_hl = Instruction {
    execute: load_r16_mem_to_r8,
    cycles: 8,
    text: format!("ld L, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::L, WideRegister::HL),
  };

  let ld_l_a = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld L, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::A),
  };

  let load_hl_b = Instruction {
    execute: ld_reg8_mem_reg16,
    cycles: 8,
    text: "ld (HL) B".to_string(),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::B),
  };

  let load_hl_c = Instruction {
    execute: ld_reg8_mem_reg16,
    cycles: 8,
    text: "ld (HL) C".to_string(),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::C),
  };

  let load_hl_d = Instruction {
    execute: ld_reg8_mem_reg16,
    cycles: 8,
    text: "ld (HL) D".to_string(),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::D),
  };

  let load_hl_e = Instruction {
    execute: ld_reg8_mem_reg16,
    cycles: 8,
    text: "ld (HL) E".to_string(),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::E),
  };

  let load_hl_h = Instruction {
    execute: ld_reg8_mem_reg16,
    cycles: 8,
    text: "ld (HL) H".to_string(),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::H),
  };

  let load_hl_l = Instruction {
    execute: ld_reg8_mem_reg16,
    cycles: 8,
    text: "ld (HL) L".to_string(),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::L),
  };

  let halt = Instruction {
    execute: halt,
    cycles: 4,
    text: "HALT".to_string(),
    data: InstructionData::default(),
  };

  let load_hl_a = Instruction {
    execute: ld_reg8_mem_reg16,
    cycles: 8,
    text: "ld (HL) A".to_string(),
    data: InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::A),
  };

  let ld_a_b = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld A, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B),
  };

  let ld_a_c = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld A, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let ld_a_d = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld A, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D),
  };

  let ld_a_e = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld A, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E),
  };

  let ld_a_h = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld A, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H),
  };

  let ld_a_l = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld A, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L),
  };

  let ld_a_hl = Instruction {
    execute: load_r16_mem_to_r8,
    cycles: 8,
    text: format!("ld A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let ld_a_a = Instruction {
    execute: ld_r8_r8,
    cycles: 4,
    text: format!("ld A, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A),
  };

  let add_a_b = Instruction {
    execute: add_r8_r8,
    cycles: 4,
    text: format!("add A, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B),
  };

  let add_a_c = Instruction {
    execute: add_r8_r8,
    cycles: 4,
    text: format!("add A, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let add_a_d = Instruction {
    execute: add_r8_r8,
    cycles: 4,
    text: format!("add A, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D),
  };

  let add_a_e = Instruction {
    execute: add_r8_r8,
    cycles: 4,
    text: format!("add A, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E),
  };

  let add_a_h = Instruction {
    execute: add_r8_r8,
    cycles: 4,
    text: format!("add A, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H),
  };

  let add_a_l = Instruction {
    execute: add_r8_r8,
    cycles: 4,
    text: format!("add A, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L),
  };

  let add_a_hl = Instruction {
    execute: add_r8_mem_r16,
    cycles: 8,
    text: format!("add A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let add_a_a = Instruction {
    execute: add_r8_r8,
    cycles: 4,
    text: format!("add A, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A),
  };

  // Add with carries

  let adc_a_b = Instruction {
    execute: adc_r8_r8,
    cycles: 4,
    text: format!("adc A, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B),
  };

  let adc_a_c = Instruction {
    execute: adc_r8_r8,
    cycles: 4,
    text: format!("adc A, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let adc_a_d = Instruction {
    execute: adc_r8_r8,
    cycles: 4,
    text: format!("adc A, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D),
  };

  let adc_a_e = Instruction {
    execute: adc_r8_r8,
    cycles: 4,
    text: format!("adc A, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E),
  };

  let adc_a_h = Instruction {
    execute: adc_r8_r8,
    cycles: 4,
    text: format!("adc A, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H),
  };

  let adc_a_l = Instruction {
    execute: adc_r8_r8,
    cycles: 4,
    text: format!("adc A, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L),
  };

  let adc_a_hl = Instruction {
    execute: adc_r8_mem_r16,
    cycles: 8,
    text: format!("adc A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let adc_a_a = Instruction {
    execute: adc_r8_r8,
    cycles: 4,
    text: format!("adc A, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A),
  };

  // Subtract
  let sub_a_b = Instruction {
    execute: sub_r8_r8,
    cycles: 4,
    text: format!("sub A, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B),
  };

  let sub_a_c = Instruction {
    execute: sub_r8_r8,
    cycles: 4,
    text: format!("sub A, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let sub_a_d = Instruction {
    execute: sub_r8_r8,
    cycles: 4,
    text: format!("sub A, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D),
  };

  let sub_a_e = Instruction {
    execute: sub_r8_r8,
    cycles: 4,
    text: format!("sub A, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E),
  };

  let sub_a_h = Instruction {
    execute: sub_r8_r8,
    cycles: 4,
    text: format!("sub A, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H),
  };

  let sub_a_l = Instruction {
    execute: sub_r8_r8,
    cycles: 4,
    text: format!("sub A, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L),
  };

  let sub_a_hl = Instruction {
    execute: sub_r8_mem_r16,
    cycles: 8,
    text: format!("sub A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let sub_a_a = Instruction {
    execute: sub_r8_r8,
    cycles: 4,
    text: format!("sub A, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A),
  };

  // Subtract with carry
  let sbc_a_b = Instruction {
    execute: sbc_r8_r8,
    cycles: 4,
    text: format!("sbc A, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B),
  };

  let sbc_a_c = Instruction {
    execute: sbc_r8_r8,
    cycles: 4,
    text: format!("sbc A, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let sbc_a_d = Instruction {
    execute: sbc_r8_r8,
    cycles: 4,
    text: format!("sbc A, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D),
  };

  let sbc_a_e = Instruction {
    execute: sbc_r8_r8,
    cycles: 4,
    text: format!("sbc A, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E),
  };

  let sbc_a_h = Instruction {
    execute: sbc_r8_r8,
    cycles: 4,
    text: format!("sbc A, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H),
  };

  let sbc_a_l = Instruction {
    execute: sbc_r8_r8,
    cycles: 4,
    text: format!("sbc A, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L),
  };

  let sbc_a_hl = Instruction {
    execute: sbc_r8_mem_r16,
    cycles: 8,
    text: format!("sbc A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let sbc_a_a = Instruction {
    execute: sbc_r8_r8,
    cycles: 4,
    text: format!("sbc A, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A),
  };

  // And
  let and_a_b = Instruction {
    execute: and_r8_r8,
    cycles: 4,
    text: format!("and A, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B),
  };

  let and_a_c = Instruction {
    execute: and_r8_r8,
    cycles: 4,
    text: format!("and A, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let and_a_d = Instruction {
    execute: and_r8_r8,
    cycles: 4,
    text: format!("and A, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D),
  };

  let and_a_e = Instruction {
    execute: and_r8_r8,
    cycles: 4,
    text: format!("and A, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E),
  };

  let and_a_h = Instruction {
    execute: and_r8_r8,
    cycles: 4,
    text: format!("and A, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H),
  };

  let and_a_l = Instruction {
    execute: and_r8_r8,
    cycles: 4,
    text: format!("and A, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L),
  };

  let and_a_hl = Instruction {
    execute: and_r8_mem_r16,
    cycles: 8,
    text: format!("and A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let and_a_a = Instruction {
    execute: and_r8_r8,
    cycles: 4,
    text: format!("and A, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A),
  };

  // Xor
  let xor_a_b = Instruction {
    execute: xor_r8_r8,
    cycles: 4,
    text: format!("xor A, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B),
  };

  let xor_a_c = Instruction {
    execute: xor_r8_r8,
    cycles: 4,
    text: format!("xor A, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let xor_a_d = Instruction {
    execute: xor_r8_r8,
    cycles: 4,
    text: format!("xor A, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D),
  };

  let xor_a_e = Instruction {
    execute: xor_r8_r8,
    cycles: 4,
    text: format!("xor A, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E),
  };

  let xor_a_h = Instruction {
    execute: xor_r8_r8,
    cycles: 4,
    text: format!("xor A, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H),
  };

  let xor_a_l = Instruction {
    execute: xor_r8_r8,
    cycles: 4,
    text: format!("xor A, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L),
  };

  let xor_a_hl = Instruction {
    execute: xor_r8_mem_r16,
    cycles: 8,
    text: format!("xor A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let xor_a_a = Instruction {
    execute: xor_r8_r8,
    cycles: 4,
    text: format!("xor A, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A),
  };

  // or
  let or_a_b = Instruction {
    execute: or_r8_r8,
    cycles: 4,
    text: format!("or A, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B),
  };

  let or_a_c = Instruction {
    execute: or_r8_r8,
    cycles: 4,
    text: format!("or A, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let or_a_d = Instruction {
    execute: or_r8_r8,
    cycles: 4,
    text: format!("or A, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D),
  };

  let or_a_e = Instruction {
    execute: or_r8_r8,
    cycles: 4,
    text: format!("or A, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E),
  };

  let or_a_h = Instruction {
    execute: or_r8_r8,
    cycles: 4,
    text: format!("or A, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H),
  };

  let or_a_l = Instruction {
    execute: or_r8_r8,
    cycles: 4,
    text: format!("or A, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L),
  };

  let or_a_hl = Instruction {
    execute: or_r8_mem_r16,
    cycles: 8,
    text: format!("or A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let or_a_a = Instruction {
    execute: or_r8_r8,
    cycles: 4,
    text: format!("or A, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A),
  };

  // cp
  let cp_a_b = Instruction {
    execute: cp_r8_r8,
    cycles: 4,
    text: format!("cp A, B"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B),
  };

  let cp_a_c = Instruction {
    execute: cp_r8_r8,
    cycles: 4,
    text: format!("cp A, C"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let cp_a_d = Instruction {
    execute: cp_r8_r8,
    cycles: 4,
    text: format!("cp A, D"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D),
  };

  let cp_a_e = Instruction {
    execute: cp_r8_r8,
    cycles: 4,
    text: format!("cp A, E"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E),
  };

  let cp_a_h = Instruction {
    execute: cp_r8_r8,
    cycles: 4,
    text: format!("cp A, H"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H),
  };

  let cp_a_l = Instruction {
    execute: cp_r8_r8,
    cycles: 4,
    text: format!("cp A, L"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L),
  };

  let cp_a_hl = Instruction {
    execute: cp_r8_mem_r16,
    cycles: 8,
    text: format!("cp A, (HL)"),
    data: InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL),
  };

  let cp_a_a = Instruction {
    execute: cp_r8_r8,
    cycles: 4,
    text: format!("cp A, A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A),
  };

  let ret_n_z = Instruction {
    execute: ret,
    cycles: 8,
    text: format!("retnz"),
    data: InstructionData::default().with_flag(ZERO_FLAG, 0),
  };

  let pop_bc = Instruction {
    execute: pop_wide_register,
    cycles: 12,
    text: format!("pop BC"),
    data: InstructionData::wide_dst(WideRegister::BC),
  };

  let jnz = Instruction {
    execute: jump_immediate,
    cycles: 12,
    text: format!("jnz NN"),
    data: InstructionData::default().with_flag(ZERO_FLAG, 0),
  };

  let jmp = Instruction {
    execute: jump_immediate,
    cycles: 8,
    text: format!("jmp NN"),
    data: InstructionData::default().with_flag(0, 0),
  };

  let callnz = Instruction {
    execute: call_immediate,
    cycles: 12,
    text: format!("callnz NN"),
    data: InstructionData::default().with_flag(ZERO_FLAG, 0),
  };

  let push_bc = Instruction {
    execute: push_wide_register,
    cycles: 16,
    text: format!("push BC"),
    data: InstructionData::wide_dst(WideRegister::BC),
  };

  let add_a_n = Instruction {
    execute: add_r8_n,
    cycles: 8,
    text: format!("add A, n"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let rst_0 = Instruction {
    execute: rst_n,
    cycles: 16,
    text: format!("rst 0"),
    data: InstructionData::rst_n(0),
  };

  let ret_z = Instruction {
    execute: ret,
    cycles: 8,
    text: format!("retz"),
    data: InstructionData::default().with_flag(ZERO_FLAG, ZERO_FLAG),
  };

  // TODO: RET AND JMP WITH NO COMPARISON TAKE A DIFFERENT NUMBER OF CYCLE

  let ret_from_fn = Instruction {
    execute: ret,
    cycles: 4,
    text: format!("ret"),
    data: InstructionData::default().with_flag(0, 0),
  };

  let jz = Instruction {
    execute: jump_immediate,
    cycles: 12,
    text: format!("jz NN"),
    data: InstructionData::default().with_flag(ZERO_FLAG, ZERO_FLAG),
  };

  let escape = Instruction {
    execute: escape,
    cycles: 4,
    text: format!("ESCAPE"),
    data: InstructionData::default(),
  };

  let callz = Instruction {
    execute: call_immediate,
    cycles: 12,
    text: format!("callz NN"),
    data: InstructionData::default().with_flag(ZERO_FLAG, ZERO_FLAG),
  };

  let call = Instruction {
    execute: call_immediate,
    cycles: 4,
    text: format!("call NN"),
    data: InstructionData::default().with_flag(0, 0),
  };

  let adc_a_n = Instruction {
    execute: adc_r8_imm,
    cycles: 8,
    text: format!("adc A, n"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let rst_8 = Instruction {
    execute: rst_n,
    cycles: 16,
    text: format!("rst 8"),
    data: InstructionData::rst_n(0x8),
  };

  let ret_n_c = Instruction {
    execute: ret,
    cycles: 8,
    text: format!("retnc"),
    data: InstructionData::default().with_flag(CARRY_FLAG, 0),
  };

  let pop_de = Instruction {
    execute: pop_wide_register,
    cycles: 12,
    text: format!("pop DE"),
    data: InstructionData::wide_dst(WideRegister::DE),
  };

  let jnc = Instruction {
    execute: jump_immediate,
    cycles: 12,
    text: format!("jnc NN"),
    data: InstructionData::default().with_flag(CARRY_FLAG, 0),
  };

  let invalid = Instruction {
    execute: invalid_op,
    cycles: 0,
    text: format!("INVALID"),
    data: InstructionData::default(),
  };

  let callnc = Instruction {
    execute: call_immediate,
    cycles: 12,
    text: format!("callnc NN"),
    data: InstructionData::default().with_flag(CARRY_FLAG, 0),
  };

  let push_de = Instruction {
    execute: push_wide_register,
    cycles: 16,
    text: format!("push DE"),
    data: InstructionData::wide_dst(WideRegister::DE),
  };

  let sub_a_imm = Instruction {
    execute: sub_r8_n,
    cycles: 8,
    text: format!("sub A, n"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let rst_10 = Instruction {
    execute: rst_n,
    cycles: 16,
    text: format!("rst 10H"),
    data: InstructionData::rst_n(0x10),
  };

  let ret_c = Instruction {
    execute: ret,
    cycles: 8,
    text: format!("retc"),
    data: InstructionData::default().with_flag(CARRY_FLAG, CARRY_FLAG),
  };

  let ret_i = Instruction {
    execute: reti,
    cycles: 8,
    text: format!("reti"),
    data: InstructionData::default(),
  };

  let jc = Instruction {
    execute: jump_immediate,
    cycles: 12,
    text: format!("jc NN"),
    data: InstructionData::default().with_flag(CARRY_FLAG, CARRY_FLAG),
  };

  let callc = Instruction {
    execute: call_immediate,
    cycles: 12,
    text: format!("callc NN"),
    data: InstructionData::default().with_flag(CARRY_FLAG, CARRY_FLAG),
  };

  let sbc_a_n = Instruction {
    execute: sbc_r8_n,
    cycles: 8,
    text: format!("sbc A, n"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let rst_18 = Instruction {
    execute: rst_n,
    cycles: 16,
    text: format!("rst 18H"),
    data: InstructionData::rst_n(0x18),
  };

  let ld_ff00_a = Instruction {
    execute: ld_ff00_imm_r8,
    cycles: 12,
    text: format!("ld (FF00 + n), A"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let pop_hl = Instruction {
    execute: pop_wide_register,
    cycles: 12,
    text: format!("pop HL"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let ld_ff00_c_a = Instruction {
    execute: ld_ff00_r8_r8,
    cycles: 8,
    text: format!("ld (FF00 + C), A"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let push_hl = Instruction {
    execute: push_wide_register,
    cycles: 16,
    text: format!("push HL"),
    data: InstructionData::wide_dst(WideRegister::HL),
  };

  let and_a_n = Instruction {
    execute: and_r8_n,
    cycles: 8,
    text: format!("and A, n"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let rst_20 = Instruction {
    execute: rst_n,
    cycles: 16,
    text: format!("rst 20H"),
    data: InstructionData::rst_n(0x20),
  };

  let add_sp_d = Instruction {
    execute: add_r16_n,
    cycles: 16,
    text: format!("add SP, n"),
    data: InstructionData::wide_dst(WideRegister::SP),
  };

  let jmp_indirect_hl = Instruction {
    execute: jump_wide_reg,
    cycles: 4,
    text: format!("jmp (HL)"),
    data: InstructionData::wide_dst(WideRegister::HL).with_flag(0, 0),
  };

  let ld_nn_a = Instruction {
    execute: load_indirect_nn_small_register,
    cycles: 16,
    text: format!("ld (NN), A"),
    data: InstructionData::small_src(SmallWidthRegister::A),
  };

  let xor_a_n = Instruction {
    execute: xor_r8_n,
    cycles: 8,
    text: format!("xor A, n"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let rst_28 = Instruction {
    execute: rst_n,
    cycles: 16,
    text: format!("rst 28H"),
    data: InstructionData::rst_n(0x28),
  };

  let ld_a_ff00 = Instruction {
    execute: ld_r8_ff00_imm,
    cycles: 8,
    text: format!("ld A, (FF00 + n)"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let pop_af = Instruction {
    execute: pop_wide_register,
    cycles: 12,
    text: format!("pop AF"),
    data: InstructionData::wide_dst(WideRegister::AF),
  };

  let ld_a_ff00_c = Instruction {
    execute: ld_r8_ff00_r8,
    cycles: 8,
    text: format!("ld A, (FF00 + C)"),
    data: InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C),
  };

  let di = Instruction {
    execute: disable_interrupts,
    cycles: 4,
    text: format!("DI"),
    data: InstructionData::default(),
  };

  let push_af = Instruction {
    execute: push_wide_register,
    cycles: 16,
    text: format!("push AF"),
    data: InstructionData::wide_dst(WideRegister::AF),
  };

  let or_a_n = Instruction {
    execute: or_r8_n,
    cycles: 8,
    text: format!("or A, n"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let rst_30 = Instruction {
    execute: rst_n,
    cycles: 16,
    text: format!("rst 30H"),
    data: InstructionData::rst_n(0x30),
  };

  let ld_hl_sp_d = Instruction {
    execute: ld_r16_r16_plus_n,
    cycles: 12,
    text: format!("ld HL, SP + d"),
    data: InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::SP),
  };

  let ld_hl_sp = Instruction {
    execute: ld_r16_r16,
    cycles: 8,
    text: format!("ld SP, HL"),
    data: InstructionData::wide_dst_wide_src(WideRegister::SP, WideRegister::HL),
  };

  let ld_a_nn = Instruction {
    execute: ld_r8_indirect_imm,
    cycles: 16,
    text: format!("ld A, (nn)"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let ei = Instruction {
    execute: enable_interrupts,
    cycles: 4,
    text: format!("EI"),
    data: InstructionData::default(),
  };

  let cp_a_n = Instruction {
    execute: cp_r8_n,
    cycles: 8,
    text: format!("cp A, n"),
    data: InstructionData::small_dst(SmallWidthRegister::A),
  };

  let rst_38 = Instruction {
    execute: rst_n,
    cycles: 16,
    text: format!("rst 38H"),
    data: InstructionData::rst_n(0x38),
  };

  vec![
    no_op,
    load_imm_bc,
    load_bc_a,
    inc_bc,
    inc_b,
    dec_b,
    load_imm_b,
    rlca,
    ld_nn_sp,
    add_hl_bc,
    ld_a_bc,
    dec_bc,
    inc_c,
    dec_c,
    ld_c_n,
    rrca,
    stop,
    load_imm_de,
    load_mem_de_a,
    inc_de,
    inc_d,
    dec_d,
    ld_d_n,
    rla,
    jr_n,
    add_hl_de,
    ld_a_de,
    dec_de,
    inc_e,
    dec_e,
    ld_e_n,
    rra,
    jr_nz_n,
    load_imm_hl,
    ldi_hl_a,
    inc_hl,
    inc_h,
    dec_h,
    ld_h_n,
    daa,
    jr_z_n,
    add_hl_hl,
    ldi_a_hl,
    dec_hl,
    inc_l,
    dec_l,
    ld_l_n,
    cpl,
    jr_nc_n,
    load_imm_sp,
    ldd_hl_a,
    inc_sp,
    inc_mem_hl,
    dec_mem_hl,
    ld_mem_hl_n,
    scf,
    jr_c_n,
    add_hl_sp,
    ldd_a_hl,
    dec_sp,
    inc_a,
    dec_a,
    ld_a_n,
    ccf,
    ld_b_b,
    ld_b_c,
    ld_b_d,
    ld_b_e,
    ld_b_h,
    ld_b_l,
    ld_b_hl,
    ld_b_a,
    ld_c_b,
    ld_c_c,
    ld_c_d,
    ld_c_e,
    ld_c_h,
    ld_c_l,
    ld_c_hl,
    ld_c_a,
    ld_d_b,
    ld_d_c,
    ld_d_d,
    ld_d_e,
    ld_d_h,
    ld_d_l,
    ld_d_hl,
    ld_d_a,
    ld_e_b,
    ld_e_c,
    ld_e_d,
    ld_e_e,
    ld_e_h,
    ld_e_l,
    ld_e_hl,
    ld_e_a,
    ld_h_b,
    ld_h_c,
    ld_h_d,
    ld_h_e,
    ld_h_h,
    ld_h_l,
    ld_h_hl,
    ld_h_a,
    ld_l_b,
    ld_l_c,
    ld_l_d,
    ld_l_e,
    ld_l_h,
    ld_l_l,
    ld_l_hl,
    ld_l_a,
    load_hl_b,
    load_hl_c,
    load_hl_d,
    load_hl_e,
    load_hl_h,
    load_hl_l,
    halt,
    load_hl_a,
    ld_a_b,
    ld_a_c,
    ld_a_d,
    ld_a_e,
    ld_a_h,
    ld_a_l,
    ld_a_hl,
    ld_a_a,
    add_a_b,
    add_a_c,
    add_a_d,
    add_a_e,
    add_a_h,
    add_a_l,
    add_a_hl,
    add_a_a,
    adc_a_b,
    adc_a_c,
    adc_a_d,
    adc_a_e,
    adc_a_h,
    adc_a_l,
    adc_a_hl,
    adc_a_a,
    sub_a_b,
    sub_a_c,
    sub_a_d,
    sub_a_e,
    sub_a_h,
    sub_a_l,
    sub_a_hl,
    sub_a_a,
    sbc_a_b,
    sbc_a_c,
    sbc_a_d,
    sbc_a_e,
    sbc_a_h,
    sbc_a_l,
    sbc_a_hl,
    sbc_a_a,
    and_a_b,
    and_a_c,
    and_a_d,
    and_a_e,
    and_a_h,
    and_a_l,
    and_a_hl,
    and_a_a,
    xor_a_b,
    xor_a_c,
    xor_a_d,
    xor_a_e,
    xor_a_h,
    xor_a_l,
    xor_a_hl,
    xor_a_a,
    or_a_b,
    or_a_c,
    or_a_d,
    or_a_e,
    or_a_h,
    or_a_l,
    or_a_hl,
    or_a_a,
    cp_a_b,
    cp_a_c,
    cp_a_d,
    cp_a_e,
    cp_a_h,
    cp_a_l,
    cp_a_hl,
    cp_a_a,
    ret_n_z,
    pop_bc,
    jnz,
    jmp,
    callnz,
    push_bc,
    add_a_n,
    rst_0,
    ret_z,
    ret_from_fn,
    jz,
    escape,
    callz,
    call,
    adc_a_n,
    rst_8,
    ret_n_c,
    pop_de,
    jnc,
    invalid.clone(),
    callnc,
    push_de,
    sub_a_imm,
    rst_10,
    ret_c,
    ret_i,
    jc,
    invalid.clone(),
    callc,
    invalid.clone(),
    sbc_a_n,
    rst_18,
    ld_ff00_a,
    pop_hl,
    ld_ff00_c_a,
    invalid.clone(),
    invalid.clone(),
    push_hl,
    and_a_n,
    rst_20,
    add_sp_d,
    jmp_indirect_hl,
    ld_nn_a,
    invalid.clone(),
    invalid.clone(),
    invalid.clone(),
    xor_a_n,
    rst_28,
    ld_a_ff00,
    pop_af,
    ld_a_ff00_c,
    di,
    invalid.clone(),
    push_af,
    or_a_n,
    rst_30,
    ld_hl_sp_d,
    ld_hl_sp,
    ld_a_nn,
    ei,
    invalid.clone(),
    invalid.clone(),
    cp_a_n,
    rst_38,
  ]
}

fn rlc_core(current: u8, registers: &mut Registers) -> u8 {
  let new_reg = current << 1 | current >> 7;
  registers.set_flags(new_reg == 0, false, false, current & (1 << 7) != 0);
  new_reg
}

/// RLC in the extended set
fn ext_rlc_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  let current = registers.read_r8(additional.small_reg_dst);
  let result = rlc_core(current, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

fn ext_rlc_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = rlc_core(current, registers);
  memory.write_u8(address, result);
}

fn rrc_core(current: u8, registers: &mut Registers) -> u8 {
  let new_reg = current << 7 | current >> 1;
  registers.set_flags(new_reg == 0, false, false, current & 1 != 0);
  new_reg
}

/// RRC in the extended set
fn ext_rrc_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  let current = registers.read_r8(additional.small_reg_dst);
  let result = rrc_core(current, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

fn ext_rrc_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = rrc_core(current, registers);
  memory.write_u8(address, result);
}

fn core_rl(reg: u8, registers: &mut Registers) -> u8 {
  let new_reg = (reg << 1) | if registers.carry() { 1 } else { 0 };
  registers.set_flags(new_reg == 0, false, false, reg & (1 << 7) != 0);
  new_reg
}

/// RL in the extended set
fn rl_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let reg = registers.read_r8(additional.small_reg_dst);
  let new_reg = core_rl(reg, registers);
  registers.write_r8(additional.small_reg_dst, new_reg);
}

fn ext_rl_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = core_rl(current, registers);
  memory.write_u8(address, result);
}

fn core_rr(reg: u8, registers: &mut Registers) -> u8 {
  let new_reg = (reg >> 1) | if registers.carry() { 1 << 7 } else { 0 };
  registers.set_flags(new_reg == 0, false, false, reg & 1 != 0);
  new_reg
}

/// RR in the extended set
fn ext_rr_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let reg = registers.read_r8(additional.small_reg_dst);
  let new_reg = core_rr(reg, registers);
  registers.write_r8(additional.small_reg_dst, new_reg);
}

fn ext_rr_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = core_rr(current, registers);
  memory.write_u8(address, result);
}

fn core_sla(reg: u8, registers: &mut Registers) -> u8 {
  let new_reg = reg << 1;
  registers.set_flags(new_reg == 0, false, false, reg & (1 << 7) != 0);
  new_reg
}

/// SLA in the extended set
fn ext_sla_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let reg = registers.read_r8(additional.small_reg_dst);
  let new_reg = core_sla(reg, registers);
  registers.write_r8(additional.small_reg_dst, new_reg);
}

fn ext_sla_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = core_sla(current, registers);
  memory.write_u8(address, result);
}

fn core_sra(reg: u8, registers: &mut Registers) -> u8 {
  // For some reason in SRA the most significant bit (0x80, 128) is ignored from the calculation.
  let new_reg = (reg & 0x80) | reg >> 1;
  registers.set_flags(new_reg == 0, false, false, isset8(reg, 0x1));
  new_reg
}

/// SRA in the extended set
fn ext_sra_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let reg = registers.read_r8(additional.small_reg_dst);
  let new_reg = core_sra(reg, registers);
  registers.write_r8(additional.small_reg_dst, new_reg);
}

fn ext_sra_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = core_sra(current, registers);
  memory.write_u8(address, result);
}

fn core_swap(reg: u8, registers: &mut Registers) -> u8 {
  let result = (reg >> 4) | ((reg & 0xF) << 4);
  registers.set_flags(result == 0, false, false, false);
  result
}

/// SWAP in the extended set
fn ext_swap_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let r1 = registers.read_r8(additional.small_reg_dst);
  let result = core_swap(r1, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

fn ext_swap_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = core_swap(current, registers);
  memory.write_u8(address, result);
}

fn srl_core(current: u8, registers: &mut Registers) -> u8 {
  registers.set_flags(current >> 1 == 0, false, false, current & 0x1 == 0x1);
  current >> 1
}

/// SRL in the extended set
fn ext_srl_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  let current = registers.read_r8(additional.small_reg_dst);
  let result = srl_core(current, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

fn ext_srl_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = srl_core(current, registers);
  memory.write_u8(address, result);
  registers.inc_pc(1);
}

fn bit_core(current: u8, bit: u8, registers: &mut Registers) {
  let selected_bit = 1 << bit;
  let target_register = current;
  registers.set_flags(
    selected_bit & target_register == 0,
    false,
    true,
    registers.carry(),
  );
}

/// BIT in the extended set
fn ext_bit_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let current = registers.read_r8(additional.small_reg_dst);
  bit_core(current, additional.bit, registers);
}

fn ext_bit_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  bit_core(current, additional.bit, registers);
}

fn res_core(current: u8, bit: u8, _registers: &mut Registers) -> u8 {
  let selected_bit = 1 << bit;
  current & (!selected_bit)
}

/// RES in the extended set
fn ext_res_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let target = registers.read_r8(additional.small_reg_dst);
  let result = res_core(target, additional.bit, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

fn ext_res_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = res_core(current, additional.bit, registers);
  memory.write_u8(address, result);
}

fn set_core(current: u8, bit: u8, _registers: &mut Registers) -> u8 {
  let selected_bit = 1 << bit;
  selected_bit | current
}

/// SET in the extended set
fn ext_set_r8(registers: &mut Registers, _memory: &mut MemoryPtr, additional: &InstructionData) {
  registers.inc_pc(1);
  let target = registers.read_r8(additional.small_reg_dst);
  let result = set_core(target, additional.bit, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

fn ext_set_indirect_r16(
  registers: &mut Registers,
  memory: &mut MemoryPtr,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = set_core(current, additional.bit, registers);
  memory.write_u8(address, result);
}

/// Instruction list when generating a row
fn next_instr_register(index: usize) -> SmallWidthRegister {
  match index {
    0 => SmallWidthRegister::B,
    1 => SmallWidthRegister::C,
    2 => SmallWidthRegister::D,
    3 => SmallWidthRegister::E,
    4 => SmallWidthRegister::H,
    5 => SmallWidthRegister::L,
    6 => SmallWidthRegister::B, /* Should never be selected, is the indirect slot */
    7 => SmallWidthRegister::A,
    _ => {
      unimplemented!();
    }
  }
}

/// The extended set is a systematic. This produces an 8-instruction row from it
fn make_extended_row(
  normal_skeleton: Instruction,
  indirect_skeleton: Instruction,
) -> Vec<Instruction> {
  let mut res = Vec::new();
  for i in 0..8 {
    if i == 6 {
      let mut next_instr = indirect_skeleton.clone();
      next_instr.text = format!("{} (HL)", indirect_skeleton.text);
      next_instr.data.wide_reg_dst = WideRegister::HL;
      res.push(next_instr);
    } else {
      let mut next_instr = normal_skeleton.clone();
      let next_instr_reg = next_instr_register(i);
      next_instr.text = format!("{} {:?}", indirect_skeleton.text, next_instr_reg);
      next_instr.data.small_reg_dst = next_instr_reg;
      res.push(next_instr);
    }
  }
  res
}

/// Make bit-set of extended rows. This enumerates the instructions for every bit on every register for the BIT, RES, and SET instructions
fn make_bit_set(normal_skeleton: Instruction, indirect_skeleton: Instruction) -> Vec<Instruction> {
  (0..8)
    .map(|bit| {
      let mut normal_skeleton = normal_skeleton.clone();
      let mut indirect_skeleton = indirect_skeleton.clone();
      normal_skeleton.data = normal_skeleton.data.with_bit(bit);
      normal_skeleton.text = format!("{} {}, ", normal_skeleton.text, bit);
      indirect_skeleton.data = indirect_skeleton.data.with_bit(bit);
      indirect_skeleton.text = format!("{} {}, ", indirect_skeleton.text, bit);
      make_extended_row(normal_skeleton, indirect_skeleton)
    })
    .flatten()
    .collect()
}

pub fn extended_instruction_set() -> Vec<Instruction> {
  // RLC
  let rlc_r8_skeleton = Instruction {
    execute: ext_rlc_r8,
    cycles: 8,
    text: format!("RLC "),
    data: InstructionData::default(),
  };

  let rlc_indirect_skeleton = Instruction {
    execute: ext_rlc_indirect_r16,
    cycles: 16,
    text: format!("RLC "),
    data: InstructionData::default(),
  };

  let rlc_row = make_extended_row(rlc_r8_skeleton, rlc_indirect_skeleton);

  // RRC
  let rrc_r8_skeleton = Instruction {
    execute: ext_rrc_r8,
    cycles: 8,
    text: format!("RRC "),
    data: InstructionData::default(),
  };

  let rrc_indirect_skeleton = Instruction {
    execute: ext_rrc_indirect_r16,
    cycles: 16,
    text: format!("RRC "),
    data: InstructionData::default(),
  };

  let rrc_row = make_extended_row(rrc_r8_skeleton, rrc_indirect_skeleton);

  // RL
  let rl_r8_skeleton = Instruction {
    execute: rl_r8,
    cycles: 8,
    text: format!("RL "),
    data: InstructionData::default(),
  };

  let rl_indirect_skeleton = Instruction {
    execute: ext_rl_indirect_r16,
    cycles: 16,
    text: format!("RL "),
    data: InstructionData::default(),
  };

  let rl_row = make_extended_row(rl_r8_skeleton, rl_indirect_skeleton);

  // RR
  let rr_r8_skeleton = Instruction {
    execute: ext_rr_r8,
    cycles: 8,
    text: format!("RR "),
    data: InstructionData::default(),
  };

  let rr_indirect_skeleton = Instruction {
    execute: ext_rr_indirect_r16,
    cycles: 16,
    text: format!("RR "),
    data: InstructionData::default(),
  };

  let rr_row = make_extended_row(rr_r8_skeleton, rr_indirect_skeleton);

  // SLA
  let sla_r8_skeleton = Instruction {
    execute: ext_sla_r8,
    cycles: 8,
    text: format!("SLA "),
    data: InstructionData::default(),
  };

  let sla_indirect_skeleton = Instruction {
    execute: ext_sla_indirect_r16,
    cycles: 16,
    text: format!("SLA "),
    data: InstructionData::default(),
  };

  let sla_row = make_extended_row(sla_r8_skeleton, sla_indirect_skeleton);

  // SRA
  let sra_r8_skeleton = Instruction {
    execute: ext_sra_r8,
    cycles: 8,
    text: format!("SRA "),
    data: InstructionData::default(),
  };

  let sra_indirect_skeleton = Instruction {
    execute: ext_sra_indirect_r16,
    cycles: 16,
    text: format!("SRA "),
    data: InstructionData::default(),
  };

  let sra_row = make_extended_row(sra_r8_skeleton, sra_indirect_skeleton);

  // SWAP
  let swap_r8_skeleton = Instruction {
    execute: ext_swap_r8,
    cycles: 8,
    text: format!("SWAP "),
    data: InstructionData::default(),
  };

  let swap_indirect_skeleton = Instruction {
    execute: ext_swap_indirect_r16,
    cycles: 16,
    text: format!("SWAP "),
    data: InstructionData::default(),
  };

  let swap_row = make_extended_row(swap_r8_skeleton, swap_indirect_skeleton);

  // SRL
  let srl_r8_skeleton = Instruction {
    execute: ext_srl_r8,
    cycles: 8,
    text: format!("SRL "),
    data: InstructionData::default(),
  };

  let srl_indirect_skeleton = Instruction {
    execute: ext_srl_indirect_r16,
    cycles: 16,
    text: format!("SRL "),
    data: InstructionData::default(),
  };

  let srl_row = make_extended_row(srl_r8_skeleton, srl_indirect_skeleton);

  // BIT
  let bit_r8_skeleton = Instruction {
    execute: ext_bit_r8,
    cycles: 8,
    text: format!("BIT "),
    data: InstructionData::default(),
  };

  let bit_indirect_skeleton = Instruction {
    execute: ext_bit_indirect_r16,
    cycles: 16,
    text: format!("BIT "),
    data: InstructionData::default(),
  };

  let bits_row = make_bit_set(bit_r8_skeleton, bit_indirect_skeleton);

  // RES
  let res_r8_skeleton = Instruction {
    execute: ext_res_r8,
    cycles: 8,
    text: format!("RES "),
    data: InstructionData::default(),
  };

  let res_indirect_skeleton = Instruction {
    execute: ext_res_indirect_r16,
    cycles: 16,
    text: format!("RES "),
    data: InstructionData::default(),
  };

  let rst_row = make_bit_set(res_r8_skeleton, res_indirect_skeleton);

  // SET
  let set_r8_skeleton = Instruction {
    execute: ext_set_r8,
    cycles: 8,
    text: format!("SET "),
    data: InstructionData::default(),
  };

  let set_indirect_skeleton = Instruction {
    execute: ext_set_indirect_r16,
    cycles: 16,
    text: format!("SET "),
    data: InstructionData::default(),
  };

  let set_row = make_bit_set(set_r8_skeleton, set_indirect_skeleton);

  rlc_row
    .iter()
    .chain(rrc_row.iter())
    .chain(rl_row.iter())
    .chain(rr_row.iter())
    .chain(sla_row.iter())
    .chain(sra_row.iter())
    .chain(swap_row.iter())
    .chain(srl_row.iter())
    .chain(bits_row.iter())
    .chain(rst_row.iter())
    .chain(set_row.iter())
    .cloned()
    .collect()
}
