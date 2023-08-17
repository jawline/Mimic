use crate::cpu::{Registers, CARRY_FLAG, ZERO_FLAG};
use crate::instruction_data::InstructionData;
use crate::memory::GameboyState;
use crate::memory::{isset16, isset32, isset8};
use crate::register::{SmallWidthRegister, WideRegister};
use crate::util::{
  carries_add8, carries_add8_with_carry, carries_sub8_with_carry, half_carry_add8, half_carry_sub8,
};
use log::trace;

pub struct InstructionSet {
  pub instructions: Vec<Instruction>,
  pub ext_instructions: Vec<Instruction>,
}

impl InstructionSet {
  pub fn new() -> Self {
    Self {
      instructions: instruction_set(),
      ext_instructions: extended_instruction_set(),
    }
  }
}

/// The instruction struct contains the implementation of and metadata on an instruction
#[derive(Clone)]
pub struct Instruction {
  pub execute: fn(registers: &mut Registers, memory: &mut GameboyState),
  pub cycles: u16,
  pub text: String,
}

/// No-op just increments the stack pointer
pub fn no_op(registers: &mut Registers, _memory: &mut GameboyState, _additional: &InstructionData) {
  registers.inc_pc(1);
}

/// Load immediate loads a 16 bit value following this instruction places it in a register
pub fn ld_imm_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
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
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let val = memory.read_u8(registers.pc() + 1);
  registers.inc_pc(2);
  let addr = registers.read_r16(additional.wide_reg_dst);
  memory.write_u8(addr, val, registers);
}

/// Load immediate loads a 8 bit value following this instruction places it in a small register
pub fn ld_imm_r8(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  //Load the 8 bit value after the opcode and store it to the dst register
  let imm_val = memory.read_u8(registers.pc() + 1);

  registers.write_r8(additional.small_reg_dst, imm_val);

  //Increment the PC by three once finished
  registers.inc_pc(2);
}

/// Write the value of small register one to the address pointed to by wide_reg_dst
pub fn ld_reg8_mem_reg16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  // Store the provided 8-bit register to the location pointed to by the 16-bit dst register
  let reg_val = registers.read_r8(additional.small_reg_one);
  let mem_dst = registers.read_r16(additional.wide_reg_dst);
  memory.write_u8(mem_dst, reg_val, registers);

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Replace the value of small_reg_dst with the value of small_reg_one
pub fn ld_r8_r8(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  additional: &InstructionData,
) {
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
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let target_addr = memory.read_u16(registers.pc() + 1);
  registers.inc_pc(3);
  registers.write_r8(additional.small_reg_dst, memory.read_u8(target_addr));
}

/// Increment the value of a wide-register by one
pub fn inc_wide_register(
  registers: &mut Registers,
  _memory: &mut GameboyState,
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
  _memory: &mut GameboyState,
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
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);

  let addr = registers.read_r16(additional.wide_reg_dst);
  let l = memory.read_u8(addr);
  let result = l + 1;

  // Increment by one and modify memory
  memory.write_u8(addr, result, registers);
  registers.set_flags(result == 0, false, half_carry_add8(l, 1), registers.carry());
}

/// Decrement the value of memory pointed by a wide register by one
/// and write it back to the same location in memory
pub fn dec_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let addr = registers.read_r16(additional.wide_reg_dst);
  let l = memory.read_u8(addr);
  let result = l - 1;

  // Increment by one and modify memory
  memory.write_u8(addr, result, registers);

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
fn add_r8_n(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = memory.read_u8(registers.pc() + 1);

  // We calculate the registers here since its shared with other adds
  let result = add_core(acc, operand, registers);
  registers.write_r8(additional.small_reg_dst, result);

  // Increment the PC by one once finished
  registers.inc_pc(2);
}

/// Add two small registers (small_reg_one to small_reg_dst)
fn add_r8_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
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
fn sub_r8_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = registers.read_r8(additional.small_reg_one);

  let result = sub_core(acc, operand, registers);
  registers.write_r8(additional.small_reg_dst, result);

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Subtract an immediate from a small dst register
fn sub_r8_n(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
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
fn and_r8_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = registers.read_r8(additional.small_reg_one);
  let result = and_core(acc, operand, registers);

  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// And imm with small reg dst
fn and_r8_n(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
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
fn or_r8_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = registers.read_r8(additional.small_reg_one);

  let result = or_core(acc, operand, registers);
  registers.write_r8(additional.small_reg_dst, result);

  registers.inc_pc(1);
}

/// bitwise or small reg dst with immediate
fn or_r8_n(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = memory.read_u8(registers.pc() + 1);
  let result = or_core(acc, operand, registers);

  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(2);
}

/// cp two small registers
fn cp_r8_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = registers.read_r8(additional.small_reg_one);

  // We discard the result but keep the changes to flags
  sub_core(acc, operand, registers);

  registers.inc_pc(1);
}

/// cp small reg dst against an immediate
fn cp_r8_n(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
  let acc = registers.read_r8(additional.small_reg_dst);
  let operand = memory.read_u8(registers.pc() + 1);
  sub_core(acc, operand, registers);
  registers.inc_pc(2);
}

/// Implement the core xor logic for 8 bit values
fn xor_core(v1: u8, v2: u8, registers: &mut Registers) -> u8 {
  let result = v1 ^ v2;
  registers.set_flags(result == 0, false, false, false);
  result
}

/// XOR two small registers
fn xor_r8_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let v_src = registers.read_r8(additional.small_reg_one);
  let v_dst = registers.read_r8(additional.small_reg_dst);
  let result = xor_core(v_dst, v_src, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// XOR small dst register with immediate
fn xor_r8_n(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
  let v_src = memory.read_u8(registers.pc() + 1);
  let v_dst = registers.read_r8(additional.small_reg_dst);
  let result = xor_core(v_dst, v_src, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(2);
}

/// The core logic for subtraction of 8-bit values through a carry
fn sbc_core(acc: u8, val: u8, registers: &mut Registers) -> u8 {
  let carry = registers.carry();
  sub_core_with_carry(acc, val, registers, carry)
}

/// Subtract through carry using two small registers (small_reg_one to small_reg_dst)
fn sbc_r8_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let v_src = registers.read_r8(additional.small_reg_one);
  let v_dst = registers.read_r8(additional.small_reg_dst);
  let result = sbc_core(v_dst, v_src, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// Subtract immediate through carry
fn sbc_r8_n(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let sub_v = memory.read_u8(registers.pc() + 1);
  let result = sbc_core(origin, sub_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(2);
}

/// Load small reg to 0xFF00 + n
fn ld_ff00_imm_r8(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  // Fetch the immediate offset
  let add_v: u16 = memory.read_u8(registers.pc() + 1).into();

  let rval = registers.read_r8(additional.small_reg_dst);
  memory.write_u8(0xFF00 + add_v, rval, registers);
  registers.inc_pc(2);
}

/// Load (0xFF00 + n) to small reg
// TODO: This and its dual function should have reversed names
fn ld_r8_ff00_imm(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  // Fetch
  let add_v: u16 = memory.read_u8(registers.pc() + 1).into();

  let read_value = memory.read_u8(0xFF00 + add_v);
  registers.write_r8(additional.small_reg_dst, read_value);
  registers.inc_pc(2);
}

/// Load small reg dst to 0xFF00 + small reg one
fn ld_ff00_r8_r8(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let addr_offset: u16 = registers.read_r8(additional.small_reg_one) as u16;
  let rval = registers.read_r8(additional.small_reg_dst);
  memory.write_u8(0xFF00 + addr_offset, rval, registers);
  registers.inc_pc(1);
}

/// Load 0xFF00 + small reg one into small_reg_dst
// TODO: This function and it's dual should have their names reversed
fn ld_r8_ff00_r8(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let addr_offset: u16 = registers.read_r8(additional.small_reg_one).into();
  let rval = memory.read_u8(0xFF00 + addr_offset);
  registers.write_r8(additional.small_reg_dst, rval);
  registers.inc_pc(1);
}

/// Generic add with carry, sets flags and returns result
fn adc_generic(acc: u8, r: u8, registers: &mut Registers) -> u8 {
  let result = acc + r + if registers.carry() { 1 } else { 0 };
  let (half_carry, carry) = carries_add8_with_carry(acc, r, registers.carry());
  registers.set_flags(result == 0, false, half_carry, carry);
  result
}

/// Add with carry an immediate to a small register
fn adc_r8_imm(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
  let add_v = memory.read_u8(registers.pc() + 1);
  let origin = registers.read_r8(additional.small_reg_dst);
  let result = adc_generic(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(2);
}

/// Add two small registers (small_reg_one to small_reg_dst)
/// Also add one if the carry flag is set
fn adc_r8_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let add_v = registers.read_r8(additional.small_reg_one);
  let result = adc_generic(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// Add value at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn add_r8_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);

  let result = add_core(origin, add_v, registers);

  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// Subtract value add wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn sub_r8_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);
  let result = sub_core(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// Subtract through carry value at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn sbc_r8_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);

  let result = sbc_core(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// And memory at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn and_r8_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);

  let result = and_core(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// Cp memory at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn cp_r8_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let target_addr = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(target_addr);
  let origin = registers.read_r8(additional.small_reg_dst);
  sub_core(origin, add_v, registers);
  registers.inc_pc(1);
}

/// or memory at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn or_r8_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);
  let result = or_core(origin, add_v, registers);

  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// XOR memory at wide_register_one in memory to small_reg_dst
/// save the result in small_reg_dst
fn xor_r8_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);

  let result = xor_core(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// Add value add wide_register_one in memory to small_reg_dst
/// Add a further 1 if the carry is set
/// save the result in small_reg_dst
fn adc_r8_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let origin = registers.read_r8(additional.small_reg_dst);
  let address = registers.read_r16(additional.wide_reg_one);
  let add_v = memory.read_u8(address);
  let result = adc_generic(origin, add_v, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

/// Rotate 8-bit register left, placing whatever is in bit 7 in the carry bit before
fn rotate_left_with_carry(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let mut a = registers.read_r8(additional.small_reg_dst);
  let carry = a & (1 << 7) != 0;
  a = a.rotate_left(1);
  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(false, false, false, carry);
  registers.inc_pc(1);
}

/// Rotate 8-bit register right, placing whatever is in bit 0 in the carry bit before
fn rotate_right_with_carry(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let mut a = registers.read_r8(additional.small_reg_dst);
  let carry = a & (1) != 0;
  a = a.rotate_right(1);
  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(false, false, false, carry);
  registers.inc_pc(1);
}

fn rotate_r8_left_through_carry(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let mut a = registers.read_r8(additional.small_reg_dst);

  let will_carry = a & (0x1 << 7) != 0;
  a = a << 1;

  if registers.carry() {
    a = a.wrapping_add(1);
  }

  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(false, false, false, will_carry);
  registers.inc_pc(1);
}

/// Rotate 8-bit register right using the carry bit as an additional bit.
/// In practice, store the current carry bit, set carry bit to value of bit 0
/// then shift right everything by one bit and then replace bit 7 with the origin
/// carry bit
fn rotate_r8_right_through_carry(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let mut a = registers.read_r8(additional.small_reg_dst);

  let will_carry = a & 0x1 != 0;
  a = a >> 1;

  if registers.carry() {
    a = a.wrapping_add(1 << 7);
  }

  registers.write_r8(additional.small_reg_dst, a);
  registers.set_flags(false, false, false, will_carry);
  registers.inc_pc(1);
}

/// Decrement the value of a small register by one
fn dec_wide_register(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  additional: &InstructionData,
) {
  // Increment the destination register by one
  registers.write_r16(
    additional.wide_reg_dst,
    registers.read_r16(additional.wide_reg_dst).wrapping_sub(1),
  );

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Decrement the value of a small register by one
fn dec_small_register(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let l = registers.read_r8(additional.small_reg_dst);
  let result = l.wrapping_sub(1);

  // Increment the destination register by one
  registers.write_r8(additional.small_reg_dst, result);

  registers.set_flags(result == 0, true, half_carry_sub8(l, 1), registers.carry());

  // Increment the PC by one once finished
  registers.inc_pc(1);
}

/// Write the contents of a wide register to an address in memory
fn load_immediate_wide_register(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  // Find the address we are writing to
  let load_address = memory.read_u16(registers.pc() + 1);

  // Write the contents of wide register one to that location
  memory.write_u16(
    load_address,
    registers.read_r16(additional.wide_reg_one),
    registers,
  );

  // Increment the PC by one once finished
  registers.inc_pc(3);
}

fn load_indirect_nn_small_register(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  // Find the address we are writing to
  let load_address = memory.read_u16(registers.pc() + 1);

  // Write the contents of wide register one to that location
  memory.write_u8(
    load_address,
    registers.read_r8(additional.small_reg_one),
    registers,
  );

  // Increment the PC by one once finished
  registers.inc_pc(3);
}

/// Place the value of a small register into the
/// memory address pointed to by the wide destination
/// register and then increment the wide destination register
/// Example, ld HL, 0 ld A, 5 ldi (HL), A will leave [0] = 5, A = 5, HL = 1
fn ldi_mem_r16_val_r8(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let wide_reg = registers.read_r16(additional.wide_reg_dst);
  let target = registers.read_r8(additional.small_reg_one);
  memory.write_u8(wide_reg, target, registers);
  registers.write_r16(additional.wide_reg_dst, wide_reg.wrapping_add(1));
  registers.inc_pc(1);
}

/// Place memory pointed to by the wide register into the small dst register
/// then increment the wide register
fn ldi_r8_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let wide_reg = registers.read_r16(additional.wide_reg_one);
  let mem = memory.read_u8(wide_reg);

  registers.write_r8(additional.small_reg_dst, mem);
  registers.write_r16(additional.wide_reg_one, wide_reg.wrapping_add(1));
  registers.inc_pc(1);
}

/// Like ldi but decrement
fn ldd_mem_r16_val_r8(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let wide_reg = registers.read_r16(additional.wide_reg_dst);
  let target = registers.read_r8(additional.small_reg_one);
  memory.write_u8(wide_reg, target, registers);
  registers.write_r16(additional.wide_reg_dst, wide_reg.wrapping_sub(1));
  registers.inc_pc(1);
}

/// Like ldi but decrement
fn ldd_r8_mem_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let wide_reg = registers.read_r16(additional.wide_reg_one);
  let mem = memory.read_u8(wide_reg);

  registers.write_r8(additional.small_reg_dst, mem);
  registers.write_r16(additional.wide_reg_one, wide_reg.wrapping_sub(1));
  registers.inc_pc(1);
}

/// Add a wide register to a wide register
fn add_r16_r16(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  additional: &InstructionData,
) {
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

fn register_plus_signed_8_bit_immediate(acc: i16, immediate: i8, registers: &mut Registers) -> i16 {
  let signed_immediate: i16 = ((immediate & 127) - (immediate & (-128))) as i16;
  let result = acc + signed_immediate;

  if signed_immediate >= 0 {
    let carry = ((acc & 0xFF) + signed_immediate) > 0xFF;
    let half_carry = ((acc & 0xF) + signed_immediate) > 0xF;
    registers.set_flags(false, false, half_carry, carry);
  } else {
    let carry = (result & 0xFF) <= (acc & 0xFF);
    let half_carry = (result & 0xF) <= (acc & 0xF);
    registers.set_flags(false, false, half_carry, carry);
  }

  result
}

/// Add an immediate byte (signed) to a wide register
fn add_r16_n(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
  let result = register_plus_signed_8_bit_immediate(
    registers.read_r16(additional.wide_reg_dst) as i16,
    memory.read_u8(registers.pc() + 1) as i8,
    registers,
  ) as u16;
  registers.write_r16(additional.wide_reg_dst, result);
  registers.inc_pc(2);
}

/// Load wide_reg_one into wide reg dst
fn ld_r16_r16(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  registers.inc_pc(1);
  registers.write_r16(
    additional.wide_reg_dst,
    registers.read_r16(additional.wide_reg_one),
  );
}

/// Add an immediate byte (signed) to wide reg one and then save it to wide reg dst
fn ld_r16_r16_plus_n(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let result = register_plus_signed_8_bit_immediate(
    registers.read_r16(additional.wide_reg_one) as i16,
    memory.read_u8(registers.pc() + 1) as i8,
    registers,
  ) as u16;

  registers.write_r16(additional.wide_reg_dst, result);
  registers.inc_pc(2);
}

/// Load a value from memory to a small register
fn load_r16_mem_to_r8(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let address = registers.read_r16(additional.wide_reg_one);
  let value = memory.read_u8(address);
  registers.write_r8(additional.small_reg_dst, value);

  registers.inc_pc(1);
}

/// Stop the processor & screen until button press
fn stop(registers: &mut Registers, _memory: &mut GameboyState, _additional: &InstructionData) {
  registers.inc_pc(2);
}

/// Escape
fn escape(registers: &mut Registers, _memory: &mut GameboyState, _additional: &InstructionData) {
  registers.inc_pc(1);
  registers.escaped = true;
}

/// Jump relative by a signed 8-bit value following the opcode
fn jump_relative_signed_immediate(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let byte = memory.read_u8(registers.pc() + 1) as i8;
  registers.inc_pc(2);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected {
    registers.cycles_elapsed_during_last_step += 4;
    registers.jump_relative(byte);
  }
}

/// Return if the flags & mask == expected
fn ret(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
  registers.inc_pc(1);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected {
    registers.cycles_elapsed_during_last_step += 12;
    let ret_pc = registers.stack_pop16(memory);
    registers.set_pc(ret_pc);
  }
}

/// Enable interrupts
fn enable_interrupts(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  _additional: &InstructionData,
) {
  trace!("Interrupts enabled");
  registers.inc_pc(1);
  registers.ime = true;
}

/// Disable interrupts
fn disable_interrupts(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  _additional: &InstructionData,
) {
  trace!("Interrupts disabled");
  registers.inc_pc(1);
  registers.ime = false;
}

/// Return and enable interrupts
fn reti(registers: &mut Registers, memory: &mut GameboyState, _additional: &InstructionData) {
  registers.ime = true;
  let ret_pc = registers.stack_pop16(memory);
  registers.set_pc(ret_pc);
}

/// Jump to destination if flags & mask == expected
fn jump_immediate(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let target_address = memory.read_u16(registers.pc() + 1);
  registers.inc_pc(3);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected {
    registers.cycles_elapsed_during_last_step += 4;
    registers.set_pc(target_address);
  }
}

/// Jump to the value stored in a wide register
fn jump_wide_reg(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let target_address = registers.read_r16(additional.wide_reg_dst);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected {
    registers.set_pc(target_address);
  }
}

/// Call function if flags & mask == expected
fn call_immediate(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let target_address = memory.read_u16(registers.pc() + 1);
  registers.inc_pc(3);
  if (registers.flags() & additional.flag_mask) == additional.flag_expected {
    registers.cycles_elapsed_during_last_step += 12;
    registers.stack_push16(registers.pc(), memory);
    registers.set_pc(target_address);
  }
}

/// DAA takes the result of an arithmetic operation and makes it binary coded
/// retrospectively
fn daa(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let mut acc = registers.read_r8(additional.small_reg_dst);
  let mut correction: u16 = if registers.carry() { 0x60 } else { 0 };
  let add_mode = !registers.subtract();

  if registers.half_carry() || (add_mode && ((acc & 0xF) > 9)) {
    correction |= 0x06;
  }

  if registers.carry() || (add_mode && ((acc & 0xFF) > 0x99)) {
    correction |= 0x60;
  }

  if add_mode {
    acc += correction as u8;
  } else {
    acc -= correction as u8;
  }

  let mut carry = registers.carry();

  if isset16(correction << 2, 0x100) {
    carry = true;
  }

  registers.write_r8(additional.small_reg_dst, acc);
  registers.set_flags(acc == 0, registers.subtract(), false, carry);
  registers.inc_pc(1);
}

fn invalid_op(
  _registers: &mut Registers,
  _memory: &mut GameboyState,
  _additional: &InstructionData,
) {
  unimplemented!();
}

/// Flip all bits in an r8
fn cpl_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
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
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let to_push = registers.read_r16(additional.wide_reg_dst);
  registers.stack_push16(to_push, memory);
}

/// Pop two bytes from the stack and store them in specified register
fn pop_wide_register(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let popped_value = registers.stack_pop16(memory);
  registers.write_r16(additional.wide_reg_dst, popped_value);
}

/// Halt the processor
/// Stop doing anything until the next interrupt
fn halt(registers: &mut Registers, _memory: &mut GameboyState, _additional: &InstructionData) {
  registers.halted = true;
  registers.inc_pc(1);
}

/// Sets the carry flag, resets negative and half carry flags, zero unaffected
fn scf(registers: &mut Registers, _memory: &mut GameboyState, _additional: &InstructionData) {
  registers.inc_pc(1);
  registers.set_flags(registers.zero(), false, false, true);
}

/// Complement the carry flag (flip it)
fn ccf(registers: &mut Registers, _memory: &mut GameboyState, _additional: &InstructionData) {
  registers.inc_pc(1);
  registers.set_flags(registers.zero(), false, false, !registers.carry());
}

/// Push current PC to stack then jump to n (8-bit immediate)
fn rst_n(registers: &mut Registers, memory: &mut GameboyState, additional: &InstructionData) {
  trace!("RST {:x}", additional.code);
  registers.inc_pc(1);
  registers.stack_push16(registers.pc(), memory);
  registers.set_pc(additional.code as u16);
}

/// This wrapper makes the instruction data to the generic implementations like add constant so
/// that the compiler should be able to inline and optimize them.
macro_rules! instr {
  ($name:expr, $cycles:expr, $method:ident, $additional:expr) => {{
    const INSTRUCTION_DATA: InstructionData = $additional;
    fn evaluate(registers: &mut Registers, memory: &mut GameboyState) {
      $method(registers, memory, &INSTRUCTION_DATA);
    }
    Instruction {
      execute: evaluate,
      cycles: $cycles,
      text: $name.to_string(),
    }
  }};
}

pub fn instruction_set() -> Vec<Instruction> {
  let no_op = instr!("nop", 4, no_op, InstructionData::const_default());
  let load_imm_bc = instr!(
    "ld BC, nn",
    12,
    ld_imm_r16,
    InstructionData::wide_dst(WideRegister::BC)
  );
  let load_bc_a = instr!(
    "ld (BC), a",
    8,
    ld_reg8_mem_reg16,
    InstructionData::wide_dst_small_in(WideRegister::BC, SmallWidthRegister::A)
  );
  let inc_bc = instr!(
    "inc BC",
    8,
    inc_wide_register,
    InstructionData::wide_dst(WideRegister::BC)
  );
  let inc_b = instr!(
    "inc B",
    4,
    inc_small_register,
    InstructionData::small_dst(SmallWidthRegister::B)
  );
  let dec_b = instr!(
    "dec B",
    4,
    dec_small_register,
    InstructionData::small_dst(SmallWidthRegister::B)
  );
  let load_imm_b = instr!(
    "ld B, n",
    8,
    ld_imm_r8,
    InstructionData::small_dst(SmallWidthRegister::B)
  );
  let rlca = instr!(
    "rlca",
    4,
    rotate_left_with_carry,
    InstructionData::small_dst(SmallWidthRegister::A)
  );
  let ld_nn_sp = instr!(
    "ld (nn), SP",
    20,
    load_immediate_wide_register,
    InstructionData::wide_src(WideRegister::SP)
  );
  let add_hl_bc = instr!(
    "add HL, BC",
    8,
    add_r16_r16,
    InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::BC)
  );
  let ld_a_bc = instr!(
    "ld A, (BC)",
    8,
    load_r16_mem_to_r8,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::BC)
  );
  let dec_bc = instr!(
    "dec BC",
    8,
    dec_wide_register,
    InstructionData::wide_dst(WideRegister::BC)
  );
  let inc_c = instr!(
    "inc C",
    4,
    inc_small_register,
    InstructionData::small_dst(SmallWidthRegister::C)
  );
  let dec_c = instr!(
    "dec C",
    4,
    dec_small_register,
    InstructionData::small_dst(SmallWidthRegister::C)
  );
  let ld_c_n = instr!(
    "ld C, n",
    8,
    ld_imm_r8,
    InstructionData::small_dst(SmallWidthRegister::C)
  );
  let rrca = instr!(
    "rrca",
    4,
    rotate_right_with_carry,
    InstructionData::small_dst(SmallWidthRegister::A)
  );
  let stop = instr!("stop", 4, stop, InstructionData::const_default());
  let load_imm_de = instr!(
    "ld DE, nn",
    12,
    ld_imm_r16,
    InstructionData::wide_dst(WideRegister::DE)
  );
  let load_mem_de_a = instr!(
    "ld (DE), A",
    8,
    ld_reg8_mem_reg16,
    InstructionData::wide_dst_small_in(WideRegister::DE, SmallWidthRegister::A)
  );
  let inc_de = instr!(
    "inc DE",
    8,
    inc_wide_register,
    InstructionData::wide_dst(WideRegister::DE)
  );
  let inc_d = instr!(
    "inc D",
    4,
    inc_small_register,
    InstructionData::small_dst(SmallWidthRegister::D)
  );
  let dec_d = instr!(
    "dec D",
    4,
    dec_small_register,
    InstructionData::small_dst(SmallWidthRegister::D)
  );
  let ld_d_n = instr!(
    "ld D, n",
    8,
    ld_imm_r8,
    InstructionData::small_dst(SmallWidthRegister::D)
  );
  let rla = instr!(
    "rla",
    4,
    rotate_r8_left_through_carry,
    InstructionData::small_dst(SmallWidthRegister::A)
  );
  let jr_n = instr!(
    "jr nn",
    8,
    jump_relative_signed_immediate,
    InstructionData::const_default().with_flag(0, 0)
  );
  let add_hl_de = instr!(
    "add HL, DE",
    8,
    add_r16_r16,
    InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::DE)
  );
  let ld_a_de = instr!(
    "ld A, (DE)",
    8,
    load_r16_mem_to_r8,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::DE)
  );
  let dec_de = instr!(
    "dec DE",
    8,
    dec_wide_register,
    InstructionData::wide_dst(WideRegister::DE)
  );
  let inc_e = instr!(
    "inc E",
    4,
    inc_small_register,
    InstructionData::small_dst(SmallWidthRegister::E)
  );
  let dec_e = instr!(
    "dec E",
    4,
    dec_small_register,
    InstructionData::small_dst(SmallWidthRegister::E)
  );
  let ld_e_n = instr!(
    "ld E, n",
    8,
    ld_imm_r8,
    InstructionData::small_dst(SmallWidthRegister::E)
  );
  let rra = instr!(
    "rra",
    4,
    rotate_r8_right_through_carry,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let jr_nz_n = instr!(
    "jrnz n",
    8,
    jump_relative_signed_immediate,
    InstructionData::const_default().with_flag(ZERO_FLAG, 0)
  );

  let load_imm_hl = instr!(
    "ld HL, nn",
    12,
    ld_imm_r16,
    InstructionData::wide_dst(WideRegister::HL)
  );

  let ldi_hl_a = instr!(
    "ldi (HL), A",
    8,
    ldi_mem_r16_val_r8,
    InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::A)
  );

  let inc_hl = instr!(
    "inc HL",
    8,
    inc_wide_register,
    InstructionData::wide_dst(WideRegister::HL)
  );

  let inc_h = instr!(
    "inc H",
    4,
    inc_small_register,
    InstructionData::small_dst(SmallWidthRegister::H)
  );

  let dec_h = instr!(
    "dec H",
    4,
    dec_small_register,
    InstructionData::small_dst(SmallWidthRegister::H)
  );

  let ld_h_n = instr!(
    "ld H, n",
    8,
    ld_imm_r8,
    InstructionData::small_dst(SmallWidthRegister::H)
  );

  let daa = instr!(
    "daa",
    4,
    daa,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let jr_z_n = instr!(
    "jr z n",
    8,
    jump_relative_signed_immediate,
    InstructionData::const_default().with_flag(ZERO_FLAG, ZERO_FLAG)
  );

  let add_hl_hl = instr!(
    "add HL, HL",
    8,
    add_r16_r16,
    InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::HL)
  );

  let ldi_a_hl = instr!(
    "ldi A, (HL)",
    8,
    ldi_r8_mem_r16,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let dec_hl = instr!(
    "dec HL",
    8,
    dec_wide_register,
    InstructionData::wide_dst(WideRegister::HL)
  );

  let inc_l = instr!(
    "inc L",
    4,
    inc_small_register,
    InstructionData::small_dst(SmallWidthRegister::L)
  );

  let dec_l = instr!(
    "dec L",
    4,
    dec_small_register,
    InstructionData::small_dst(SmallWidthRegister::L)
  );

  let ld_l_n = instr!(
    "ld L, n",
    8,
    ld_imm_r8,
    InstructionData::small_dst(SmallWidthRegister::L)
  );

  let cpl = instr!(
    "cpl",
    4,
    cpl_r8,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let jr_nc_n = instr!(
    "JRNC n",
    8,
    jump_relative_signed_immediate,
    InstructionData::const_default().with_flag(CARRY_FLAG, 0)
  );

  let load_imm_sp = instr!(
    "ld SP, nn",
    12,
    ld_imm_r16,
    InstructionData::wide_dst(WideRegister::SP)
  );

  let ldd_hl_a = instr!(
    "ldd (HL), A",
    8,
    ldd_mem_r16_val_r8,
    InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::A)
  );

  let inc_sp = instr!(
    "inc SP",
    8,
    inc_wide_register,
    InstructionData::wide_dst(WideRegister::SP)
  );

  let inc_mem_hl = instr!(
    "inc (HL)",
    12,
    inc_mem_r16,
    InstructionData::wide_dst(WideRegister::HL)
  );

  let dec_mem_hl = instr!(
    "dec (HL)",
    12,
    dec_mem_r16,
    InstructionData::wide_dst(WideRegister::HL)
  );

  let ld_mem_hl_n = instr!(
    "ld (HL), n",
    12,
    ld_mem_r16_immediate,
    InstructionData::wide_dst(WideRegister::HL)
  );

  let scf = instr!("SCF", 4, scf, InstructionData::const_default());

  let jr_c_n = instr!(
    "JRC n",
    8,
    jump_relative_signed_immediate,
    InstructionData::const_default().with_flag(CARRY_FLAG, CARRY_FLAG)
  );

  let add_hl_sp = instr!(
    "add HL, SP",
    8,
    add_r16_r16,
    InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::SP)
  );

  let ldd_a_hl = instr!(
    "ldd A, (HL)",
    8,
    ldd_r8_mem_r16,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let dec_sp = instr!(
    "dec SP",
    8,
    dec_wide_register,
    InstructionData::wide_dst(WideRegister::SP)
  );

  let inc_a = instr!(
    "inc A",
    4,
    inc_small_register,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let dec_a = instr!(
    "dec A",
    4,
    dec_small_register,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let ld_a_n = instr!(
    "ld A, n",
    8,
    ld_imm_r8,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let ccf = instr!("ccf", 4, ccf, InstructionData::const_default());

  let ld_b_b = instr!(
    "ld B, B",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::B)
  );

  let ld_b_c = instr!(
    "ld B, C",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::C)
  );

  let ld_b_d = instr!(
    "ld B, D",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::D)
  );

  let ld_b_e = instr!(
    "ld B, E",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::E)
  );

  let ld_b_h = instr!(
    "ld B, H",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::H)
  );

  let ld_b_l = instr!(
    "ld B, L",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::L)
  );

  let ld_b_hl = instr!(
    "ld B, (HL)",
    8,
    load_r16_mem_to_r8,
    InstructionData::small_dst_wide_src(SmallWidthRegister::B, WideRegister::HL)
  );

  let ld_b_a = instr!(
    "ld B, A",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::B, SmallWidthRegister::A)
  );

  let ld_c_b = instr!(
    "ld C, B",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::B)
  );

  let ld_c_c = instr!(
    "ld C, C",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::C)
  );

  let ld_c_d = instr!(
    "ld C, D",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::D)
  );

  let ld_c_e = instr!(
    "ld C, E",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::E)
  );

  let ld_c_h = instr!(
    "ld C, H",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::H)
  );

  let ld_c_l = instr!(
    "ld C, L",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::L)
  );

  let ld_c_hl = instr!(
    "ld C, (HL)",
    8,
    load_r16_mem_to_r8,
    InstructionData::small_dst_wide_src(SmallWidthRegister::C, WideRegister::HL)
  );

  let ld_c_a = instr!(
    "ld C, A",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::C, SmallWidthRegister::A)
  );

  let ld_d_b = instr!(
    "ld D, B",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::B)
  );

  let ld_d_c = instr!(
    "ld D, C",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::C)
  );

  let ld_d_d = instr!(
    "ld D, D",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::D)
  );

  let ld_d_e = instr!(
    "ld D, E",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::E)
  );

  let ld_d_h = instr!(
    "ld D, H",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::H)
  );

  let ld_d_l = instr!(
    "ld D, L",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::L)
  );

  let ld_d_hl = instr!(
    "ld D, (HL)",
    8,
    load_r16_mem_to_r8,
    InstructionData::small_dst_wide_src(SmallWidthRegister::D, WideRegister::HL)
  );

  let ld_d_a = instr!(
    "ld D, A",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::D, SmallWidthRegister::A)
  );

  let ld_e_b = instr!(
    "ld E, B",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::B)
  );

  let ld_e_c = instr!(
    "ld E, C",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::C)
  );

  let ld_e_d = instr!(
    "ld E, D",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::D)
  );

  let ld_e_e = instr!(
    "ld E, E",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::E)
  );

  let ld_e_h = instr!(
    "ld E, H",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::H)
  );

  let ld_e_l = instr!(
    "ld E, L",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::L)
  );

  let ld_e_hl = instr!(
    "ld E, (HL)",
    8,
    load_r16_mem_to_r8,
    InstructionData::small_dst_wide_src(SmallWidthRegister::E, WideRegister::HL)
  );

  let ld_e_a = instr!(
    "ld E, A",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::E, SmallWidthRegister::A)
  );

  let ld_h_b = instr!(
    "ld H, B",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::B)
  );

  let ld_h_c = instr!(
    "ld H, C",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::C)
  );

  let ld_h_d = instr!(
    "ld H, D",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::D)
  );

  let ld_h_e = instr!(
    "ld H, E",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::E)
  );

  let ld_h_h = instr!(
    "ld H, H",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::H)
  );

  let ld_h_l = instr!(
    "ld H, L",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::L)
  );

  let ld_h_hl = instr!(
    "ld H, (HL)",
    8,
    load_r16_mem_to_r8,
    InstructionData::small_dst_wide_src(SmallWidthRegister::H, WideRegister::HL)
  );

  let ld_h_a = instr!(
    "ld H, A",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::H, SmallWidthRegister::A)
  );

  let ld_l_b = instr!(
    "ld L, B",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::B)
  );

  let ld_l_c = instr!(
    "ld L, C",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::C)
  );

  let ld_l_d = instr!(
    "ld L, D",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::D)
  );

  let ld_l_e = instr!(
    "ld L, E",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::E)
  );

  let ld_l_h = instr!(
    "ld L, H",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::H)
  );

  let ld_l_l = instr!(
    "ld L, L",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::L)
  );

  let ld_l_hl = instr!(
    "ld L, (HL)",
    8,
    load_r16_mem_to_r8,
    InstructionData::small_dst_wide_src(SmallWidthRegister::L, WideRegister::HL)
  );

  let ld_l_a = instr!(
    "ld L, A",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::L, SmallWidthRegister::A)
  );

  let load_hl_b = instr!(
    "ld (HL), B",
    8,
    ld_reg8_mem_reg16,
    InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::B)
  );

  let load_hl_c = instr!(
    "ld (HL), C",
    8,
    ld_reg8_mem_reg16,
    InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::C)
  );

  let load_hl_d = instr!(
    "ld (HL), D",
    8,
    ld_reg8_mem_reg16,
    InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::D)
  );

  let load_hl_e = instr!(
    "ld (HL), E",
    8,
    ld_reg8_mem_reg16,
    InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::E)
  );

  let load_hl_h = instr!(
    "ld (HL), H",
    8,
    ld_reg8_mem_reg16,
    InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::H)
  );

  let load_hl_l = instr!(
    "ld (HL), L",
    8,
    ld_reg8_mem_reg16,
    InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::L)
  );

  let halt = instr!("halt", 4, halt, InstructionData::const_default());

  let load_hl_a = instr!(
    "ld (HL), A",
    8,
    ld_reg8_mem_reg16,
    InstructionData::wide_dst_small_in(WideRegister::HL, SmallWidthRegister::A)
  );

  let ld_a_b = instr!(
    "ld A, B",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B)
  );

  let ld_a_c = instr!(
    "ld A, C",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let ld_a_d = instr!(
    "ld A, D",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D)
  );

  let ld_a_e = instr!(
    "ld A, E",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E)
  );

  let ld_a_h = instr!(
    "ld A, H",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H)
  );

  let ld_a_l = instr!(
    "ld A, L",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L)
  );

  let ld_a_hl = instr!(
    "ld A, (HL)",
    8,
    load_r16_mem_to_r8,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let ld_a_a = instr!(
    "ld A, A",
    4,
    ld_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A)
  );

  let add_a_b = instr!(
    "add A, B",
    4,
    add_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B)
  );

  let add_a_c = instr!(
    "add A, C",
    4,
    add_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let add_a_d = instr!(
    "add A, D",
    4,
    add_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D)
  );

  let add_a_e = instr!(
    "add A, E",
    4,
    add_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E)
  );

  let add_a_h = instr!(
    "add A, H",
    4,
    add_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H)
  );

  let add_a_l = instr!(
    "add A, L",
    4,
    add_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L)
  );

  let add_a_hl = instr!(
    "add A, (HL)",
    8,
    add_r8_mem_r16,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let add_a_a = instr!(
    "add A, A",
    4,
    add_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A)
  );

  // Add with carries

  let adc_a_b = instr!(
    "adc A, B",
    4,
    adc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B)
  );

  let adc_a_c = instr!(
    "adc A, C",
    4,
    adc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let adc_a_d = instr!(
    "adc A, D",
    4,
    adc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D)
  );

  let adc_a_e = instr!(
    "adc A, E",
    4,
    adc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E)
  );

  let adc_a_h = instr!(
    "adc A, H",
    4,
    adc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H)
  );

  let adc_a_l = instr!(
    "adc A, L",
    4,
    adc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L)
  );

  let adc_a_hl = instr!(
    "adc A, (HL)",
    8,
    adc_r8_mem_r16,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let adc_a_a = instr!(
    "adc A, A",
    4,
    adc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A)
  );

  // Subtract
  let sub_a_b = instr!(
    "sub A, B",
    4,
    sub_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B)
  );

  let sub_a_c = instr!(
    "sub A, C",
    4,
    sub_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let sub_a_d = instr!(
    "sub A, D",
    4,
    sub_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D)
  );

  let sub_a_e = instr!(
    "sub A, E",
    4,
    sub_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E)
  );

  let sub_a_h = instr!(
    "sub A, H",
    4,
    sub_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H)
  );

  let sub_a_l = instr!(
    "sub A, L",
    4,
    sub_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L)
  );

  let sub_a_hl = instr!(
    "sub A, (HL)",
    8,
    sub_r8_mem_r16,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let sub_a_a = instr!(
    "sub A, A",
    4,
    sub_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A)
  );

  // Subtract with carry
  let sbc_a_b = instr!(
    "sbc A, B",
    4,
    sbc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B)
  );

  let sbc_a_c = instr!(
    "sbc A, C",
    4,
    sbc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let sbc_a_d = instr!(
    "sbc A, D",
    4,
    sbc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D)
  );

  let sbc_a_e = instr!(
    "sbc A, E",
    4,
    sbc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E)
  );

  let sbc_a_h = instr!(
    "sbc A, H",
    4,
    sbc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H)
  );

  let sbc_a_l = instr!(
    "sbc A, L",
    4,
    sbc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L)
  );

  let sbc_a_hl = instr!(
    "sbc A, (HL)",
    8,
    sbc_r8_mem_r16,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let sbc_a_a = instr!(
    "sbc A, A",
    4,
    sbc_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A)
  );

  // And
  let and_a_b = instr!(
    "and A, B",
    4,
    and_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B)
  );

  let and_a_c = instr!(
    "and A, C",
    4,
    and_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let and_a_d = instr!(
    "and A, D",
    4,
    and_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D)
  );

  let and_a_e = instr!(
    "and A, E",
    4,
    and_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E)
  );

  let and_a_h = instr!(
    "and A, H",
    4,
    and_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H)
  );

  let and_a_l = instr!(
    "and A, L",
    4,
    and_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L)
  );

  let and_a_hl = instr!(
    "and A, (HL)",
    8,
    and_r8_mem_r16,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let and_a_a = instr!(
    "and A, A",
    4,
    and_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A)
  );

  // Xor
  let xor_a_b = instr!(
    "xor A, B",
    4,
    xor_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B)
  );

  let xor_a_c = instr!(
    "xor A, C",
    4,
    xor_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let xor_a_d = instr!(
    "xor A, D",
    4,
    xor_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D)
  );

  let xor_a_e = instr!(
    "xor A, E",
    4,
    xor_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E)
  );

  let xor_a_h = instr!(
    "xor A, H",
    4,
    xor_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H)
  );

  let xor_a_l = instr!(
    "xor A, L",
    4,
    xor_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L)
  );

  let xor_a_hl = instr!(
    "xor A, (HL)",
    8,
    xor_r8_mem_r16,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let xor_a_a = instr!(
    "xor A, A",
    4,
    xor_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A)
  );

  // or
  let or_a_b = instr!(
    "or A, B",
    4,
    or_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B)
  );

  let or_a_c = instr!(
    "or A, C",
    4,
    or_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let or_a_d = instr!(
    "or A, D",
    4,
    or_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D)
  );

  let or_a_e = instr!(
    "or A, E",
    4,
    or_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E)
  );

  let or_a_h = instr!(
    "or A, H",
    4,
    or_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H)
  );

  let or_a_l = instr!(
    "or A, L",
    4,
    or_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L)
  );

  let or_a_hl = instr!(
    "or A, (HL)",
    8,
    or_r8_mem_r16,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let or_a_a = instr!(
    "or A, A",
    4,
    or_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A)
  );

  // cp
  let cp_a_b = instr!(
    "cp A, B",
    4,
    cp_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::B)
  );

  let cp_a_c = instr!(
    "cp A, C",
    4,
    cp_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let cp_a_d = instr!(
    "cp A, D",
    4,
    cp_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::D)
  );

  let cp_a_e = instr!(
    "cp A, E",
    4,
    cp_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::E)
  );

  let cp_a_h = instr!(
    "cp A, H",
    4,
    cp_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::H)
  );

  let cp_a_l = instr!(
    "cp A, L",
    4,
    cp_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::L)
  );

  let cp_a_hl = instr!(
    "cp A, (HL)",
    8,
    cp_r8_mem_r16,
    InstructionData::small_dst_wide_src(SmallWidthRegister::A, WideRegister::HL)
  );

  let cp_a_a = instr!(
    "cp A, A",
    4,
    cp_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::A)
  );

  let ret_n_z = instr!(
    "retnz",
    8,
    ret,
    InstructionData::const_default().with_flag(ZERO_FLAG, 0)
  );

  let pop_bc = instr!(
    "pop BC",
    12,
    pop_wide_register,
    InstructionData::wide_dst(WideRegister::BC)
  );

  let jnz = instr!(
    "jnz NN",
    12,
    jump_immediate,
    InstructionData::const_default().with_flag(ZERO_FLAG, 0)
  );

  let jmp = instr!(
    "jmp NN",
    8,
    jump_immediate,
    InstructionData::const_default().with_flag(0, 0)
  );

  let callnz = instr!(
    "callnz NN",
    12,
    call_immediate,
    InstructionData::const_default().with_flag(ZERO_FLAG, 0)
  );

  let push_bc = instr!(
    "push BC",
    16,
    push_wide_register,
    InstructionData::wide_dst(WideRegister::BC)
  );

  let add_a_n = instr!(
    "add A, n",
    8,
    add_r8_n,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let rst_0 = instr!("rst 0", 16, rst_n, InstructionData::rst_n(0));

  let ret_z = instr!(
    "retz",
    8,
    ret,
    InstructionData::const_default().with_flag(ZERO_FLAG, ZERO_FLAG)
  );

  // TODO: RET AND JMP WITH NO COMPARISON TAKE A DIFFERENT NUMBER OF CYCLE

  let ret_from_fn = instr!(
    "ret",
    4,
    ret,
    InstructionData::const_default().with_flag(0, 0)
  );

  let jz = instr!(
    "jz NN",
    12,
    jump_immediate,
    InstructionData::const_default().with_flag(ZERO_FLAG, ZERO_FLAG)
  );

  let escape = instr!("ESCAPE", 0, escape, InstructionData::const_default());

  let callz = instr!(
    "callz NN",
    12,
    call_immediate,
    InstructionData::const_default().with_flag(ZERO_FLAG, ZERO_FLAG)
  );

  let call = instr!(
    "call NN",
    12,
    call_immediate,
    InstructionData::const_default().with_flag(0, 0)
  );

  let adc_a_n = instr!(
    "adc A, n",
    8,
    adc_r8_imm,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let rst_8 = instr!("rst 8", 16, rst_n, InstructionData::rst_n(0x8));

  let ret_n_c = instr!(
    "retnc",
    8,
    ret,
    InstructionData::const_default().with_flag(CARRY_FLAG, 0)
  );

  let pop_de = instr!(
    "pop DE",
    12,
    pop_wide_register,
    InstructionData::wide_dst(WideRegister::DE)
  );

  let jnc = instr!(
    "jnc NN",
    12,
    jump_immediate,
    InstructionData::const_default().with_flag(CARRY_FLAG, 0)
  );

  let invalid = instr!("INVALID", 0, invalid_op, InstructionData::const_default());

  let callnc = instr!(
    "callnc NN",
    12,
    call_immediate,
    InstructionData::const_default().with_flag(CARRY_FLAG, 0)
  );

  let push_de = instr!(
    "push DE",
    16,
    push_wide_register,
    InstructionData::wide_dst(WideRegister::DE)
  );

  let sub_a_imm = instr!(
    "sub A, n",
    8,
    sub_r8_n,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let rst_10 = instr!("rst 10H", 16, rst_n, InstructionData::rst_n(0x10));

  let ret_c = instr!(
    "retc",
    8,
    ret,
    InstructionData::const_default().with_flag(CARRY_FLAG, CARRY_FLAG)
  );

  let ret_i = instr!("reti", 16, reti, InstructionData::const_default());

  let jc = instr!(
    "jc NN",
    12,
    jump_immediate,
    InstructionData::const_default().with_flag(CARRY_FLAG, CARRY_FLAG)
  );

  let callc = instr!(
    "callc NN",
    12,
    call_immediate,
    InstructionData::const_default().with_flag(CARRY_FLAG, CARRY_FLAG)
  );

  let sbc_a_n = instr!(
    "sbc A, n",
    8,
    sbc_r8_n,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let rst_18 = instr!("rst 18H", 16, rst_n, InstructionData::rst_n(0x18));

  let ld_ff00_a = instr!(
    "ld (FF00 + n), A",
    12,
    ld_ff00_imm_r8,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let pop_hl = instr!(
    "pop HL",
    12,
    pop_wide_register,
    InstructionData::wide_dst(WideRegister::HL)
  );

  let ld_ff00_c_a = instr!(
    "ld (FF00 + C), A",
    8,
    ld_ff00_r8_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let push_hl = instr!(
    "push HL",
    16,
    push_wide_register,
    InstructionData::wide_dst(WideRegister::HL)
  );

  let and_a_n = instr!(
    "and A, n",
    8,
    and_r8_n,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let rst_20 = instr!("rst 20H", 16, rst_n, InstructionData::rst_n(0x20));

  let add_sp_d = instr!(
    "add SP, n",
    16,
    add_r16_n,
    InstructionData::wide_dst(WideRegister::SP)
  );

  let jmp_indirect_hl = instr!(
    "jmp (HL)",
    4,
    jump_wide_reg,
    InstructionData::wide_dst(WideRegister::HL).with_flag(0, 0)
  );

  let ld_nn_a = instr!(
    "ld (NN), A",
    16,
    load_indirect_nn_small_register,
    InstructionData::small_src(SmallWidthRegister::A)
  );

  let xor_a_n = instr!(
    "xor A, n",
    8,
    xor_r8_n,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let rst_28 = instr!("rst 28H", 16, rst_n, InstructionData::rst_n(0x28));

  let ld_a_ff00 = instr!(
    "ld A, (FF00 + n)",
    8,
    ld_r8_ff00_imm,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let pop_af = instr!(
    "pop AF",
    12,
    pop_wide_register,
    InstructionData::wide_dst(WideRegister::AF)
  );

  let ld_a_ff00_c = instr!(
    "ld A, (FF00 + C)",
    8,
    ld_r8_ff00_r8,
    InstructionData::small_dst_small_src(SmallWidthRegister::A, SmallWidthRegister::C)
  );

  let di = instr!(
    "DI",
    4,
    disable_interrupts,
    InstructionData::const_default()
  );

  let push_af = instr!(
    "push AF",
    16,
    push_wide_register,
    InstructionData::wide_dst(WideRegister::AF)
  );

  let or_a_n = instr!(
    "or A, n",
    8,
    or_r8_n,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let rst_30 = instr!("rst 30H", 16, rst_n, InstructionData::rst_n(0x30));

  let ld_hl_sp_d = instr!(
    "ld HL, SP + d",
    12,
    ld_r16_r16_plus_n,
    InstructionData::wide_dst_wide_src(WideRegister::HL, WideRegister::SP)
  );

  let ld_hl_sp = instr!(
    "ld SP, HL",
    8,
    ld_r16_r16,
    InstructionData::wide_dst_wide_src(WideRegister::SP, WideRegister::HL)
  );

  let ld_a_nn = instr!(
    "ld A, (nn)",
    16,
    ld_r8_indirect_imm,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let ei = instr!("EI", 4, enable_interrupts, InstructionData::const_default());

  let cp_a_n = instr!(
    "cp A, n",
    8,
    cp_r8_n,
    InstructionData::small_dst(SmallWidthRegister::A)
  );

  let rst_38 = instr!("rst 38H", 16, rst_n, InstructionData::rst_n(0x38));

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
fn ext_rlc_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let current = registers.read_r8(additional.small_reg_dst);
  let result = rlc_core(current, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

fn ext_rlc_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = rlc_core(current, registers);
  memory.write_u8(address, result, registers);
}

fn rrc_core(current: u8, registers: &mut Registers) -> u8 {
  let new_reg = current << 7 | current >> 1;
  registers.set_flags(new_reg == 0, false, false, current & 1 != 0);
  new_reg
}

/// RRC in the extended set
fn ext_rrc_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let current = registers.read_r8(additional.small_reg_dst);
  let result = rrc_core(current, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

fn ext_rrc_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = rrc_core(current, registers);
  memory.write_u8(address, result, registers);
}

fn core_rl(reg: u8, registers: &mut Registers) -> u8 {
  let new_reg = (reg << 1) | if registers.carry() { 1 } else { 0 };
  registers.set_flags(new_reg == 0, false, false, reg & (1 << 7) != 0);
  new_reg
}

/// RL in the extended set
fn rl_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  registers.inc_pc(1);
  let reg = registers.read_r8(additional.small_reg_dst);
  let new_reg = core_rl(reg, registers);
  registers.write_r8(additional.small_reg_dst, new_reg);
}

fn ext_rl_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = core_rl(current, registers);
  memory.write_u8(address, result, registers);
}

fn core_rr(reg: u8, registers: &mut Registers) -> u8 {
  let new_reg = (reg >> 1) | if registers.carry() { 1 << 7 } else { 0 };
  registers.set_flags(new_reg == 0, false, false, reg & 1 != 0);
  new_reg
}

/// RR in the extended set
fn ext_rr_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  registers.inc_pc(1);
  let reg = registers.read_r8(additional.small_reg_dst);
  let new_reg = core_rr(reg, registers);
  registers.write_r8(additional.small_reg_dst, new_reg);
}

fn ext_rr_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = core_rr(current, registers);
  memory.write_u8(address, result, registers);
}

fn core_sla(reg: u8, registers: &mut Registers) -> u8 {
  let new_reg = reg << 1;
  registers.set_flags(new_reg == 0, false, false, reg & (1 << 7) != 0);
  new_reg
}

/// SLA in the extended set
fn ext_sla_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  registers.inc_pc(1);
  let reg = registers.read_r8(additional.small_reg_dst);
  let new_reg = core_sla(reg, registers);
  registers.write_r8(additional.small_reg_dst, new_reg);
}

fn ext_sla_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = core_sla(current, registers);
  memory.write_u8(address, result, registers);
}

fn core_sra(reg: u8, registers: &mut Registers) -> u8 {
  // For some reason in SRA the most significant bit (0x80, 128) is ignored from the calculation.
  let new_reg = (reg & 0x80) | reg >> 1;
  registers.set_flags(new_reg == 0, false, false, isset8(reg, 0x1));
  new_reg
}

/// SRA in the extended set
fn ext_sra_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  registers.inc_pc(1);
  let reg = registers.read_r8(additional.small_reg_dst);
  let new_reg = core_sra(reg, registers);
  registers.write_r8(additional.small_reg_dst, new_reg);
}

fn ext_sra_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = core_sra(current, registers);
  memory.write_u8(address, result, registers);
}

fn core_swap(reg: u8, registers: &mut Registers) -> u8 {
  let result = (reg >> 4) | ((reg & 0xF) << 4);
  registers.set_flags(result == 0, false, false, false);
  result
}

/// SWAP in the extended set
fn ext_swap_r8(
  registers: &mut Registers,
  _memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let r1 = registers.read_r8(additional.small_reg_dst);
  let result = core_swap(r1, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

fn ext_swap_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = core_swap(current, registers);
  memory.write_u8(address, result, registers);
}

fn srl_core(current: u8, registers: &mut Registers) -> u8 {
  registers.set_flags(current >> 1 == 0, false, false, current & 0x1 == 0x1);
  current >> 1
}

/// SRL in the extended set
fn ext_srl_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  let current = registers.read_r8(additional.small_reg_dst);
  let result = srl_core(current, registers);
  registers.write_r8(additional.small_reg_dst, result);
  registers.inc_pc(1);
}

fn ext_srl_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = srl_core(current, registers);
  memory.write_u8(address, result, registers);
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
fn ext_bit_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  registers.inc_pc(1);
  let current = registers.read_r8(additional.small_reg_dst);
  bit_core(current, additional.bit, registers);
}

fn ext_bit_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
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
fn ext_res_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  registers.inc_pc(1);
  let target = registers.read_r8(additional.small_reg_dst);
  let result = res_core(target, additional.bit, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

fn ext_res_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = res_core(current, additional.bit, registers);
  memory.write_u8(address, result, registers);
}

fn set_core(current: u8, bit: u8, _registers: &mut Registers) -> u8 {
  let selected_bit = 1 << bit;
  selected_bit | current
}

/// SET in the extended set
fn ext_set_r8(registers: &mut Registers, _memory: &mut GameboyState, additional: &InstructionData) {
  registers.inc_pc(1);
  let target = registers.read_r8(additional.small_reg_dst);
  let result = set_core(target, additional.bit, registers);
  registers.write_r8(additional.small_reg_dst, result);
}

fn ext_set_indirect_r16(
  registers: &mut Registers,
  memory: &mut GameboyState,
  additional: &InstructionData,
) {
  registers.inc_pc(1);
  let address = registers.read_r16(additional.wide_reg_dst);
  let current = memory.read_u8(address);
  let result = set_core(current, additional.bit, registers);
  memory.write_u8(address, result, registers);
}

/// The extended set is a systematic. This produces an 8-instruction row from it
macro_rules! make_extended_row {
  ($name:expr, $method:ident, $cycles:expr, $method_indirect:ident, $cycles_indirect:expr, $with_bit:expr) => {{
    vec![
      instr!(
        format!("{} {:?}", $name, SmallWidthRegister::B),
        $cycles,
        $method,
        InstructionData::small_dst(SmallWidthRegister::B).with_bit($with_bit)
      ),
      instr!(
        format!("{} {:?}", $name, SmallWidthRegister::C),
        $cycles,
        $method,
        InstructionData::small_dst(SmallWidthRegister::C).with_bit($with_bit)
      ),
      instr!(
        format!("{} {:?}", $name, SmallWidthRegister::D),
        $cycles,
        $method,
        InstructionData::small_dst(SmallWidthRegister::D).with_bit($with_bit)
      ),
      instr!(
        format!("{} {:?}", $name, SmallWidthRegister::E),
        $cycles,
        $method,
        InstructionData::small_dst(SmallWidthRegister::E).with_bit($with_bit)
      ),
      instr!(
        format!("{} {:?}", $name, SmallWidthRegister::H),
        $cycles,
        $method,
        InstructionData::small_dst(SmallWidthRegister::H).with_bit($with_bit)
      ),
      instr!(
        format!("{} {:?}", $name, SmallWidthRegister::L),
        $cycles,
        $method,
        InstructionData::small_dst(SmallWidthRegister::L).with_bit($with_bit)
      ),
      instr!(
        format!("{} (HL)", $name),
        $cycles_indirect,
        $method_indirect,
        InstructionData::wide_dst(WideRegister::HL).with_bit($with_bit)
      ),
      instr!(
        format!("{} {:?}", $name, SmallWidthRegister::A),
        $cycles,
        $method,
        InstructionData::small_dst(SmallWidthRegister::A).with_bit($with_bit)
      ),
    ]
  }};

  ($name:expr, $method:ident, $cycles:expr, $method_indirect:ident, $cycles_indirect:expr) => {{
    make_extended_row!(
      $name,
      $method,
      $cycles,
      $method_indirect,
      $cycles_indirect,
      0
    )
  }};
}

/// Make bit-set of extended rows. This enumerates the instructions for every bit on every register for the BIT, RES, and SET instructions
macro_rules! make_bit_set {
  ($name:expr, $method:ident, $cycles:expr, $method_indirect:ident, $cycles_indirect:expr) => {{
    vec![
      make_extended_row!(
        format!("{} {}", $name, 0),
        $method,
        $cycles,
        $method_indirect,
        $cycles_indirect,
        0
      ),
      make_extended_row!(
        format!("{} {}", $name, 1),
        $method,
        $cycles,
        $method_indirect,
        $cycles_indirect,
        1
      ),
      make_extended_row!(
        format!("{} {}", $name, 2),
        $method,
        $cycles,
        $method_indirect,
        $cycles_indirect,
        2
      ),
      make_extended_row!(
        format!("{} {}", $name, 3),
        $method,
        $cycles,
        $method_indirect,
        $cycles_indirect,
        3
      ),
      make_extended_row!(
        format!("{} {}", $name, 4),
        $method,
        $cycles,
        $method_indirect,
        $cycles_indirect,
        4
      ),
      make_extended_row!(
        format!("{} {}", $name, 5),
        $method,
        $cycles,
        $method_indirect,
        $cycles_indirect,
        5
      ),
      make_extended_row!(
        format!("{} {}", $name, 6),
        $method,
        $cycles,
        $method_indirect,
        $cycles_indirect,
        6
      ),
      make_extended_row!(
        format!("{} {}", $name, 7),
        $method,
        $cycles,
        $method_indirect,
        $cycles_indirect,
        7
      ),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<Instruction>>()
  }};
}

pub fn extended_instruction_set() -> Vec<Instruction> {
  // RLC
  let rlc_row = make_extended_row!("rlc", ext_rlc_r8, 8, ext_rlc_indirect_r16, 16);

  // RRC
  let rrc_row = make_extended_row!("rrc", ext_rrc_r8, 8, ext_rrc_indirect_r16, 16);

  // RL
  let rl_row = make_extended_row!("rl", rl_r8, 8, ext_rl_indirect_r16, 16);

  // RR
  let rr_row = make_extended_row!("rr", ext_rr_r8, 8, ext_rr_indirect_r16, 16);

  // SLA
  let sla_row = make_extended_row!("sla", ext_sla_r8, 8, ext_sla_indirect_r16, 16);

  // SRA
  let sra_row = make_extended_row!("sra", ext_sra_r8, 8, ext_sra_indirect_r16, 16);

  // SWAP
  let swap_row = make_extended_row!("swap", ext_swap_r8, 8, ext_swap_indirect_r16, 16);

  // SRL
  let srl_row = make_extended_row!("srl", ext_srl_r8, 8, ext_srl_indirect_r16, 16);

  // BIT
  let bits_row = make_bit_set!("bit", ext_bit_r8, 8, ext_bit_indirect_r16, 16);

  // RES
  let rst_row = make_bit_set!("res", ext_res_r8, 8, ext_res_indirect_r16, 16);

  // SET
  let set_row = make_bit_set!("set", ext_set_r8, 8, ext_set_indirect_r16, 16);

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
