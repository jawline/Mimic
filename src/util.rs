use crate::memory::{isset16, isset8, GameboyState};

pub const STAT: u16 = 0xFF41;

/// TODO: We should mask out writes to the STAT register with this
const STAT_FLAG_MASK: u8 = 0x4 | 0x2 | 0x1;

pub fn stat(mem: &GameboyState) -> u8 {
  mem.read_u8(STAT)
}

pub fn update_stat_flags(flags: u8, mem: &mut GameboyState) {
  // Mask out the flag bits from the current stat for our replacement
  let current_stat = stat(mem) & (!STAT_FLAG_MASK);
  let new_stat = current_stat | flags;
  mem.write_u8(STAT, new_stat);
}

pub fn stat_interrupts_with_masked_flags(flags: u8, mem: &mut GameboyState) -> u8 {
  let current_stat = stat(mem) & STAT_FLAG_MASK;
  let new_stat = current_stat | (flags & (!STAT_FLAG_MASK));
  new_stat
}

pub fn carries_add8_with_carry(val: u8, operand: u8, carry: bool) -> (bool, bool) {
  // Upgrade them to 16 bits so they are wide enough to
  // contain 0x100
  let val = val as u16;
  let operand = operand as u16;
  let carry = if carry { 1 } else { 0 };
  let sum = val + operand + carry;
  let carry_into = sum ^ val ^ operand ^ carry;

  let half_carry = isset16(carry_into, 0x10);
  let carry = isset16(carry_into, 0x100);

  (half_carry, carry)
}

pub fn carries_add16_signed_8bit(val: u16, operand: u8) -> (bool, bool) {
  let operand = operand as u16;
  let result = val + operand;
  let carry_bitvector = result ^ val ^ operand;
  (
    isset16(carry_bitvector, 0x10),
    isset16(carry_bitvector, 0x100),
  )
}

pub fn carries_sub16_signed_8bit(val: u16, operand: u8) -> (bool, bool) {
  carries_sub8((val & 0xFF) as u8, operand)
}

pub fn carries_sub8_with_carry(val: u8, operand: u8, carry: bool) -> (bool, bool) {
  // Upgrade them to 16 bits so they are wide enough to
  // contain 0x100

  let val = val as u16;
  let operand = operand as u16;
  let carry = if carry { 1 } else { 0 };
  let sum = val - operand - carry;
  let carry_into = sum ^ val ^ operand ^ carry;

  let half_carry = isset16(carry_into, 0x10);
  let carry = isset16(carry_into, 0x100);

  (half_carry, carry)
}

pub fn carries_add8(val: u8, operand: u8) -> (bool, bool) {
  carries_add8_with_carry(val, operand, false)
}

pub fn half_carry_add8(val: u8, operand: u8) -> bool {
  isset8((val & 0xF) + (operand & 0xF), 0x10)
}

pub fn half_carry_sub8(val: u8, operand: u8) -> bool {
  isset8((val & 0xF) - (operand & 0xF), 0x10)
}

pub fn carries_sub8(val: u8, operand: u8) -> (bool, bool) {
  carries_sub8_with_carry(val, operand, false)
}
