use crate::memory::MemoryPtr;

pub const STAT: u16 = 0xFF41;

/// TODO: We should mask out writes to the STAT register with this
const STAT_FLAG_MASK: u8 = 0x4 | 0x2 | 0x1;

pub fn stat(mem: &MemoryPtr) -> u8 {
  mem.read_u8(STAT)
}

pub fn update_stat_flags(flags: u8, mem: &mut MemoryPtr) {
  // Mask out the flag bits from the current stat for our replacement
  let current_stat = stat(mem) & (!STAT_FLAG_MASK);
  let new_stat = current_stat | flags;
  mem.write_u8(STAT, new_stat);
}

pub fn stat_interrupts_with_masked_flags(flags: u8, mem: &mut MemoryPtr) -> u8 {
  let current_stat = stat(mem) & STAT_FLAG_MASK;
  let new_stat = current_stat | (flags & (!STAT_FLAG_MASK));
  new_stat
}
