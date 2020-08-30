use std::vec::Vec;
use log::{trace, warn, error};

/**
 * A trait representing a addressable memory region (ROM or RAM) in the Gameboy.
 */
pub trait MemoryChunk {
  fn write_u8(&mut self, address: u16, value: u8);
  fn read_u8(&self, address: u16) -> u8;
}

impl dyn MemoryChunk {
  pub fn write_u16(&mut self, address: u16, value: u8) {
    unimplemented!();
  }
  pub fn read_u16(&mut self, address: u16) -> u16 {
    unimplemented!();
  }
}

/**
 * Read only chunk of memory loaded as bytes 
 */
pub struct RomChunk {
  pub bytes: Vec<u8>
}

impl MemoryChunk for RomChunk {
  fn write_u8(&mut self, address: u16, _: u8) {
    warn!("tried to write to {} in RomChunk", address);
  }

  fn read_u8(&self, address: u16) -> u8 {
    trace!("read from {} in RomChunk", address);
    self.bytes[address as usize]
  }
}

pub type MemoryMapRegion = (u16, u16);

pub struct MemoryMapEntry {
  chunk: Box<dyn MemoryChunk>,
  region: MemoryMapRegion
}

impl MemoryMapEntry {
  pub fn from(chunk: Box<dyn MemoryChunk>, region: MemoryMapRegion) -> MemoryMapEntry {
    MemoryMapEntry {
      chunk: chunk,
      region: region
    }
  }
 
  fn contains(&self, address: u16) -> bool {
    let (min, max) = self.region;
    address >= min && address <= max
  }
}

impl MemoryChunk for MemoryMapEntry {
  fn write_u8(&mut self, address: u16, val: u8) {
    self.chunk.write_u8(address, val); 
  }
  fn read_u8(&self, address: u16) -> u8 {
    return self.chunk.read_u8(address);
  }
}

/**
 * The memory map root.
 * Takes regions of memory and combines them into a single address space.
 * Handles address translation
 *
 * If two entries conflict in addressing, the first takes precedent.
 */
pub struct MemoryMap {
  entries: Vec<MemoryMapEntry> 
}

impl MemoryMap {
  pub fn from(entries: Vec<MemoryMapEntry>) -> MemoryMap {
    MemoryMap {
      entries: entries
    }
  }
  fn find_entry(&self, address: u16) -> Option<usize> {
    for i in 0..self.entries.len() {
      if self.entries[i].contains(address) {
        return Some(i);
      }
    }
    None
  }
}

/**
 * NOTE: This currently iterates a list, could probably be done more efficiently
 * by precomputing some lookup table on change.
 */
impl MemoryChunk for MemoryMap {
  fn write_u8(&mut self, address: u16, val: u8) {
    if let Some(entry_idx) = self.find_entry(address) {
      trace!("write to {} map entry {}", address, entry_idx);
      self.entries[entry_idx].write_u8(address, val);
    } else {
      error!("write to unmapped address {}", address);
    }
  }

  fn read_u8(&self, address: u16) -> u8 {
    if let Some(entry_idx) = self.find_entry(address) {
      trace!("read {} map entry {}", address, entry_idx);
      self.entries[entry_idx].read_u8(address)
    } else {
      panic!("read from unmapped address {}", address);
    }
  }
}
