use std::io;
use std::io::prelude::*;
use std::fs::File;
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
  pub fn write_u16(&mut self, address: u16, value: u16) {
    let lower = value & 0xFF;
    let upper = value >> 8;
    self.write_u8(address + 1, upper as u8);
    self.write_u8(address, lower as u8);
  }
  pub fn read_u16(&mut self, address: u16) -> u16 {
    let upper = self.read_u8(address + 1);
    let lower = self.read_u8(address);
    let result = ((upper as u16) << 8) + (lower as u16);
    result
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
    warn!("tried to write to {:x} in RomChunk", address);
  }
  fn read_u8(&self, address: u16) -> u8 {
    //trace!("read from {:x} in RomChunk", address);
    self.bytes[address as usize]
  }
}

impl RomChunk {
  pub fn from_file(path: &str) -> io::Result<RomChunk> {
    let mut f = File::open(path)?;
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer)?;
    Ok(RomChunk {
      bytes: buffer
    })
  }
}

/**
 * RAM read/write memory as bytes
 */
pub struct RamChunk {
  pub bytes: Vec<u8>
}

impl MemoryChunk for RamChunk {
  fn write_u8(&mut self, address: u16, v: u8) {
    //trace!("write {} to {:x} in RamChunk", v, address);
    self.bytes[address as usize] = v;
  }
  fn read_u8(&self, address: u16) -> u8 {
    //trace!("read from {:x} in RamChunk", address);
    self.bytes[address as usize]
  }
}

impl RamChunk {
  pub fn new(size: usize) -> RamChunk {
    RamChunk {
      bytes: vec![0; size]
    }
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

  fn address_offset(&self, address: u16) -> u16 {
    let (min, _) = self.region;
    return address - min;
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
      trace!("write {:x} to {:x} map entry {}", val, address, entry_idx);
      let address = self.entries[entry_idx].address_offset(address);
      self.entries[entry_idx].write_u8(address, val);
    } else {
      error!("write {} to unmapped address {:x}", val, address);
    }
  }

  fn read_u8(&self, address: u16) -> u8 {
    if let Some(entry_idx) = self.find_entry(address) {
      trace!("read {:x} map entry {}", address, entry_idx);
      if address == 0xFF80 { trace!("magic read!"); return 0x0; /* TODO: Tetris wants a gamepad input but we currently don't support it with the magic registers. */ }
      let address = self.entries[entry_idx].address_offset(address);
      self.entries[entry_idx].read_u8(address)
    } else {
      panic!("read from unmapped address {:x}", address);
    }
  }
}
