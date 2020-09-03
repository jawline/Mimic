mod memory;
mod cpu;
mod machine;
mod instruction;

use std::io;
use std::error::Error;
use memory::{RomChunk, MemoryMapEntry, MemoryMap};
use log::info;

fn main() -> io::Result<()> {
  info!("preparing initial state");

  let boot_rom = RomChunk::from_file("/home/blake/gb/bios.gb");

  let root_map = MemoryMap::from(
    vec![MemoryMapEntry::from(Box::new(boot_rom?), (0, 255))]
  );

  let mut gameboy_state = machine::Machine {
    cpu: cpu::CPU::new(),
    memory: Box::new(root_map)
  };

  info!("starting core loop");

  loop {
    gameboy_state.step();
  }
}
