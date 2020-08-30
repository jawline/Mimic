mod memory;
mod cpu;
mod machine;
mod instruction;

use memory::{RomChunk, MemoryMapEntry, MemoryMap};
use log::info;

fn main() {
  info!("preparing initial state");

  let boot: Vec<u8> = [0; 255].to_vec();

  let boot_rom = RomChunk { bytes: boot };

  let root_map = MemoryMap::from(
    vec![MemoryMapEntry::from(Box::new(boot_rom), (0, 255))]
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
