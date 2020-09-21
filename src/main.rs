mod memory;
mod cpu;
mod gpu;
mod machine;
mod instruction;

use std::io;
use memory::{RomChunk, RamChunk, MemoryMapEntry, MemoryMap};
use log::info;
use gpu::GPU;

fn main() -> io::Result<()> {
  env_logger::init();
  info!("preparing initial state");

  let gpu = GPU::new();

  let gb_test = RomChunk::from_file("/home/blake/gb/test2.gb");
  let boot_rom = RomChunk::from_file("/home/blake/gb/bios.gb");
  let high_ram = RamChunk::new(255);

  // Ram for the sprite buffers etc
  let vram = RamChunk::new(0x2000);

  // Cartridge might have RAM
  let cart_ram = RamChunk::new(0x2000);

  let root_map = MemoryMap::from(
    vec![
      MemoryMapEntry::from(Box::new(boot_rom?), (0, 255)),
      MemoryMapEntry::from(Box::new(gb_test?), (0, 0x7FFF)),
      MemoryMapEntry::from(Box::new(vram), (0x8000, 0x9FFF)),
      MemoryMapEntry::from(Box::new(cart_ram), (0xA000, 0xBFFF)),
      MemoryMapEntry::from(Box::new(high_ram), (0xFF00, 0xFFFF)),
    ]
  );

  let mut gameboy_state = machine::Machine {
    cpu: cpu::CPU::new(),
    gpu: gpu,
    memory: Box::new(root_map)
  };

  info!("starting core loop");

  loop {
    gameboy_state.step();
  }
}
