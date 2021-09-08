mod clock;
mod cpu;
mod gpu;
mod instruction;
mod machine;
mod memory;
mod sdl;
mod terminal;
mod util;

use std::io;

use clock::CLOCK;
use cpu::CPU;
use gpu::GPU;
use log::info;
use machine::Machine;
use memory::{GameboyState, RomChunk};

use clap::{AppSettings, Clap};

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Clap)]
#[clap(version = "1.0", author = "Blake Loring <blake@parsed.uk>")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
  /// Sets a custom config file. Could have been an Option<T> with no default too
  #[clap(short, long)]
  bios: Option<String>,
  #[clap(short, long)]
  rom: String,
  #[clap(short, long)]
  cli_mode: bool,
  #[clap(short, long)]
  cli_midpoint_rendering: bool,
  #[clap(short, long)]
  invert: bool,
  #[clap(short, long)]
  skip_bios: bool,
  #[clap(short, long)]
  no_threshold: bool,
}

fn main() -> io::Result<()> {
  env_logger::init();
  let opts: Opts = Opts::parse();

  let bios_file = opts.bios;
  let rom_file = opts.rom;
  let mut skip_bios = opts.skip_bios;

  info!("loading BIOS: {:#?} TEST: {}", bios_file, rom_file);

  info!("preparing initial state");

  let boot_rom = if let Some(bios_file) = bios_file {
    RomChunk::from_file(&bios_file)?
  } else {
    skip_bios = true;
    RomChunk::empty()
  };

  info!("loaded BIOS");

  let gb_test = RomChunk::from_file(&rom_file)?;

  info!("Loaded rom");

  let root_map = GameboyState::new(boot_rom, gb_test);

  let mut gameboy_state = Machine {
    cpu: CPU::new(),
    gpu: GPU::new(),
    clock: CLOCK::new(),
    memory: root_map,
  };

  if skip_bios {
    // Skip boot
    use crate::memory::MemoryChunk;
    gameboy_state.cpu.registers.set_pc(0x100);
    gameboy_state.memory.write_u8(0xFF50, 1);
  }

  if !opts.cli_mode {
    sdl::run(gameboy_state)
  } else {
    terminal::run(
      gameboy_state,
      !opts.cli_midpoint_rendering,
      opts.invert,
      !opts.no_threshold,
    )
  }
}
