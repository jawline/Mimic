mod sdl;
mod clock;
mod cpu;
mod gpu;
mod instruction;
mod machine;
mod memory;
mod util;
mod terminal;

use std::env;
use std::io;
use std::time::Instant;

use sdl2;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{Texture, WindowCanvas};
use sdl2::EventPump;

use clock::CLOCK;
use cpu::{CPU, JOYPAD};
use gpu::{GpuStepState, BYTES_PER_ROW, GB_SCREEN_HEIGHT, GB_SCREEN_WIDTH, GPU};
use log::{info, trace};
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
    bios: String,
    #[clap(short, long)]
    rom: String,
    #[clap(short, long)]
    cli_mode: bool,
    #[clap(short, long)]
    skip_bios: bool,
}

fn main() -> io::Result<()> {
  env_logger::init();
  let opts: Opts = Opts::parse();

  let bios_file = opts.bios;
  let rom_file = opts.rom;

  info!("loading BIOS: {} TEST: {}", bios_file, rom_file);

  info!("preparing initial state");

  let boot_rom = RomChunk::from_file(&bios_file)?;
  let gb_test = RomChunk::from_file(&rom_file)?;

  let root_map = GameboyState::new(boot_rom, gb_test);

  let mut gameboy_state = Machine {
    cpu: CPU::new(),
    gpu: GPU::new(),
    clock: CLOCK::new(),
    memory: root_map,
  };

  if opts.skip_bios {
    // Skip boot
    use crate::memory::MemoryChunk;
    gameboy_state.cpu.registers.set_pc(0x100);
    gameboy_state.memory.write_u8(0xFF50, 1);
  }

  if !opts.cli_mode {
    sdl::run(gameboy_state)
  } else {
    terminal::run(gameboy_state)
  }
}
