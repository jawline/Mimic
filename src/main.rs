#![feature(drain_filter)]
extern crate cpal;

mod clock;
mod cpu;
mod frame_timer;
mod instruction;
mod instruction_data;
mod machine;
mod memory;
mod ppu;
mod register;
mod sdl;
mod sound;
mod terminal;
mod util;

use crate::sound::Sound;
use clock::Clock;
use cpu::Cpu;
use instruction::InstructionSet;
use log::info;
use machine::{Machine, MachineState};
use memory::{GameboyState, RomChunk};
use ppu::Ppu;
use std::error::Error;

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
  sdl_mode: bool,
  #[clap(long)]
  cli_midpoint_rendering: bool,
  #[clap(long)]
  invert: bool,
  #[clap(short, long)]
  skip_bios: bool,
  #[clap(short, long)]
  no_threshold: bool,
  #[clap(long, default_value = "4")]
  frameskip_rate: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();

  let bios_file = opts.bios;
  let rom_file = opts.rom;
  let mut skip_bios = opts.skip_bios;
  let savestate_path = format!("{}.save", rom_file);

  let gameboy = match Machine::load_state(&savestate_path) {
    Ok(machine) => machine,
    Err(_) => {
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

      info!("Cart type: {}", root_map.cart_type);

      let mut gameboy_state = MachineState {
        cpu: Cpu::new(),
        ppu: Ppu::new(),
        clock: Clock::new(),
        sound: Sound::new(),
        memory: root_map,
      };

      if skip_bios {
        // Skip boot
        gameboy_state.cpu.registers.set_pc(0x100);
        gameboy_state.memory.write_u8(0xFF50, 1);
      }

      Machine {
        state: gameboy_state,
        instruction_set: InstructionSet::new(),
      }
    }
  };

  if opts.sdl_mode {
    sdl::run(gameboy, &savestate_path)?;
    Ok(())
  } else {
    terminal::run(
      gameboy,
      &savestate_path,
      opts.frameskip_rate,
      !opts.cli_midpoint_rendering,
      opts.invert,
      !opts.no_threshold,
    )?;
    Ok(())
  }
}
