use std::error::Error;

use gb_int::clock::Clock;
use gb_int::cpu::Cpu;
use gb_int::headless;
use gb_int::instruction::InstructionSet;
use gb_int::machine::{Machine, MachineState};
use gb_int::memory::{GameboyState, RomChunk};
use gb_int::ppu::Ppu;
use gb_int::sdl;
use gb_int::sound::Sound;
use gb_int::terminal;

use clap::Parser;
use log::info;

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Opts {
  /// Sets a custom config file. Could have been an Option<T> with no default too
  #[arg(short, long)]
  bios: Option<String>,
  #[arg(short, long)]
  rom: String,
  #[arg(short, long)]
  mode: String,
  #[arg(long)]
  cli_midpoint_rendering: bool,
  #[arg(long)]
  invert: bool,
  #[arg(short, long)]
  skip_bios: bool,
  #[arg(short, long)]
  no_threshold: bool,
  #[arg(long, default_value = "4")]
  frameskip_rate: u32,
  #[arg(short, long)]
  disable_sound: bool,
  #[arg(long)]
  disable_framebuffer: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();

  let bios_file = opts.bios;
  let rom_file = opts.rom;
  let mut skip_bios = opts.skip_bios;
  let savestate_path = format!("{}.save", rom_file);

  let gameboy = match Machine::load_state(
    &savestate_path,
    opts.disable_sound,
    opts.disable_framebuffer,
  ) {
    Ok(machine) => machine,
    Err(_) => {
      info!("loading BIOS: {:#?} TEST: {}", bios_file, rom_file);

      info!("preparing initial state");

      let boot_rom = if let Some(bios_file) = bios_file {
        RomChunk::from_file(&bios_file)?
      } else {
        skip_bios = true;
        RomChunk::empty(256)
      };

      info!("loaded BIOS");

      let gb_test = RomChunk::from_file(&rom_file)?;

      info!("Loaded rom");

      let root_map = GameboyState::new(boot_rom, gb_test, false);

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
        gameboy_state
          .memory
          .write_u8(0xFF50, 1, &gameboy_state.cpu.registers);
      }

      Machine {
        state: gameboy_state,
        instruction_set: InstructionSet::new(),
        disable_sound: opts.disable_sound,
        disable_framebuffer: opts.disable_framebuffer,
      }
    }
  };

  if opts.mode == "sdl" {
    sdl::run(gameboy, &savestate_path)?;
    Ok(())
  } else if opts.mode == "terminal" {
    terminal::run(
      gameboy,
      &savestate_path,
      opts.frameskip_rate,
      !opts.cli_midpoint_rendering,
      opts.invert,
      !opts.no_threshold,
    )?;
    Ok(())
  } else if opts.mode == "headless" {
    headless::run(gameboy)?;
    Ok(())
  } else {
    panic!(
      "opts.mode must be one of sdl|terminal|headless but was {}",
      opts.mode
    )
  }
}
