#![feature(duration_consts_float)]
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
  #[clap(short, long)]
  playback_file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let (_device, _stream, sample_rate, sound_tx) = crate::sound::open_device()?;
  Ok(())
}
