use clap::Parser;
use std::error::Error;
use std::{thread, time};

use gb_int::encoded_file::*;
use gb_int::sound;

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Opts {
  #[clap(short, long)]
  playback_file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let instructions = parse_file(&opts.playback_file)?;
  let (_device, _stream, sample_rate, sound_tx) = sound::open_device()?;

  to_wave(&instructions, sound_tx, sample_rate, || Ok(()))?;

  loop {
    let ten_millis = time::Duration::from_millis(10);
    thread::sleep(ten_millis);
  }

  // TODO: Find a better way than looping forever - we should be able to test when our buffer is completely empty
}
