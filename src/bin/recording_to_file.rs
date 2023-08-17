use clap::Parser;
use std::error::Error;
use std::fs::remove_file;
use std::sync::mpsc;

use hound;

use gb_int::encoded_file::*;

const MIN_SECONDS: f64 = 5.;
const MAX_SECONDS: f64 = 160.;

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Opts {
  #[arg(short, long)]
  file: String,
  #[arg(short, long)]
  output: String,
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let instructions = parse_file(&opts.file)?;

  let (sound_tx, sound_rx) = mpsc::channel();

  let wav_spec = hound::WavSpec {
    channels: 1,
    sample_rate: VEC_SAMPLE_RATE as u32,
    bits_per_sample: 16,
    sample_format: hound::SampleFormat::Int,
  };

  let mut writer = hound::WavWriter::create(opts.output.clone(), wav_spec)?;

  let mut min_sample = i16::MAX;
  let mut max_sample = i16::MIN;
  let mut samples = 0;

  let mut seconds_of_audio = 0.;

  let res = to_wave(&instructions, sound_tx, VEC_SAMPLE_RATE, || {
    while let Ok(sample) = sound_rx.try_recv() {
      let sample = (sample * (i16::MAX as f32)) as i16;
      max_sample = std::cmp::max(sample, max_sample);
      min_sample = std::cmp::min(sample, min_sample);
      samples += 1;
      seconds_of_audio = samples as f64 / VEC_SAMPLE_RATE as f64;
      writer.write_sample(sample)?;
    }

    // Early exit at MAX_SECONDS to avoid burning CPU.
    if seconds_of_audio > MAX_SECONDS {
      use std::io::{Error, ErrorKind};
      Err(Error::new(ErrorKind::Other, "hit max seconds"))?;
    }

    Ok(())
  });

  // Delete the file on error.
  match res {
    Ok(()) => (),
    Err(_) => {
      println!("Deleting output file on error");
      remove_file(opts.output.clone())?
    }
  };

  println!("{} seconds of audio", seconds_of_audio);

  if seconds_of_audio < MIN_SECONDS || seconds_of_audio > MAX_SECONDS {
    println!("Sample failed quality heuristics. Deleting");
    remove_file(opts.output)?;
  }

  println!("Written");

  Ok(())
}
