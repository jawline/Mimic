use clap::{AppSettings, Clap};
use log::info;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::{thread, time};

use gb_int::clock::Clock;
use gb_int::cpu::Cpu;
use gb_int::machine::MachineState;
use gb_int::memory::{GameboyState, RomChunk};
use gb_int::ppu::Ppu;
use gb_int::sound::{self, Sound};

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Clap)]
#[clap(version = "1.0", author = "Blake Loring <blake@parsed.uk>")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
  #[clap(short, long)]
  playback_file: String,
}

#[derive(Debug)]
enum Type {
  Lsb {
    frequency: u8,
  },
  Msb {
    trigger: bool,
    length_enable: bool,
    frequency: u8,
  },
  Vol {
    volume: u8,
    add: bool,
    period: u8,
  },
  Duty {
    duty: u8,
    length_load: u8,
  },
}

#[derive(Debug)]
struct Instruction {
  at: usize,
  channel: usize,
  type_: Type,
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
  P: AsRef<Path>,
{
  let file = File::open(filename)?;
  Ok(io::BufReader::new(file).lines())
}

fn try_bool_or_u8(s: &str) -> Result<bool, Box<dyn Error>> {
  match s.parse::<bool>() {
    Ok(v) => Ok(v),
    Err(_invalid_bool_error) => Ok(s.parse::<u8>()? != 0),
  }
}

fn parse_file(filename: &str) -> Result<Vec<Instruction>, Box<dyn Error>> {
  let mut res = Vec::new();
  let lines = read_lines(filename)?;
  for line in lines {
    let line = line?;
    println!("LINE: {}", line);
    // TODO: I don't actually need to allocate here if I use iter functions
    let parts: Vec<String> = line.split(" ").map(|x| x.to_string()).collect();
    if parts[0] == "CH" && parts.len() > 5 {
      let channel: usize = parts[1].parse::<usize>()?;
      let at: usize = parts[parts.len() - 1].parse::<usize>()?;
      if let Some(type_) = match parts[2].as_str() {
        "FREQLSB" => {
          let frequency = parts[3].parse::<u8>()?;
          Some(Type::Lsb { frequency })
        }
        "FREQMSB" => {
          let frequency = parts[3].parse::<u8>()?;
          let length_enable = try_bool_or_u8(&parts[4])?;
          let trigger = try_bool_or_u8(&parts[5])?;
          Some(Type::Msb {
            frequency,
            length_enable,
            trigger,
          })
        }
        "VOLENVPER" => {
          let volume = parts[3].parse::<u8>()?;
          let add = try_bool_or_u8(&parts[4])?;
          let period = parts[5].parse::<u8>()?;
          Some(Type::Vol {
            volume,
            add,
            period,
          })
        }
        "DUTYLL" => {
          let duty = parts[3].parse::<u8>()?;
          let length_load = parts[4].parse::<u8>()?;
          Some(Type::Duty { duty, length_load })
        }
        &_ => {
            println!("FAILED ON LINE");
          /* There is a lot of other noise in stdouts so this isn't necessarily an error */
          None
        }
      } {
        res.push(Instruction { at, channel, type_ });
      }
    } else {  } 
  }
  Ok(res)
}

fn write_lsb(m: &mut MachineState, addr: u16, val: u8) {
  m.memory.write_u8(addr, val, &mut m.cpu.registers);
}

fn write_msb(m: &mut MachineState, addr: u16, trigger: bool, length_enable: bool, frequency: u8) {
  let trigger = if trigger { 1 << 7 } else { 0 };
  let length_enable = if length_enable { 1 << 6 } else { 0 };
  let frequency = frequency & 0b0000_0111;
  m.memory.write_u8(
    addr,
    trigger | length_enable | frequency,
    &mut m.cpu.registers,
  );
}

fn write_voladdperiod(m: &mut MachineState, addr: u16, volume: u8, add: bool, period: u8) {
  let volume = volume << 4;
  let add = if add { 1 << 3 } else { 0 };
  let period = period & 0b0000_0111;
  m.memory
    .write_u8(addr, volume | add | period, &mut m.cpu.registers);
}

fn write_duty(m: &mut MachineState, addr: u16, duty: u8, load_length: u8) {
  let duty = duty << 6;
  let load_length = load_length & 0b0011_1111;
  println!("DUTY: {}", duty);
  m.memory
    .write_u8(addr, duty | load_length, &mut m.cpu.registers);
}

fn base_address(ch: usize) -> u16 {
  println!("Channel: {}", ch);
  match ch {
    1 => 0xFF11,
    2 => 0xFF16,
    _ => panic!("this should be impossible"),
  }
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let instructions = parse_file(&opts.playback_file)?;
  let (_device, _stream, sample_rate, sound_tx) = sound::open_device()?;

  info!("preparing initial state");

  let boot_rom = RomChunk::empty(256);
  let gb_test = RomChunk::empty(8096);
  let root_map = GameboyState::new(boot_rom, gb_test);

  let mut gameboy_state = MachineState {
    cpu: Cpu::new(),
    ppu: Ppu::new(),
    clock: Clock::new(),
    sound: Sound::new(),
    memory: root_map,
  };

  gameboy_state.cpu.registers.last_clock = 4;

  let mut next = 0;
  let mut elapsed = 0;

  while instructions.len() > next {
    elapsed += 4;

    if elapsed > instructions[next].at {
      println!("Moving to next instr");

      let todo = &instructions[next];
      match todo.type_ {
        Type::Lsb { frequency } => {
          write_lsb(
            &mut gameboy_state,
            base_address(todo.channel) + 2,
            frequency,
          );
        }
        Type::Msb {
          trigger,
          length_enable,
          frequency,
        } => {
          write_msb(
            &mut gameboy_state,
            base_address(todo.channel) + 3,
            trigger,
            length_enable,
            frequency,
          );
        }
        Type::Vol {
          volume,
          add,
          period,
        } => {
          write_voladdperiod(
            &mut gameboy_state,
            base_address(todo.channel) + 1,
            volume,
            add,
            period,
          );
        }
        Type::Duty { duty, length_load } => {
          write_duty(
            &mut gameboy_state,
            base_address(todo.channel),
            duty,
            length_load,
          );
        }
      }

      elapsed = 0;
      next += 1;
    }

    gameboy_state.sound.step(
      &mut gameboy_state.cpu,
      &mut gameboy_state.memory,
      sample_rate,
      &sound_tx,
      false,
    );
  }

  loop {
    let ten_millis = time::Duration::from_millis(10);
    thread::sleep(ten_millis);
  }

  // TODO: Find a better way than looping forever - we should be able to test when our buffer is completely empty
}
