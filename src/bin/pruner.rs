use clap::{AppSettings, Clap};
use rand::{thread_rng, Rng};
use std::cmp::{max, min};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead};
use std::path::Path;

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Clap)]
#[clap(version = "1.0", author = "Blake Loring <blake@parsed.uk>")]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
  #[clap(short, long)]
  recording: String,
  #[clap(short, long)]
  out: String,
}

#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash)]
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

#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash)]
struct Instruction {
  at: usize,
  channel: usize,
  type_: Type,
}

impl fmt::Display for Instruction {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    use Type::*;
    match self.type_ {
      Lsb { frequency } => write!(
        f,
        "CH {} FREQLSB {} AT {}",
        self.channel, frequency, self.at
      ),
      Msb {
        trigger,
        length_enable,
        frequency,
      } => write!(
        f,
        "CH {} FREQMSB {} {} {} AT {}",
        self.channel, frequency, length_enable, trigger, self.at
      ),
      Vol {
        volume,
        add,
        period,
      } => write!(
        f,
        "CH {} VOLENVPER {} {} {} AT {}",
        self.channel, volume, add, period, self.at
      ),
      Duty { duty, length_load } => write!(
        f,
        "CH {} DUTYLL {} {} AT {}",
        self.channel, duty, length_load, self.at
      ),
    }
    //write!(f, "({}, {})", self.x, self.y)
  }
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
          /* There is a lot of other noise in stdouts so this isn't necessarily an error */
          None
        }
      } {
        res.push(Instruction { at, channel, type_ });
      }
    }
  }
  Ok(res)
}

fn offset(cycles: &[Instruction], i: usize, j: usize) -> usize {
  (i * (cycles.len() + 1)) + j
}

/// A stolen longest subsequence algorithm adjusted for instructions
#[allow(dead_code)]
fn find_repeating_subsequence(cycles: &[Instruction]) -> Vec<Instruction> {
  let mut dp = vec![0; (cycles.len() + 1) * (cycles.len() + 1)];
  let n = cycles.len();
  for i in 1..(n + 1) {
    for j in 1..(n + 1) {
      if i != j && cycles[i - 1] == cycles[j - 1] {
        dp[offset(cycles, i, j)] = 1 + dp[offset(cycles, i - 1, j - 1)];
      } else {
        dp[offset(cycles, i, j)] = max(dp[offset(cycles, i, j - 1)], dp[offset(cycles, i - 1, j)]);
      }
    }
  }

  let mut i = n;
  let mut j = n;
  let mut sequence = Vec::new();

  while i > 0 && j > 0 {
    if dp[offset(cycles, i, j)] == dp[offset(cycles, i - 1, j - 1)] + 1 {
      sequence.push(cycles[i - 1]);
      i -= 1;
      j -= 1;
    } else if dp[offset(cycles, i, j)] == dp[offset(cycles, i - 1, j)] {
      i -= 1;
    } else {
      j -= 1;
    }
  }

  sequence.reverse();

  sequence
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let instructions = parse_file(&opts.recording)?;

  let mut rng = thread_rng();

  for i in 0..2 {
    let path = format!("{}/{}", opts.out, i);
    println!("Writing next file to {}", path);
    let mut file = File::create(path)?;
    let limit = rng.gen_range(0..instructions.len());
    for instruction in &instructions[limit..min(limit + 10000, instructions.len())] {
      write!(file, "{}\n", instruction)?;
    }
  }

  /*
  let mut locations: HashMap<Instruction, Vec<usize>> = HashMap::new();
  let mut i = 0;
  for instruction in &instructions {
    if let Some(v) = locations.get_mut(instruction) {
      v.push(i);
    } else {
      locations.insert(*instruction, vec![i]);
    }
    i = i + 1;
  }

  for (instruction, cycle_points) in locations.iter() {
      println!("{}", cycle_points.len());
    if cycle_points.len() > 3 && cycle_points.len() < 30_000 {
      let longest_pattern = find_repeating_subsequence(&cycle_points);
      if longest_pattern.len() > 0 {
        println!("{:?}", longest_pattern);
      }
    }
  } */

  Ok(())
}
