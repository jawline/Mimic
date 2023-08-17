mod playback;

pub use playback::*;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum Type {
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

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct Instruction {
  pub at: usize,
  pub channel: usize,
  pub type_: Type,
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

pub fn parse_file_into_chunks_where_buttons_are_not_being_pressed(
  filename: &str,
) -> Result<Vec<Vec<Instruction>>, Box<dyn Error>> {
  let mut chunks = Vec::new();
  let mut res = Vec::new();
  let lines = read_lines(filename)?;

  let mut observing = true;
  for line in lines {
    let line = line?;
    //println!("LINE: {}", line);

    if line == "PRESSING BUTTONS" {
      if res.len() > 0 {
        chunks.push(res);
        res = Vec::new();
      }
      observing = false;
    } else if line == "OBSERVING" {
      observing = true;
    } else if observing {
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
      }
    }
  }

  if observing && res.len() > 0 {
    // Finalize any in-progress chunk
    chunks.push(res);
  }

  Ok(chunks)
}

pub fn parse_file(filename: &str) -> Result<Vec<Instruction>, Box<dyn Error>> {
  let res = parse_file_into_chunks_where_buttons_are_not_being_pressed(filename)?;
  Ok(res.into_iter().flatten().collect())
}
