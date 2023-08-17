use crate::clock::Clock;
use crate::cpu::Cpu;
use crate::encoded_file::{Instruction, Type};
use crate::machine::MachineState;
use crate::memory::{GameboyState, RomChunk};
use crate::ppu::Ppu;
use crate::sound::Sound;
use log::info;
use std::error::Error;
use std::sync::mpsc::{self, Sender};

pub const VEC_SAMPLE_RATE: usize = 48000;

pub fn write_lsb(m: &mut MachineState, addr: u16, val: u8) {
  m.memory.write_u8(addr, val, &mut m.cpu.registers);
}

pub fn write_msb(
  m: &mut MachineState,
  addr: u16,
  trigger: bool,
  length_enable: bool,
  frequency: u8,
) {
  let trigger = if trigger { 1 << 7 } else { 0 };
  let length_enable = if length_enable { 1 << 6 } else { 0 };
  let frequency = frequency & 0b0000_0111;
  m.memory.write_u8(
    addr,
    trigger | length_enable | frequency,
    &mut m.cpu.registers,
  );
}

pub fn write_voladdperiod(m: &mut MachineState, addr: u16, volume: u8, add: bool, period: u8) {
  let volume = volume << 4;
  let add = if add { 1 << 3 } else { 0 };
  let period = period & 0b0000_0111;
  m.memory
    .write_u8(addr, volume | add | period, &mut m.cpu.registers);
}

pub fn write_duty(m: &mut MachineState, addr: u16, duty: u8, load_length: u8) {
  let duty = duty << 6;
  let load_length = load_length & 0b0011_1111;
  m.memory
    .write_u8(addr, duty | load_length, &mut m.cpu.registers);
}

pub fn base_address(ch: usize) -> u16 {
  match ch {
    1 => 0xFF11,
    2 => 0xFF16,
    _ => panic!("this should be impossible"),
  }
}

pub fn to_wave<F>(
  instructions: &[Instruction],
  output_channel: Sender<f32>,
  sample_rate: usize,
  mut step_callback: F,
) -> Result<(), Box<dyn Error>>
where
  F: FnMut() -> Result<(), Box<dyn Error>>,
{
  info!("preparing initial state");

  let boot_rom = RomChunk::empty(256);
  let gb_test = RomChunk::empty(8096);
  let root_map = GameboyState::new(boot_rom, gb_test, false);

  let mut gameboy_state = MachineState {
    cpu: Cpu::new(),
    ppu: Ppu::new(),
    clock: Clock::new(),
    sound: Sound::new(),
    memory: root_map,
  };

  gameboy_state.cpu.registers.cycles_elapsed_during_last_step = 4;

  let mut next = 0;
  let mut elapsed = 0;

  while instructions.len() > next {
    elapsed += 4;

    if elapsed > instructions[next].at {
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
      &output_channel,
      false,
    );

    step_callback()?;
  }

  Ok(())
}

pub fn to_wave_vec(instructions: &[Instruction]) -> Result<Vec<f32>, Box<dyn Error>> {
  let (sound_tx, sound_rx) = mpsc::channel();
  to_wave(instructions, sound_tx, VEC_SAMPLE_RATE, || Ok(()))?;
  Ok(sound_rx.iter().collect())
}
