use clap::Parser;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::str::from_utf8;

use std::sync::mpsc;

use gb_int::{
  clock::Clock,
  cpu::{Cpu, INTERRUPTS_ENABLED_ADDRESS, TIMER, TIMER_ADDRESS, VBLANK, VBLANK_ADDRESS},
  instruction::InstructionSet,
  instruction_compiler::{write_machine_code, Address, Program},
  machine::{Machine, MachineState},
  memory::{GameboyState, RomChunk},
  ppu::Ppu,
  sound,
  sound::Sound,
};

#[repr(C, packed(1))]
struct GbsHeader {
  gbs_header: [u8; 3],
  version: u8,
  song_count: u8,
  first_song: u8,
  load_address: u16,
  init_address: u16,
  play_address: u16,
  stack_pointer: u16,
  timer_modulo: u8,
  timer_control: u8,
  title: [u8; 32],
  author: [u8; 32],
  copyright: [u8; 32],
}

impl GbsHeader {
  fn header(&self) -> String {
    from_utf8(&self.gbs_header).unwrap().trim().to_string()
  }

  fn title(&self) -> String {
    from_utf8(&self.title).unwrap().trim().to_string()
  }

  fn author(&self) -> String {
    from_utf8(&self.author).unwrap().trim().to_string()
  }

  fn copyright(&self) -> String {
    from_utf8(&self.copyright).unwrap().trim().to_string()
  }
}

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Opts {
  #[arg(short, long)]
  playback_file: String,
  #[arg(short, long)]
  track: u8,
  #[arg(short, long)]
  disable_sound: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
  env_logger::init();
  let opts: Opts = Opts::parse();
  let f = File::open(opts.playback_file)?;
  let mut reader = BufReader::new(f);
  let mut buffer = Vec::new();
  reader.read_to_end(&mut buffer)?;

  let header = buffer.as_ptr() as *const GbsHeader;
  let header: &GbsHeader = unsafe { &*header };

  println!("{}", header.header());
  println!("{}", header.title());
  println!("{}", header.author());
  println!("{}", header.copyright());

  let boot_rom = RomChunk::empty(256);
  let mut sound_rom = RomChunk::empty(0x10000);

  let load_address = header.load_address;
  let init_address = header.init_address;
  let play_address = header.play_address;
  let timer_modulo = header.timer_modulo;
  let timer_control = header.timer_control;
  let song_count = header.song_count;

  if opts.track >= song_count {
    println!("Song requested exceeds song count");
    return Ok(());
  }

  println!(
    "Load: {:x} Init: {:x} Play: {:x} TIMER MODULO: {:x} TIMER CONTROL: {:x} TRACKS: {}",
    load_address, init_address, play_address, timer_modulo, timer_control, song_count
  );

  let data = &buffer[std::mem::size_of::<GbsHeader>()..];

  for (index, &byte) in data.iter().enumerate() {
    let write_address = load_address as usize + index;
    if write_address > 0xFFFF {
      println!("ROM data exceeded GB address space?");
      break;
    }
    sound_rom.force_write_u8(write_address as u16, byte);
    //println!("{:x}: {:x}", write_address, byte);
  }

  println!("Programming custom logic");

  let start_of_custom_code = 0x100;
  use Address::*;
  use Program::*;

  // For each RST jump to load_address + RST
  for addr in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38] {
    write_machine_code(
      &[Jump {
        address: Absolute(load_address + addr),
      }],
      |index, byte| {
        sound_rom.force_write_u8(addr + index, byte);
      },
    );
  }

  // Program a custom piece of code at 0x100 that initializes and then plays the GBS track loaded
  // at init_address and play_address.
  write_machine_code(
    &[
      /* At 0x100 have Jump 0x100 to create a return loop */
      Jump {
        address: Absolute(start_of_custom_code),
      },
      /* Set A register to track */
      SetA {
        immediate: opts.track,
      },
      /* Call INIT address in GBS header */
      Call {
        address: Absolute(init_address),
      },
      /* Call PLAY address in GBS header */
      Call {
        address: Absolute(play_address),
      },
      /* Jump to 0x100 after the ret to sit in a busy loop */
      Jump {
        address: Absolute(start_of_custom_code),
      },
    ],
    |index, byte| {
      sound_rom.force_write_u8(start_of_custom_code + index, byte);
    },
  );

  let enable_interrupts_and_jump_to_play_address = [
    EnableInterrupts,
    Jump {
      address: Absolute(play_address),
    },
  ];

  // Program a custom VBLANK register that calls the play function, enables interrupts, and then
  // jumps to 0x100. Vblank is commonly used as a song timer to trigger play to be called roughly
  // once every 60 seconds.
  write_machine_code(
    &enable_interrupts_and_jump_to_play_address,
    |index, byte| {
      sound_rom.force_write_u8(VBLANK_ADDRESS + index, byte);
    },
  );

  // Program timer with the same function as VBLANK. In some ROMs this is used.
  write_machine_code(
    &enable_interrupts_and_jump_to_play_address,
    |index, byte| {
      sound_rom.force_write_u8(TIMER_ADDRESS + index, byte);
    },
  );

  println!("Programmed ROM");

  let mut root_map = GameboyState::new(boot_rom, sound_rom, false);

  root_map.boot_enabled = false;

  println!("Disabled boot mode");

  let gameboy_state = MachineState {
    cpu: Cpu::new(),
    ppu: Ppu::new(),
    clock: Clock::new(),
    sound: Sound::new(),
    memory: root_map,
  };

  // Drop will cause the device to terminate and close the stream
  // so we need to return it from the if.
  let (_device, _stream, sample_rate, sound_tx) = if opts.disable_sound {
    println!("Using a dummy sound device");
    let (sound_tx, _sound_rx) = mpsc::channel();
    (None, None, 1_000_000, sound_tx)
  } else {
    println!("Opening real sound device");
    let (device, stream, sample_rate, sound_tx) = sound::open_device()?;
    (Some(device), Some(stream), sample_rate, sound_tx)
  };

  println!("Opened sound device");

  // This will be unused but we need to provide a buffer. Make it small so we crash if
  // disable_framebuffer isn't working
  let mut pixel_buffer = vec![0; 1];

  let mut gameboy = Machine {
    state: gameboy_state,
    instruction_set: InstructionSet::new(),
    disable_sound: opts.disable_sound,
    disable_framebuffer: true,
  };

  println!("Machine created");

  gameboy.state.cpu.registers.set_sp(header.stack_pointer);

  println!(
    "Set stack pointer to {:x}",
    gameboy.state.cpu.registers.sp()
  );

  gameboy
    .state
    .cpu
    .registers
    .set_pc(start_of_custom_code + 0x3);

  gameboy.state.cpu.registers.ime = true;
  gameboy.state.memory.disable_rom_upper_writes = true;
  gameboy.state.memory.print_sound_registers = true;

  // If timer_control or timer_modulo are nonzero then use the timer interrupt otherwise use the
  // VSync interrupt.

  gameboy
    .state
    .memory
    .write_u8(0xFF06, timer_modulo, &gameboy.state.cpu.registers);

  gameboy
    .state
    .memory
    .write_u8(0xFF07, timer_control, &gameboy.state.cpu.registers);

  gameboy.state.memory.write_u8(
    INTERRUPTS_ENABLED_ADDRESS,
    TIMER,
    &gameboy.state.cpu.registers,
  );
  gameboy.state.memory.write_u8(
    INTERRUPTS_ENABLED_ADDRESS,
    VBLANK,
    &gameboy.state.cpu.registers,
  );

  println!("INIT done, preparing to play");

  loop {
    loop {
      gameboy.step(&mut pixel_buffer, sample_rate, &sound_tx);
    }
  }
}
