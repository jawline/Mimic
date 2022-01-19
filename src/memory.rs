use crate::cpu::Registers;
use crate::util::{stat_interrupts_with_masked_flags, STAT};
use log::{error, info, trace, warn};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::vec::Vec;

pub const DIV_REGISTER: u16 = 0xFF04;
pub const TIMA_REGISTER: u16 = 0xFF05;
pub const MOD_REGISTER: u16 = 0xFF06;
pub const TAC_REGISTER: u16 = 0xFF07;

const RAM_BANK_SIZE: usize = 0x2000;
const END_OF_BOOT: u16 = 0x101;
const END_OF_FIXED_ROM: u16 = 0x4000;
const END_OF_BANKED_ROM: u16 = 0x8000;
const END_OF_VRAM: u16 = 0xA000;
const END_OF_CARTRIDGE_RAM: u16 = 0xC000;
const END_OF_INTERNAL_RAM: u16 = 0xE000;
const END_OF_ECHO_RAM: u16 = 0xFE00;

const ROM_BANK_SIZE: usize = 0x4000;
const GAMEPAD_ADDRESS: u16 = 0xFF00;
const BOOT_ROM_ADDRESS: u16 = 0xFF50;

const fn is_mbc1(cart_type: u8) -> bool {
  match cart_type {
    1 | 2 | 3 => true,
    _ => false,
  }
}

const fn is_mbc3(cart_type: u8) -> bool {
  match cart_type {
    0x11 | 0x12 | 0x13 => true,
    _ => false,
  }
}

/// Returns an 8-bit bitvector with the specified flags set
pub fn set8(val: u8, bit: u8) -> u8 {
  val | bit
}

/// Return a 8-bit bitvector with the specified flags
pub fn unset8(val: u8, bit: u8) -> u8 {
  val & !bit
}

/// Check if an 8-bit value has a given bit set
pub fn isset8(val: u8, bit: u8) -> bool {
  val & bit != 0
}

/// Check if a 16-bit bit has been set
pub fn isset16(val: u16, bit: u16) -> bool {
  val & bit != 0
}

/// Check if a 32-bit bit has been set
pub fn isset32(val: u32, bit: u32) -> bool {
  val & bit != 0
}

/**
 * Read only chunk of memory loaded as bytes
 */
#[derive(Serialize, Deserialize)]
pub struct RomChunk {
  pub bytes: Vec<u8>,
}

impl RomChunk {
  fn read_u8(&self, address: u16) -> u8 {
    self.bytes[address as usize]
  }
  pub fn from_file(path: &str) -> io::Result<RomChunk> {
    info!("Loading {}", path);
    let mut f = File::open(path)?;
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer)?;
    Ok(RomChunk { bytes: buffer })
  }

  pub fn empty() -> RomChunk {
    RomChunk { bytes: Vec::new() }
  }

  fn wide_read_u8(&self, address: usize) -> u8 {
    self.bytes[address]
  }
}

/**
 * RAM read/write memory as bytes
 */
#[derive(Serialize, Deserialize)]
pub struct RamChunk {
  pub bytes: Vec<u8>,
}

impl RamChunk {
  fn write_u8(&mut self, address: u16, v: u8) {
    self.bytes[address as usize] = v;
  }
  fn read_u8(&self, address: u16) -> u8 {
    self.bytes[address as usize]
  }
}

impl RamChunk {
  pub fn new(size: usize) -> RamChunk {
    RamChunk {
      bytes: vec![0; size],
    }
  }

  fn wide_write_u8(&mut self, address: usize, v: u8) {
    self.bytes[address] = v;
  }

  fn wide_read_u8(&self, address: usize) -> u8 {
    self.bytes[address]
  }
}

#[derive(Serialize, Deserialize)]
pub struct GameboyState {
  boot: RomChunk,
  cart: RomChunk,
  cart_ram: RamChunk,
  vram: RamChunk,
  iram: RamChunk,
  high_ram: RamChunk,
  boot_enabled: bool,
  ram_on: bool,
  ram_mode: bool,
  gamepad_high: bool,

  pub last_clock: usize,

  pub cart_type: u8,

  /**
   * Current timer state
   * The div register is really interesting. While reads only give you the lower byte, under the
   * hood it's actually 16 bits wide and impacts the div and tima.
   * We store it here so it can be accessed both by the timer module and read by read_u8.
   */
  pub div: u16,

  /**
   * Rom and Ram bank settings
   */
  pub rom_bank: usize,
  pub ram_bank: usize,

  /**
   * Values to track the gamepad state
   */
  pub a: bool,
  pub b: bool,
  pub start: bool,
  pub select: bool,
  pub left: bool,
  pub right: bool,
  pub up: bool,
  pub down: bool,
}

impl GameboyState {
  pub fn new(boot: RomChunk, cart: RomChunk) -> GameboyState {
    GameboyState {
      cart_type: cart.read_u8(0x147),

      boot: boot,
      cart: cart,
      cart_ram: RamChunk::new(0x8000),
      vram: RamChunk::new(0x2000),
      iram: RamChunk::new(0x2000),
      high_ram: RamChunk::new(0x200),
      boot_enabled: true,
      ram_on: false,
      ram_mode: false,
      last_clock: 0,

      /// The divider starts at zero
      div: 0,

      /**
       * Rom bank defaults
       */
      rom_bank: 1,
      ram_bank: 0,

      /**
       * Gamepad default config
       */
      a: false,
      b: false,
      start: false,
      select: false,
      left: false,
      right: false,
      up: false,
      down: false,
      gamepad_high: false,
    }
  }

  /// Convert the current gamepad state into it's gameboy register representation (so that the running program can read it)
  fn gamepad_state(&self) -> u8 {
    const A_BUTTON: u8 = 1;
    const B_BUTTON: u8 = 1 << 1;
    const SELECT: u8 = 1 << 2;
    const START: u8 = 1 << 3;

    const RIGHT: u8 = 1;
    const LEFT: u8 = 1 << 1;
    const UP: u8 = 1 << 2;
    const DOWN: u8 = 1 << 3;

    let mut pad_state = 0;

    if !self.gamepad_high {
      //A, B, Select, Start
      if !self.a {
        pad_state = set8(pad_state, A_BUTTON);
      }
      if !self.b {
        pad_state = set8(pad_state, B_BUTTON);
      }
      if !self.select {
        pad_state = set8(pad_state, SELECT);
      }
      if !self.start {
        pad_state = set8(pad_state, START);
      }
    } else {
      //Right left up down
      if !self.left {
        pad_state = set8(pad_state, LEFT);
      }
      if !self.right {
        pad_state = set8(pad_state, RIGHT);
      }
      if !self.up {
        pad_state = set8(pad_state, UP);
      }
      if !self.down {
        pad_state = set8(pad_state, DOWN);
      }
    }
    pad_state
  }

  /// Convert a write to the gamepad register to an internal representation
  fn gamepad_write(&mut self, val: u8) {
    const DPAD_BIT: u8 = 1 << 4;
    const BUTTONS_BIT: u8 = 1 << 5;
    if isset8(val, DPAD_BIT) {
      self.gamepad_high = false;
    } else if isset8(val, BUTTONS_BIT) {
      self.gamepad_high = true;
    }
  }

  /// Calculates the RAM bank that should be activated
  fn set_ram_bank(&mut self, bank: u8) {
    if self.ram_mode {
      self.ram_bank = (bank & 0x3) as usize;
      info!("Set ram bank to {}", self.ram_bank);
    } else if is_mbc3(self.cart_type) {
      self.set_rom_bank_upper(bank);
    }
  }

  /// Calculate the ROM bank that should be activated given a write to a ROM-bank set region of the memory.
  fn set_rom_bank(&mut self, bank: u8) {
    if is_mbc1(self.cart_type) {
      let bank = (bank & 0b00011111) as usize; // Truncate to 5 bits
      let bank = if bank == 0 { 1 } else { bank };
      self.rom_bank = bank;
      info!("Set rom bank to {}", self.rom_bank);
    } else if is_mbc3(self.cart_type) {
      let bank = (bank & 0b01111111) as usize; // Truncate to 7 bits
      let bank = if bank == 0 { 1 } else { bank };
      self.rom_bank = bank;
      info!("Set rom bank to {}", self.rom_bank);
    }
  }

  fn set_rom_bank_upper(&mut self, bank: u8) {
    let bank = (bank & 0x3) << 5;
    let bank = bank as usize + self.rom_bank;
    self.rom_bank = bank;
    info!("set rom bank upper 2 bits {} {}", bank, self.rom_bank);
  }
}

impl GameboyState {
  fn write_high_mem(&mut self, address: u16, val: u8) {
    if address == GAMEPAD_ADDRESS {
      self.gamepad_write(val);
    } else if address == 0xFF02 && val == 0x81 {
      print!("{}", self.read_u8(0xFF01) as char);
    } else if address == DIV_REGISTER {
      self.div = 0;
    } else if address == TIMA_REGISTER {
      self.high_ram.write_u8(address - END_OF_ECHO_RAM, val);
    } else if address == TAC_REGISTER {
      self.high_ram.write_u8(address - END_OF_ECHO_RAM, val & 0x7);
    } else if address == 0xFF46 {
      // OAM DMA Mode. TODO: Read up on this and do it better
      for i in 0..160 {
        let v = self.read_u8(((val as u16) << 8) + i);
        self.high_ram.write_u8(i as u16, v);
      }
    } else if address == BOOT_ROM_ADDRESS {
      // Writing a 1 to this register disables the boot rom
      self.boot_enabled = false;
    } else if address == STAT {
      let new_stat = stat_interrupts_with_masked_flags(val, self);
      self.high_ram.write_u8(address - END_OF_ECHO_RAM, new_stat);
    } else {
      self.high_ram.write_u8(address - END_OF_ECHO_RAM, val);
    }
  }

  pub fn core_write(&mut self, address: u16, val: u8) {
    trace!("write {:x} to {:x}", val, address);
    if address < END_OF_BANKED_ROM {
      if address < 0x2000 {
        self.ram_on = val == 0x0A;
      } else if address >= 0x2000 && address < 0x4000 {
        // Writes to this area of ROM memory trigger a bank change
        self.set_rom_bank(val);
      } else if address >= 0x4000 && address < 0x6000 {
        if self.ram_mode {
          self.set_ram_bank(val);
        } else {
          self.set_rom_bank_upper(val);
        }
      } else if address < 0x8000 {
        info!("ram_mode");
        self.ram_mode = val != 0;
      }
    } else if address < END_OF_VRAM {
      self.vram.write_u8(address - END_OF_BANKED_ROM, val)
    } else if address < END_OF_CARTRIDGE_RAM {
      let address = (address - END_OF_VRAM) as usize;
      let address = address + (RAM_BANK_SIZE * self.ram_bank);
      self.cart_ram.wide_write_u8(address, val)
    } else if address < END_OF_INTERNAL_RAM {
      self.iram.write_u8(address - END_OF_CARTRIDGE_RAM, val)
    } else if address < END_OF_ECHO_RAM {
      // TODO: mirror ram, do I need?
      error!("illegal write to {:x}", address);
      unimplemented!();
    } else {
      self.write_high_mem(address, val);
    }
  }

  pub fn write_u8(&mut self, address: u16, val: u8, registers: &Registers) {
    if address == 0xFF10 {
      println!(
        "CH 1 SWEEP {} AT {}",
        val,
        registers.total_clock - self.last_clock
      );
      self.last_clock = registers.total_clock;
    }

    if address == 0xFF11 {
      println!(
        "CH 1 DUTYLL {} AT {}",
        val,
        registers.total_clock - self.last_clock
      );
      self.last_clock = registers.total_clock;
    }

    if address == 0xFF12 {
      println!(
        "CH 1 VOLENVPER {} AT {}",
        val,
        registers.total_clock - self.last_clock
      );
      self.last_clock = registers.total_clock;
    }

    if address == 0xFF13 {
      println!(
        "CH 1 FREQLSB {} AT {}",
        val,
        registers.total_clock - self.last_clock
      );
      self.last_clock = registers.total_clock;
    }

    if address == 0xFF14 {
      if isset8(val, 0b0100_0000) {
        println!(
          "CH 1 LENGTH_ENABLE AT {}",
          registers.total_clock - self.last_clock
        );
      } else {
        println!(
          "CH 1 LENGTH_DISABLE AT {}",
          registers.total_clock - self.last_clock
        )
      }

      if isset8(val, 0b1000_0000) {
        println!("CH 1 TRIGGER AT {}", 0);
      }

      self.last_clock = registers.total_clock;
    }

    self.core_write(address, val)
  }

  /// Write a special register exactly, ignoring any rules for that memory address such as writes
  /// truncate to 0.
  pub fn write_special_register(&mut self, address: u16, val: u8) {
    if address >= END_OF_ECHO_RAM {
      self.high_ram.write_u8(address - END_OF_ECHO_RAM, val);
    }
  }

  /// Read u8 will appear in traces but core_read won't, this reduces the noise a little debugging
  /// TODO: Rename
  pub fn core_read(&self, address: u16) -> u8 {
    if address < END_OF_FIXED_ROM {
      if self.boot_enabled && address < END_OF_BOOT {
        return self.boot.read_u8(address);
      }
      self.cart.read_u8(address)
    } else if address < END_OF_BANKED_ROM {
      let address = (address - END_OF_FIXED_ROM) as usize;
      let bank_offset = self.rom_bank * ROM_BANK_SIZE;
      trace!("banked read offset {} {}", self.rom_bank, bank_offset);
      self.cart.wide_read_u8(bank_offset + address)
    } else if address < END_OF_VRAM {
      self.vram.read_u8(address - END_OF_BANKED_ROM)
    } else if address < END_OF_CARTRIDGE_RAM {
      let address = (address - END_OF_VRAM) as usize;
      let address = address + (RAM_BANK_SIZE * self.ram_bank);
      trace!("cart ram: {}", address);
      self.cart_ram.wide_read_u8(address)
    } else if address < END_OF_INTERNAL_RAM {
      self.iram.read_u8(address - END_OF_CARTRIDGE_RAM)
    } else if address < END_OF_ECHO_RAM {
      warn!(
        "echo ram read: {} {}",
        address,
        address - END_OF_INTERNAL_RAM
      );
      self.iram.read_u8(address - END_OF_INTERNAL_RAM)
    } else {
      if address == GAMEPAD_ADDRESS {
        self.gamepad_state()
      } else if address == DIV_REGISTER {
        // The high bits of the div register are the value read (essentially it's a rolling
        // increment of the clock signals upper byte, so it will be 1 after 256 cycles)
        (self.div >> 8) as u8
      } else {
        self.high_ram.read_u8(address - END_OF_ECHO_RAM)
      }
    }
  }

  /// Read a byte of state
  pub fn read_u8(&self, address: u16) -> u8 {
    let result = self.core_read(address);
    trace!("read {:x} {:x}", address, result);
    result
  }

  pub fn write_u16(&mut self, address: u16, value: u16, registers: &Registers) {
    let lower = value & 0xFF;
    let upper = value >> 8;
    self.write_u8(address + 1, upper as u8, registers);
    self.write_u8(address, lower as u8, registers);
  }

  pub fn read_u16(&mut self, address: u16) -> u16 {
    let upper = self.read_u8(address + 1);
    let lower = self.read_u8(address);
    let result = ((upper as u16) << 8) + (lower as u16);
    result
  }
}
