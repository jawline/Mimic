use crate::instruction::InstructionSet;
use crate::memory::{isset8, set8, unset8, GameboyState};
use crate::register::{
  RegisterPair, SmallWidthRegister, SmallWidthRegister::*, WideRegister, WideRegister::*,
};
use log::{debug, trace};
use serde::{Deserialize, Serialize};

pub const INTERRUPTS_ENABLED_ADDRESS: u16 = 0xFFFF;
pub const INTERRUPTS_HAPPENED_ADDRESS: u16 = 0xFF0F;

/// The bit set if the VBLANK interrupt has fired
pub const VBLANK: u8 = 0x1;

/// The bit is set of the STAT interrupt has fired
pub const STAT: u8 = 0x1 << 1;

/// The bit set if the TIMER interrupt has fired
pub const TIMER: u8 = 0x1 << 2;

/// The location to jump to for a VBLANK interrupt
pub const VBLANK_ADDRESS: u16 = 0x40;

/// The location to jump to for a STAT interrupt
pub const STAT_ADDRESS: u16 = 0x48;

/// The location to jump to for a TIMER interrupt
pub const TIMER_ADDRESS: u16 = 0x50;

/// The bit set if the JOYPAD PRESSED interrupt has fired
pub const JOYPAD: u8 = 0x1 << 4;

/// The interrupt address for a joypad pressed interrupt
pub const JOYPAD_ADDRESS: u16 = 0x60;

/// CPU state and registers for a Z80 gameboy processor.
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct Registers {
  pc: u16,
  sp: u16,
  bc: RegisterPair,
  af: RegisterPair,
  de: RegisterPair,
  hl: RegisterPair,

  /// This indicates the last opcode processed was an escape opcode, triggering the extended instruction set
  /// While this flag is true we should not process interrupts
  pub escaped: bool,

  /// Is the CPU halted
  pub halted: bool,

  /// Interrupts master enable flag
  pub ime: bool,

  /// How many cycles passed in the last CPU step
  pub last_clock: u16,
  pub total_clock: usize,
}

/// The position of flags in the F register
pub const ZERO_FLAG: u8 = 0x80;
pub const SUBTRACT_FLAG: u8 = 0x40;
pub const HALF_CARRY_FLAG: u8 = 0x20;
pub const CARRY_FLAG: u8 = 0x10;
pub const FLAGS_MASK: u8 = ZERO_FLAG | HALF_CARRY_FLAG | SUBTRACT_FLAG | CARRY_FLAG;

impl Registers {
  /// Get the current program counter
  pub fn pc(&self) -> u16 {
    self.read_r16(WideRegister::PC)
  }

  /// Get the current stack pointer
  pub fn sp(&self) -> u16 {
    self.read_r16(WideRegister::SP)
  }

  /// Set the program counter to a given value
  pub fn set_pc(&mut self, pc: u16) {
    self.write_r16(WideRegister::PC, pc);
  }

  /// Set the stack pointer to a given value
  pub fn set_sp(&mut self, sp: u16) {
    self.write_r16(WideRegister::SP, sp);
  }

  /// Add a 16-bit value to the program counter
  pub fn inc_pc(&mut self, by: u16) {
    self.set_pc(self.pc() + by);
  }

  /// Push a 16 bit value from the stack and return it
  pub fn stack_push16<'a>(&mut self, value: u16, memory: &mut GameboyState) {
    let new_sp = self.sp().wrapping_sub(2);
    memory.write_u16(new_sp, value, &self);
    self.set_sp(new_sp);
  }

  /// Pop a 16 bit value from the stack and return it
  pub fn stack_pop16<'a>(&mut self, memory: &mut GameboyState) -> u16 {
    let sp = self.sp();
    let rval = memory.read_u16(sp);
    self.set_sp(sp.wrapping_add(2));
    rval
  }

  /// Increase or decrease the program counter by a signed 8-bit integer.
  pub fn jump_relative(&mut self, by: i8) {
    // TODO: Check this works for negative values
    let by = by as i16;
    let new_location = self.pc.wrapping_add(by as u16);
    trace!("relative jump {} by {} -> {}", self.pc, by, new_location);
    self.set_pc(new_location);
  }

  /// Read an 8-bit register
  pub fn read_r8(&self, reg: SmallWidthRegister) -> u8 {
    match reg {
      B => self.bc.l,
      C => self.bc.r,
      A => self.af.l,
      F => self.af.r,
      D => self.de.l,
      E => self.de.r,
      H => self.hl.l,
      L => self.hl.r,
      SmallUnset => panic!("read small unset"),
    }
  }

  /// Write an 8-bit register
  pub fn write_r8(&mut self, reg: SmallWidthRegister, val: u8) {
    match reg {
      B => self.bc.l = val,
      C => self.bc.r = val,
      A => self.af.l = val,
      F => self.af.r = val & FLAGS_MASK,
      D => self.de.l = val,
      E => self.de.r = val,
      H => self.hl.l = val,
      L => self.hl.r = val,
      SmallUnset => panic!("write small unset"),
    };
  }

  /// Read a 16-bit register (BC, AF, DE, and HL are the joins of their 8-bit registers)
  pub fn read_r16(&self, reg: WideRegister) -> u16 {
    match reg {
      PC => self.pc,
      SP => self.sp,
      BC => self.bc.as_u16(),
      AF => self.af.as_u16(),
      DE => self.de.as_u16(),
      HL => self.hl.as_u16(),
      WideUnset => panic!("access bad wide register"),
    }
  }

  /// Write a 16-bit register (BC, AF, DE, and HL are the joins of their 8-bit registers)
  pub fn write_r16(&mut self, reg: WideRegister, val: u16) {
    trace!("Writing {:x} -> {:?}", val, reg);
    match reg {
      PC => self.pc = val,
      SP => {
        self.sp = val;
      }
      BC => self.bc.write_u16(val),
      AF => {
        self.af.write_u16(val);
        // Mask the flags register because the other 4 bits are unusable
        self.af.r = self.af.r & FLAGS_MASK
      }
      DE => self.de.write_u16(val),
      HL => self.hl.write_u16(val),
      WideUnset => panic!("write bad wide register"),
    };
  }

  /// Fetch the current processor flags
  pub fn flags(&self) -> u8 {
    self.read_r8(SmallWidthRegister::F)
  }

  /// Return true if the zero flag is set
  pub fn zero(&self) -> bool {
    isset8(self.flags(), ZERO_FLAG)
  }

  /// Return true if the negative (subtract) flag is set
  pub fn subtract(&self) -> bool {
    isset8(self.flags(), SUBTRACT_FLAG)
  }

  /// Returns true if the half carry flag is set
  pub fn half_carry(&self) -> bool {
    isset8(self.flags(), HALF_CARRY_FLAG)
  }

  /// Return true of the carry flag is set
  pub fn carry(&self) -> bool {
    isset8(self.flags(), CARRY_FLAG)
  }

  /// Helper function to set or unset a given flag using it's bit mask
  fn set_flag(flags: u8, flag: u8, set: bool) -> u8 {
    match set {
      true => set8(flags, flag),
      false => unset8(flags, flag),
    }
  }

  /// Set the processor current flags to these states
  pub fn set_flags(&mut self, zero: bool, negative: bool, half_carry: bool, carry: bool) {
    trace!(
      "Set flags to Z: {} N: {} H: {} C: {}",
      zero,
      negative,
      half_carry,
      carry
    );
    let mut current_flags = self.read_r8(SmallWidthRegister::F);
    current_flags = Registers::set_flag(current_flags, CARRY_FLAG, carry);
    current_flags = Registers::set_flag(current_flags, HALF_CARRY_FLAG, half_carry);
    current_flags = Registers::set_flag(current_flags, SUBTRACT_FLAG, negative);
    current_flags = Registers::set_flag(current_flags, ZERO_FLAG, zero);
    self.write_r8(SmallWidthRegister::F, current_flags);
  }
}

#[derive(Serialize, Deserialize)]
pub struct Cpu {
  pub registers: Registers,
}

impl Cpu {
  pub fn new() -> Self {
    Self {
      registers: Registers::default(),
    }
  }

  /// Trigger a specific interrupt (disable IME, push PC to stack and jump to interrupt handler)
  fn fire_interrupt(&mut self, location: u16, memory: &mut GameboyState) {
    self.registers.ime = false;
    self.registers.stack_push16(self.registers.pc(), memory);
    self.registers.set_pc(location);
  }

  /// Check if there are any interrupts waiting to fire
  pub fn check_interrupt(&mut self, memory: &mut GameboyState) {
    let triggered = memory.read_u8(INTERRUPTS_HAPPENED_ADDRESS);

    // If any interrupt is triggered then unhalt the processor.
    if triggered != 0 {
      self.registers.halted = false;
    }

    if self.registers.ime {
      let enabled = memory.read_u8(INTERRUPTS_ENABLED_ADDRESS);

      trace!("ENABLED INTERRUPTS {:b}", enabled);
      trace!("TRIGGERED INTERRUPTS {:b}", triggered);

      let interrupted = triggered & enabled;

      // We can only execute one interrupt at a time.
      // When an interrupt fired we clear its interrupt bit in the IF
      // we then disable the interrupt enable flag and push the PC to the stack
      // finally we jump to the interrupt handler.
      // The interrupt precedence is decided by the priority which is expressed in
      // the order of if-else statements.
      if isset8(interrupted, VBLANK) {
        debug!("VBLANK INTERRUPT");
        Self::clear_interrupt_happened(memory, VBLANK, &self.registers);
        self.fire_interrupt(VBLANK_ADDRESS, memory);
      } else if isset8(interrupted, STAT) {
        debug!("STAT INTERRUPT");
        Self::clear_interrupt_happened(memory, STAT, &self.registers);
        self.fire_interrupt(STAT_ADDRESS, memory);
      } else if isset8(interrupted, TIMER) {
        debug!("TIMER INT");
        Self::clear_interrupt_happened(memory, TIMER, &self.registers);
        self.fire_interrupt(TIMER_ADDRESS, memory);
      } else if isset8(interrupted, JOYPAD) {
        debug!("JOYPAD PRESSED");
        Self::clear_interrupt_happened(memory, JOYPAD, &self.registers);
        self.fire_interrupt(JOYPAD_ADDRESS, memory);
      }
    }
  }

  /// Clear an interrupt bit in the interrupts that have triggered register
  pub fn clear_interrupt_happened(memory: &mut GameboyState, interrupt: u8, registers: &Registers) {
    memory.write_u8(
      INTERRUPTS_HAPPENED_ADDRESS,
      memory.read_u8(INTERRUPTS_HAPPENED_ADDRESS) & !interrupt,
      registers,
    );
  }

  /// Set an interrupt triggered bit in memory
  pub fn set_interrupt_happened(memory: &mut GameboyState, interrupt: u8, registers: &Registers) {
    memory.write_u8(
      INTERRUPTS_HAPPENED_ADDRESS,
      memory.read_u8(INTERRUPTS_HAPPENED_ADDRESS) | interrupt,
      registers,
    );
  }

  /// Step the emulator by a single instruction
  pub fn step(&mut self, memory: &mut GameboyState, instructions: &InstructionSet) {
    if !self.registers.halted {
      let opcode = memory.core_read(self.registers.pc());

      let inst;

      if self.registers.escaped {
        trace!("Selected opcode from extended set since escaped is set");
        inst = &instructions.ext_instructions[opcode as usize];
        self.registers.escaped = false;
      } else {
        inst = &instructions.instructions[opcode as usize];
      }

      debug!(
        "INSTR={} PC={:x} SP={:x} BC={:x} AF={:x} DE={:x} HL={:x}\n B={:x} C={:x} A={:x} F={:x} D={:x} E={:x} H={:x} L={:x} Z={} N={} H={} C={} HALTED={} IME={}",
        inst.text,
        self.registers.pc(),
        self.registers.sp(),
        self.registers.read_r16(WideRegister::BC),
        self.registers.read_r16(WideRegister::AF),
        self.registers.read_r16(WideRegister::DE),
        self.registers.read_r16(WideRegister::HL),
        self.registers.bc.l, self.registers.bc.r,
        self.registers.af.l, self.registers.af.r,
        self.registers.de.l, self.registers.de.r,
        self.registers.hl.l, self.registers.hl.r,
        self.registers.zero(), self.registers.subtract(), self.registers.half_carry(), self.registers.carry(), self.registers.halted, self.registers.ime
      );

      trace!("{} ({:x})", inst.text, opcode);
      self.registers.last_clock = 0;
      (inst.execute)(&mut self.registers, memory);
      //trace!("post-step: {:?}", self.registers);
      // Some instructions mutate the last clock like JR
      self.registers.last_clock += inst.cycles;
    } else {
      self.registers.last_clock = 4;
    }

    if !self.registers.escaped {
      // if ime is flagged on and there is an interrupt waiting then trigger it
      self.check_interrupt(memory);
    }
  }
}
