use serde::{Deserialize, Serialize};

/// Represents a register pair that can be addressed either as two u8's or one u16
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct RegisterPair {
  pub l: u8,
  pub r: u8,
}

impl RegisterPair {
  pub fn as_u16(&self) -> u16 {
    let high_portion = (self.l as u16) << 8;
    let low_portion = self.r as u16;
    high_portion + low_portion
  }

  pub fn write_u16(&mut self, v: u16) {
    self.l = (v >> 8) as u8;
    self.r = (v & 0xFF) as u8;
  }
}

/// Enum to address all the 8-bit registers
#[derive(Debug, Clone, Copy)]
pub enum SmallWidthRegister {
  B,
  C,
  A,
  F,
  D,
  E,
  H,
  L,
  SmallUnset, // Used to identify errors in instructions
}

/// Enum to address all the 16-bit wide registers
#[derive(Debug, Clone, Copy)]
pub enum WideRegister {
  PC,
  SP,
  BC,
  AF,
  DE,
  HL,
  WideUnset,
}
