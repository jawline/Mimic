use crate::register::{SmallWidthRegister, WideRegister};

/// We re-use some instruction functions for multiple register implementations
/// This struct carries data for the single-implementation for many opcode instruction methods
#[derive(Clone, Debug)]
pub struct InstructionData {
  pub code: u8,
  pub flag_mask: u8,
  pub flag_expected: u8,
  pub bit: u8,

  pub small_reg_one: SmallWidthRegister,
  pub small_reg_dst: SmallWidthRegister,

  pub wide_reg_one: WideRegister,
  pub wide_reg_dst: WideRegister,
}

/// When not in use fields get a default value. This can add some risk, if an instruction data is
/// improperly configured it can be hard to debug, I should revisit this at some point and consider
/// optional values with .unwrap, but it might not be worth it for the clarity.
impl Default for InstructionData {
  fn default() -> InstructionData {
    Self::const_default()
  }
}

impl InstructionData {
  pub const fn const_default() -> InstructionData {
    InstructionData {
      code: 0,
      bit: 0,
      flag_mask: 0,
      flag_expected: 0,
      small_reg_one: SmallWidthRegister::SmallUnset,
      small_reg_dst: SmallWidthRegister::SmallUnset,
      wide_reg_one: WideRegister::WideUnset,
      wide_reg_dst: WideRegister::WideUnset,
    }
  }

  pub const fn rst_n(code: u8) -> InstructionData {
    let mut m = InstructionData::const_default();
    m.code = code;
    m
  }

  pub const fn with_flag(mut self, mask: u8, expected: u8) -> InstructionData {
    self.flag_mask = mask;
    self.flag_expected = expected;
    self
  }

  pub const fn with_bit(mut self, bit: u8) -> InstructionData {
    self.bit = bit;
    self
  }

  pub const fn small_src(r: SmallWidthRegister) -> InstructionData {
    let mut a = InstructionData::const_default();
    a.small_reg_one = r;
    a
  }

  pub const fn small_dst(r: SmallWidthRegister) -> InstructionData {
    let mut a = InstructionData::const_default();
    a.small_reg_dst = r;
    a
  }

  pub const fn wide_dst(r: WideRegister) -> InstructionData {
    let mut a = InstructionData::const_default();
    a.wide_reg_dst = r;
    a
  }

  pub const fn wide_src(r: WideRegister) -> InstructionData {
    let mut a = InstructionData::const_default();
    a.wide_reg_one = r;
    a
  }

  pub const fn wide_dst_small_in(r: WideRegister, l: SmallWidthRegister) -> InstructionData {
    let mut a = InstructionData::wide_dst(r);
    a.small_reg_one = l;
    a
  }

  pub const fn wide_dst_wide_src(r: WideRegister, l: WideRegister) -> InstructionData {
    let mut a = InstructionData::wide_dst(r);
    a.wide_reg_one = l;
    a
  }

  pub const fn small_dst_wide_src(r: SmallWidthRegister, l: WideRegister) -> InstructionData {
    let mut a = InstructionData::small_dst(r);
    a.wide_reg_one = l;
    a
  }

  pub const fn small_dst_small_src(
    r: SmallWidthRegister,
    l: SmallWidthRegister,
  ) -> InstructionData {
    let mut a = InstructionData::small_dst(r);
    a.small_reg_one = l;
    a
  }
}
