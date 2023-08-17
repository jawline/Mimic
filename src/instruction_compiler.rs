pub enum Address {
  Absolute(u16),
}

impl Address {
  pub fn to_u16(&self) -> u16 {
    match self {
      Address::Absolute(value) => *value,
    }
  }
}

pub enum Program {
  Jump { address: Address },
  Call { address: Address },
  EnableInterrupts,
  SetA { immediate: u8 },
}

impl Program {
  pub fn emit(&self) -> Vec<u8> {
    match self {
      Program::Jump { address } => {
        let address = address.to_u16();
        vec![0xC3, address.to_le_bytes()[0], address.to_le_bytes()[1]]
      }
      Program::Call { address } => {
        let address = address.to_u16();
        vec![0xCD, address.to_le_bytes()[0], address.to_le_bytes()[1]]
      }
      Program::EnableInterrupts => vec![0xFB],
      Program::SetA { immediate } => vec![0x3E, *immediate],
    }
  }
}

pub fn to_machine_code(program: &[Program]) -> Vec<u8> {
  program.iter().map(|p| p.emit()).flatten().collect()
}

pub fn write_machine_code<F>(program: &[Program], mut f: F)
where
  F: FnMut(u16, u8),
{
  let code = to_machine_code(program);
  for (index, byte) in code.iter().enumerate() {
    f(index as u16, *byte)
  }
}
