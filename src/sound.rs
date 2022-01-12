use crate::cpu::{Cpu, STAT, VBLANK};
use crate::memory::{isset8, GameboyState};
use crate::util::{self, stat};
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};

pub struct Sound {
}

impl Sound {
    pub fn step(&mut self, cpu: &mut Cpu, mem: &mut GameboyState) {
    }
}
