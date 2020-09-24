use std::rc::Rc;

use crate::cpu::{CPU, INTERRUPTS_HAPPENED_ADDRESS, VBLANK};
use crate::memory::MemoryChunk;

use log::{trace, info};

pub const GB_SCREEN_WIDTH: u32 = 160;
pub const GB_SCREEN_HEIGHT: u32 = 144;
pub const BYTES_PER_PIXEL: u32 = 3;
pub const BYTES_PER_ROW: u32 = GB_SCREEN_WIDTH * BYTES_PER_PIXEL;

const TILESET_ONE_ADDR: u16 = 0x8000;

const SCY: u16 = 0xFF42;
const SCX: u16 = 0xFF43;
const LCD_CONTROL: u16 = 0xFF40;
const MAP: u16 = 0x9800;
const BGMAP: u16 = 0x9C00;

const CURRENT_SCANLINE: u16 = 0xFF44;

#[derive(Debug, Copy, Clone)]
enum Mode {
  OAM,
  VRAM,
  HBLANK,
  VBLANK,
}

pub enum GpuStepState {
  None,
  VBlank,
  HBlank,
}

pub struct GPU {
  cycles_in_mode: u16,
  current_line: u8,
  mode: Mode,
}

impl GPU {

  pub fn new() -> GPU {
    info!("GPU initialized");
    GPU {
      cycles_in_mode: 0,
      current_line: 0,
      mode: Mode::OAM,
    }
  }

  fn tile_value(&self, id: u16, x: u16, y: u16, mem: &mut Box<dyn MemoryChunk>) -> u8 {
    const TILE_SIZE: u16 = 16;
    let tile_addr = TILESET_ONE_ADDR + (TILE_SIZE * id);
    let y_addr = tile_addr + (y * 2);
    let mask_x = 1 << (7 - x);
    let low_bit = if mem.read_u8(y_addr) & mask_x != 0 { 1 } else { 0 };
    let high_bit = if mem.read_u8(y_addr + 1) & mask_x != 0 { 2 } else { 0 };
    low_bit + high_bit
  }

  fn lcd_control(&self, mem: &mut Box<dyn MemoryChunk>) -> u8 {
    mem.read_u8(LCD_CONTROL)
  }

  fn window(&self, mem: &mut Box<dyn MemoryChunk>) -> bool {
    self.lcd_control(mem) & (1 << 5) != 0
  }

  fn bgmap(&self, mem: &mut Box<dyn MemoryChunk>) -> bool {
    self.lcd_control(mem) & (1 << 3) != 0
  }

  fn bgtile(&self, mem: &mut Box<dyn MemoryChunk>) -> bool {
    self.lcd_control(mem) & (1 << 4) != 0
  }

  fn enter_mode(&mut self, mode: Mode) {
    self.cycles_in_mode = 0;
    self.mode = mode;
  }

  fn write_px(pixels: &mut [u8], x: u8, y: u8, val: u8) {
    let x = x as usize;
    let y = y as usize;
    let canvas_offset = (GB_SCREEN_WIDTH * 3) as usize * y;
    pixels[(x * 3) + canvas_offset] = val;
    pixels[(x * 3) + canvas_offset + 1] = val;
    pixels[(x * 3) + canvas_offset + 2] = val;
  }

  fn fetch_tile(&self, addr: u16, mem: &mut Box<dyn MemoryChunk>) -> u16 {
    let tile = mem.read_u8(addr) as u16;
    if !self.bgtile(mem) && tile < 128 {
      tile + 256
    } else {
      tile
    }
  }

  fn scx(&self, mem: &mut Box<dyn MemoryChunk>) -> u8 {
    mem.read_u8(SCX)
  }

  fn scy(&self, mem: &mut Box<dyn MemoryChunk>) -> u8 {
    mem.read_u8(SCY)
  }

  fn render_line(&mut self, mem: &mut Box<dyn MemoryChunk>, pixels: &mut [u8]) {
    trace!("Rendering full line {}", self.current_line);

    let scy = self.scy(mem);
    let scx = self.scx(mem);
    let window = self.window(mem);
    let background_map_selected = self.bgmap(mem);

    let map_line = scy + self.current_line;
    let map_line_offset = ((map_line as u16) >> 3) * 32;

    let map_offset = if background_map_selected {
      BGMAP
    } else {
      MAP
    } + map_line_offset;

    info!("SCY: {} SCX: {} WINDOW: {} BGM: {} MAP_LINE: {:x} OFFSET: {:x} MO: {:x}", scy, scx, window, background_map_selected, map_line, map_line_offset, map_offset);

    let mut line_offset = (scx >> 3) as u16;
    let mut tile = self.fetch_tile(map_offset + line_offset, mem);

    let mut x = scx & 7;
    let y = ((self.current_line + scy) & 7) as u16;

    for i in 0..160 {
      let val = self.tile_value(tile, x as u16, y as u16, mem);
      if val != 0 {
        info!("VAL NOT ZERO!");
      }
      GPU::write_px(pixels, i, self.current_line, if val != 0 { 0 } else { 255 });
      x += 1;
      if x == 8 {
        x = 0;
        line_offset = (line_offset + 1) & 31;
        tile = self.fetch_tile(map_offset + line_offset, mem);
      }
    }
  }

  fn update_scanline(&mut self, mem: &mut Box<dyn MemoryChunk>) {
    mem.write_u8(CURRENT_SCANLINE, self.current_line as u8);
    info!("CURRENT SCANLINE: {}", mem.read_u8(CURRENT_SCANLINE));
  }

  pub fn step(&mut self, cpu: &mut CPU, mem: &mut Box<dyn MemoryChunk>, draw: &mut [u8]) -> GpuStepState {
    self.cycles_in_mode += cpu.registers.last_clock;
    //trace!("GPU mode {:?} step by {} to {}", self.mode, cpu.registers.last_clock, self.cycles_in_mode);
    let current_mode = self.mode;
    match current_mode {
      Mode::OAM => {
        if self.cycles_in_mode >= 80 {
          self.enter_mode(Mode::VRAM);
        }
        GpuStepState::None
      },
      Mode::VRAM => {
        if self.cycles_in_mode >= 172 {
          self.render_line(mem, draw);
          self.enter_mode(Mode::HBLANK);
        }
        GpuStepState::None
      },
      Mode::HBLANK => {
        if self.cycles_in_mode >= 204 {
          self.current_line += 1;
          self.update_scanline(mem);
          if self.current_line == 144 {
            self.enter_mode(Mode::VBLANK);
          } else {
            self.enter_mode(Mode::OAM);
          }
        }
        GpuStepState::HBlank
      },
      Mode::VBLANK => {
        if self.cycles_in_mode % 204 == 0 {
          self.current_line += 1;
          self.update_scanline(mem);
        }

        if self.cycles_in_mode > 4560 {
          self.current_line = 0;
          mem.write_u8(
            INTERRUPTS_HAPPENED_ADDRESS,
            mem.read_u8(INTERRUPTS_HAPPENED_ADDRESS) | VBLANK
          );
          self.enter_mode(Mode::OAM);
          GpuStepState::VBlank
        } else {
          GpuStepState::None
        }
      },
    }
  }
}
