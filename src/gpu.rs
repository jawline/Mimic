use crate::cpu::{CPU, VBLANK};
use crate::memory::MemoryPtr;

use log::{info, trace};

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
const OAM: u16 = 0xFE00;
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

struct Sprite {
  pub x: i16,
  pub y: i16,
  pub tile: u8,
  pub palette: bool,
  pub xflip: bool,
  pub yflip: bool,
  pub prio: bool,
}

impl Sprite {
  fn fetch(id: u16, mem: &mut MemoryPtr) -> Self {
    let address = OAM + (id * 4);
    let y = (mem.read_u8(address) as i16) - 16;
    let x = (mem.read_u8(address + 1) as i16) - 8;
    let tile = mem.read_u8(address + 2);
    let meta = mem.read_u8(address + 3);
    Sprite {
      x: x,
      y: y,
      tile: tile,
      palette: meta & (1 << 4) != 0,
      xflip: meta & (1 << 5) != 0,
      yflip: meta & (1 << 6) != 0,
      prio: meta & (1 << 7) == 0,
    }
  }
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

  fn tile_value(&self, id: u16, x: u16, y: u16, mem: &mut MemoryPtr) -> u8 {
    const TILE_SIZE: u16 = 16;
    let tile_addr = TILESET_ONE_ADDR + (TILE_SIZE * id);
    let y_addr = tile_addr + (y * 2);
    let mask_x = 1 << (7 - x);
    let low_bit = if mem.read_u8(y_addr) & mask_x != 0 {
      1
    } else {
      0
    };
    let high_bit = if mem.read_u8(y_addr + 1) & mask_x != 0 {
      2
    } else {
      0
    };
    low_bit + high_bit
  }

  fn lcd_control(&self, mem: &mut MemoryPtr) -> u8 {
    mem.read_u8(LCD_CONTROL)
  }

  fn show_background(lcd: u8) -> bool {
    lcd & 1 != 0
  }

  fn show_sprites(lcd: u8) -> bool {
    lcd & (1 << 1) != 0
  }

  fn window(lcd: u8) -> bool {
    lcd & (1 << 5) != 0
  }

  fn bgmap(lcd: u8) -> bool {
    lcd & (1 << 3) != 0
  }

  fn bgtile(lcd: u8) -> bool {
    lcd & (1 << 4) != 0
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

  fn fetch_tile(&self, addr: u16, bgtile: bool, mem: &mut MemoryPtr) -> u16 {
    let tile = mem.read_u8(addr) as u16;
    if !bgtile && tile < 128 {
      tile + 256
    } else {
      tile
    }
  }

  fn scx(mem: &mut MemoryPtr) -> u8 {
    mem.read_u8(SCX)
  }

  fn scy(mem: &mut MemoryPtr) -> u8 {
    mem.read_u8(SCY)
  }

  fn pal(v: u8) -> u8 {
    match v {
      0 => 255,
      1 => 160,
      2 => 80,
      _ => 0,
    }
  }

  fn render_line(&mut self, mem: &mut MemoryPtr, pixels: &mut [u8]) {
    trace!("Rendering full line {}", self.current_line);

    let lcd = self.lcd_control(mem);
    let scy = GPU::scy(mem);
    let scx = GPU::scx(mem);
    let render_bg = GPU::show_background(lcd);
    let render_sprites = GPU::show_sprites(lcd);
    let bgtile = GPU::bgtile(lcd);
    let window = GPU::window(lcd);
    let background_map_selected = GPU::bgmap(lcd);
    let mut hit = vec![false; GB_SCREEN_WIDTH as usize];

    if render_bg {
      let map_line = scy + self.current_line;
      let map_line_offset = ((map_line as u16) >> 3) * 32;

      let map_offset = if background_map_selected { BGMAP } else { MAP } + map_line_offset;

      trace!(
        "SCY: {} SCX: {} WINDOW: {} BGM: {} MAP_LINE: {:x} OFFSET: {:x} MO: {:x}",
        scy,
        scx,
        window,
        background_map_selected,
        map_line,
        map_line_offset,
        map_offset
      );

      let mut line_offset = (scx >> 3) as u16;
      let mut tile = self.fetch_tile(map_offset + line_offset, bgtile, mem);

      let mut x = scx & 7;
      let y = ((self.current_line + scy) & 7) as u16;

      for i in 0..GB_SCREEN_WIDTH {
        let val = self.tile_value(tile, x as u16, y as u16, mem);
        if val != 0 {
          hit[i as usize] = true;
        }
        GPU::write_px(pixels, i as u8, self.current_line, GPU::pal(val));
        x += 1;
        if x == 8 {
          x = 0;
          line_offset = (line_offset + 1) & 31;
          tile = self.fetch_tile(map_offset + line_offset, bgtile, mem);
        }
      }
    }

    if render_sprites {
      for i in 0..40 {
        let sprite = Sprite::fetch(i, mem);

        let hits_line_y =
          sprite.y <= self.current_line as i16 && sprite.y + 8 > self.current_line as i16;

        if hits_line_y {
          let tile_y = if sprite.yflip {
            7 - (self.current_line - sprite.y as u8)
          } else {
            self.current_line - sprite.y as u8
          };

          for x in 0..8 {
            let color = self.tile_value(sprite.tile as u16, x, tile_y as u16, mem);
            let low_x = sprite.x + x as i16;
            if low_x >= 0 && low_x < 160 && color != 0 && (sprite.prio || !hit[low_x as usize]) {
              let pval = GPU::pal(color);
              GPU::write_px(pixels, low_x as u8, self.current_line, pval);
            }
          }
        }
      }
    }
  }

  fn update_scanline(&mut self, mem: &mut MemoryPtr) {
    mem.write_u8(CURRENT_SCANLINE, self.current_line as u8);
  }

  pub fn step(&mut self, cpu: &mut CPU, mem: &mut MemoryPtr, draw: &mut [u8]) -> GpuStepState {
    self.cycles_in_mode += cpu.registers.last_clock;
    trace!(
      "GPU mode {:?} step by {} to {}",
      self.mode,
      cpu.registers.last_clock,
      self.cycles_in_mode
    );
    let current_mode = self.mode;
    match current_mode {
      Mode::OAM => {
        if self.cycles_in_mode >= 80 {
          self.enter_mode(Mode::VRAM);
        }
        GpuStepState::None
      }
      Mode::VRAM => {
        if self.cycles_in_mode >= 172 {
          self.render_line(mem, draw);
          self.enter_mode(Mode::HBLANK);
        }
        GpuStepState::None
      }
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
      }
      Mode::VBLANK => {
        if self.cycles_in_mode % 204 == 0 {
          self.current_line += 1;
          self.update_scanline(mem);
        }

        if self.cycles_in_mode > 4560 {
          self.current_line = 0;
          CPU::set_interrupt_happened(mem, VBLANK);
          self.enter_mode(Mode::OAM);
          GpuStepState::VBlank
        } else {
          GpuStepState::None
        }
      }
    }
  }
}
