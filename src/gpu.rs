use crate::cpu::{CPU, STAT, VBLANK};
use crate::memory::{isset8, MemoryPtr};
use crate::util::{self, stat};
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
const LYC_SCANLINE: u16 = 0xFF45;

const PAL_BG_REG: u16 = 0xFF47;
const PAL_OBJ0_REG: u16 = 0xFF48;
const PAL_OBJ1_REG: u16 = 0xFF49;

/// If these are set then the CPU should interrupt when the GPU does a state transition
const STAT_INTERRUPT_LYC_EQUALS_LC: u8 = 0x1 << 6;
const STAT_INTERRUPT_DURING_OAM: u8 = 0x1 << 5;
const STAT_INTERRUPT_DURING_V_BLANK: u8 = 0x1 << 4;
const STAT_INTERRUPT_DURING_H_BLANK: u8 = 0x1 << 3;

/// Set when LYC=LCDC LY
const STAT_COINCIDENCE_FLAG: u8 = 0x4;
const STAT_H_BLANK: u8 = 0x0;
const STAT_V_BLANK: u8 = 0x1;
const STAT_OAM: u8 = 0x2;
const STAT_TRANSFERRING_TO_LCD: u8 = 0x3;
const STAT_LYC_EQUALS_LCDC: u8 = 0x4;

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

#[derive(Debug)]
struct Sprite {
  pub x: u8,
  pub y: u8,
  pub tile: u8,
  pub palette: bool,
  pub xflip: bool,
  pub yflip: bool,
  pub prio: bool,
}

impl Sprite {
  fn address(id: u16) -> u16 {
    OAM + (id * 4)
  }

  fn pos(id: u16, mem: &mut MemoryPtr) -> (u8, u8) {
    let address = OAM + (id * 4);
    let y = mem.read_u8(address);
    let x = mem.read_u8(address + 1);
    (x - 8, y - 16)
  }

  fn visible(id: u16, mem: &mut MemoryPtr) -> bool {
    let (x, y) = Sprite::pos(id, mem);
    y != 0 && x != 0
  }

  fn fetch(id: u16, mem: &mut MemoryPtr) -> Self {
    let address = Sprite::address(id);
    let (x, y) = Sprite::pos(id, mem);
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

  fn try_fire(stat: u8, interrupt: u8, mem: &mut MemoryPtr) {
    if isset8(stat, interrupt) {
      CPU::set_interrupt_happened(mem, STAT);
    }
  }

  fn try_fire_stat_interrupt(&mut self, mode: Mode, mem: &mut MemoryPtr) {
    let stat = util::stat(mem);

    match mode {
      Mode::OAM => {
        Self::try_fire(stat, STAT_INTERRUPT_DURING_OAM, mem);
      }
      Mode::VRAM => {}
      Mode::HBLANK => {
        Self::try_fire(stat, STAT_INTERRUPT_DURING_H_BLANK, mem);
      }
      Mode::VBLANK => {
        Self::try_fire(stat, STAT_INTERRUPT_DURING_V_BLANK, mem);
      }
    }
  }

  fn update_stat_register(&mut self, mode: Mode, mem: &mut MemoryPtr) {
    let mode = match mode {
      Mode::OAM => STAT_OAM,
      Mode::VRAM => STAT_TRANSFERRING_TO_LCD,
      Mode::HBLANK => STAT_H_BLANK,
      Mode::VBLANK => STAT_V_BLANK,
    };

    let current_stat = stat(mem) & STAT_LYC_EQUALS_LCDC;
    util::update_stat_flags(current_stat | mode, mem);
  }

  fn enter_mode(&mut self, mode: Mode, mem: &mut MemoryPtr) {
    self.cycles_in_mode = 0;
    self.mode = mode;
    self.update_stat_register(mode, mem);
    self.try_fire_stat_interrupt(mode, mem);
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

  fn lyc(mem: &mut MemoryPtr) -> u8 {
    mem.read_u8(LYC_SCANLINE)
  }

  fn pal(v: u8, control_reg: u16, mem: &mut MemoryPtr) -> u8 {
    let mut palette = [255, 160, 96, 0];

    // TODO: This is a horrible way, these could all be cached
    let palette_register = mem.read_u8(control_reg);

    for i in 0..4 {
      match palette_register >> (i * 2) & 0x3 {
        0 => palette[i] = 255,
        1 => palette[i] = 192,
        2 => palette[i] = 96,
        3 => palette[i] = 0,
        _ => panic!("impossible condition"),
      }
    }

    palette[(v & 0x3) as usize]
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
      let map_line_offset = ((map_line as u16) >> 3) << 5;

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

        GPU::write_px(
          pixels,
          i as u8,
          self.current_line,
          GPU::pal(val, PAL_BG_REG, mem),
        );

        x += 1;

        if x == 8 {
          x = 0;
          line_offset = (line_offset + 1) & 31;
          tile = self.fetch_tile(map_offset + line_offset, bgtile, mem);
        }
      }
    } else {
      trace!("BG disabled on line: {}", self.current_line);
    }

    if render_sprites {
      for i in 0..40 {
        if Sprite::visible(i, mem) {
          let sprite = Sprite::fetch(i, mem);

          let hits_line_y = sprite.y <= self.current_line && sprite.y + 8 > self.current_line;
          if hits_line_y {
            let tile_y = if sprite.yflip {
              7 - (self.current_line - sprite.y as u8)
            } else {
              self.current_line - sprite.y as u8
            };

            for x in 0..8u8 {
              let tile_x: u8 = if sprite.xflip { 7 - x } else { x };

              let color = self.tile_value(sprite.tile as u16, tile_x as u16, tile_y as u16, mem);
              let low_x = sprite.x + x;
              if low_x >= 0 && low_x < 160 && color != 0 && (sprite.prio || !hit[low_x as usize]) {
                let pval = GPU::pal(
                  color,
                  if sprite.palette {
                    PAL_OBJ1_REG
                  } else {
                    PAL_OBJ0_REG
                  },
                  mem,
                );
                GPU::write_px(pixels, low_x as u8, self.current_line, pval);
              }
            }
          }
        }
      }
    }
  }

  fn update_stat_lyc(&self, mem: &mut MemoryPtr) {
    let lyc = GPU::lyc(mem);

    let current_stat = stat(mem);
    let coincidence_triggered = lyc == self.current_line;

    if coincidence_triggered {
      Self::try_fire(current_stat, STAT_INTERRUPT_LYC_EQUALS_LC, mem);
    }

    let new_stat = if coincidence_triggered {
      current_stat | STAT_LYC_EQUALS_LCDC
    } else {
      current_stat & (!STAT_LYC_EQUALS_LCDC)
    };

    util::update_stat_flags(new_stat, mem);
  }

  fn update_scanline(&mut self, mem: &mut MemoryPtr) {
    mem.write_u8(CURRENT_SCANLINE, self.current_line as u8);
    self.update_stat_lyc(mem);
  }

  fn change_scanline(&mut self, new_scanline: u8, mem: &mut MemoryPtr) {
    self.current_line = new_scanline;
    if new_scanline <= 153 {
      self.update_scanline(mem);
    }
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
          self.enter_mode(Mode::VRAM, mem);
        }
        GpuStepState::None
      }
      Mode::VRAM => {
        if self.cycles_in_mode >= 172 {
          self.render_line(mem, draw);
          self.enter_mode(Mode::HBLANK, mem);
        }
        GpuStepState::None
      }
      Mode::HBLANK => {
        if self.cycles_in_mode >= 204 {
          self.change_scanline(self.current_line + 1, mem);
          if self.current_line == 144 {
            self.enter_mode(Mode::VBLANK, mem);
          } else {
            self.enter_mode(Mode::OAM, mem);
          }
        }
        GpuStepState::HBlank
      }
      Mode::VBLANK => {
        if self.cycles_in_mode >= 114 {
          self.change_scanline(self.current_line + 1, mem);
        }

        if self.current_line > 153 {
          self.change_scanline(0, mem);
          CPU::set_interrupt_happened(mem, VBLANK);
          self.enter_mode(Mode::OAM, mem);
          GpuStepState::VBlank
        } else {
          GpuStepState::None
        }
      }
    }
  }
}
