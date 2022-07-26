use crate::cpu::Registers;
use crate::cpu::{Cpu, STAT, VBLANK};
use crate::memory::{isset8, GameboyState};
use crate::util::{self, stat};
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};

pub const GB_SCREEN_WIDTH: u32 = 160;
pub const GB_SCREEN_HEIGHT: u32 = 144;
pub const BYTES_PER_PIXEL: u32 = 3;
pub const BYTES_PER_ROW: u32 = GB_SCREEN_WIDTH * BYTES_PER_PIXEL;

const TILESET_ONE_ADDR: u16 = 0x8000;

const SCY: u16 = 0xFF42;
const SCX: u16 = 0xFF43;
const LCD_CONTROL: u16 = 0xFF40;
const WX: u16 = 0xFF4B;
const WY: u16 = 0xFF4A;
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
const STAT_H_BLANK: u8 = 0x0;
const STAT_V_BLANK: u8 = 0x1;
const STAT_OAM: u8 = 0x2;
const STAT_TRANSFERRING_TO_LCD: u8 = 0x3;
const STAT_LYC_EQUALS_LCDC: u8 = 0x4;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
enum Mode {
  OAM,
  VRAM,
  HBLANK,
  VBLANK,
}

pub enum PpuStepState {
  None,
  VBlank,
  HBlank,
}

#[derive(Serialize, Deserialize)]
pub struct Ppu {
  cycles_in_mode: u16,
  current_line: u8,
  wx: u8,
  wy: u8,
  mode: Mode,
}

#[derive(Debug)]
struct Sprite {
  pub x: i32,
  pub y: i32,
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

  fn pos(id: u16, mem: &mut GameboyState) -> (i32, i32) {
    let address = OAM + (id * 4);
    let y = mem.core_read(address) as i32;
    let x = mem.core_read(address + 1) as i32;
    (x - 8, y - 16)
  }

  fn visible(id: u16, mem: &mut GameboyState) -> bool {
    let address = OAM + (id * 4);
    let y = mem.core_read(address);
    let x = mem.core_read(address + 1);
    x != 0 || y != 0
  }

  fn fetch(id: u16, mem: &mut GameboyState) -> Self {
    let address = Sprite::address(id);
    let (x, y) = Sprite::pos(id, mem);
    let tile = mem.core_read(address + 2);
    let meta = mem.core_read(address + 3);
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

impl Ppu {
  pub fn new() -> Self {
    info!("PPU initialized");
    Self {
      cycles_in_mode: 0,
      current_line: 0,
      wx: 0,
      wy: 0,
      mode: Mode::OAM,
    }
  }

  fn tile_value(&self, id: u16, x: u16, y: u16, mem: &mut GameboyState) -> u8 {
    const TILE_SIZE: u16 = 16;
    let tile_addr = TILESET_ONE_ADDR + (TILE_SIZE * id);
    let y_addr = tile_addr + (y * 2);
    let mask_x = 1 << (7 - x);
    let low_bit = if mem.core_read(y_addr) & mask_x != 0 {
      1
    } else {
      0
    };
    let high_bit = if mem.core_read(y_addr + 1) & mask_x != 0 {
      2
    } else {
      0
    };
    low_bit + high_bit
  }

  fn lcd_control(&self, mem: &mut GameboyState) -> u8 {
    mem.core_read(LCD_CONTROL)
  }

  fn show_background(lcd: u8) -> bool {
    lcd & 1 != 0
  }

  fn show_sprites(lcd: u8) -> bool {
    lcd & (1 << 1) != 0
  }

  fn window(lcd: u8) -> bool {
    isset8(lcd, 1 << 5)
  }

  fn bgmap(lcd: u8) -> bool {
    isset8(lcd, 1 << 3)
  }

  fn big_sprites(lcd: u8) -> bool {
    isset8(lcd, 1 << 2)
  }

  fn window_bgmap(lcd: u8) -> bool {
    isset8(lcd, 1 << 6)
  }

  fn bgtile(lcd: u8) -> bool {
    lcd & (1 << 4) != 0
  }

  fn try_fire(stat: u8, interrupt: u8, mem: &mut GameboyState, registers: &Registers) {
    if isset8(stat, interrupt) {
      Cpu::set_interrupt_happened(mem, STAT, registers);
    }
  }

  fn try_fire_interrupts(&mut self, mode: Mode, mem: &mut GameboyState, registers: &Registers) {
    let stat = util::stat(mem);

    match mode {
      Mode::OAM => {
        Self::try_fire(stat, STAT_INTERRUPT_DURING_OAM, mem, registers);
      }
      Mode::VRAM => {}
      Mode::HBLANK => {
        Self::try_fire(stat, STAT_INTERRUPT_DURING_H_BLANK, mem, registers);
      }
      Mode::VBLANK => {
        Self::try_fire(stat, STAT_INTERRUPT_DURING_V_BLANK, mem, registers);
        Cpu::set_interrupt_happened(mem, VBLANK, registers);
      }
    }
  }

  fn reset_window(&mut self, mode: Mode, mem: &mut GameboyState) {
    match mode {
      Mode::OAM => {
        self.wx = mem.core_read(WX);
        self.wy = mem.core_read(WY);
      }
      Mode::VRAM => {}
      Mode::HBLANK => {
        self.wx = mem.core_read(WX);
      }
      Mode::VBLANK => {}
    }
  }

  fn enter_mode(&mut self, mode: Mode, mem: &mut GameboyState, registers: &Registers) {
    self.cycles_in_mode = 0;
    self.mode = mode;
    self.reset_window(mode, mem);
    self.update_stat_lyc(mem, registers);
    self.try_fire_interrupts(mode, mem, registers);
  }

  fn write_px(pixels: &mut [u8], x: u8, y: u8, val: u8) {
    let x = x as usize;
    let y = y as usize;
    let canvas_offset = (GB_SCREEN_WIDTH * 3) as usize * y;
    pixels[(x * 3) + canvas_offset] = val;
    pixels[(x * 3) + canvas_offset + 1] = val;
    pixels[(x * 3) + canvas_offset + 2] = val;
  }

  fn fetch_tile(&self, addr: u16, bgtile: bool, mem: &mut GameboyState) -> u16 {
    let tile = mem.core_read(addr) as u16;
    if !bgtile && tile < 128 {
      tile + 256
    } else {
      tile
    }
  }

  fn scx(mem: &mut GameboyState) -> u8 {
    mem.core_read(SCX)
  }

  fn scy(mem: &mut GameboyState) -> u8 {
    mem.core_read(SCY)
  }

  fn lyc(mem: &mut GameboyState) -> u8 {
    mem.core_read(LYC_SCANLINE)
  }

  fn pal(v: u8, control_reg: u16, mem: &mut GameboyState) -> u8 {
    let mut palette = [255, 160, 96, 0];

    // TODO: This is a horrible way, these could all be cached
    let palette_register = mem.core_read(control_reg);

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

  fn render_line(&mut self, mem: &mut GameboyState, pixels: &mut [u8], disable_framebuffer: bool) {
    if disable_framebuffer {
      return;
    }

    if self.current_line >= GB_SCREEN_HEIGHT as u8 {
      return;
    }

    trace!("Rendering full line {}", self.current_line);

    let lcd = self.lcd_control(mem);
    let scy = Self::scy(mem);
    let scx = Self::scx(mem);
    let render_bg = Self::show_background(lcd);
    let render_sprites = Self::show_sprites(lcd);
    let big_sprites = Self::big_sprites(lcd);
    let bgtile = Self::bgtile(lcd);
    let window = Self::window(lcd);
    let background_map_selected = Self::bgmap(lcd);
    let window_map_selected = Self::window_bgmap(lcd);
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

        Self::write_px(
          pixels,
          i as u8,
          self.current_line,
          Self::pal(val, PAL_BG_REG, mem),
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

    if window && self.current_line >= self.wy {
      let map_line = self.current_line - self.wy;
      let map_line_offset = ((map_line as u16) >> 3) << 5;

      let map_offset = if window_map_selected { BGMAP } else { MAP } + map_line_offset;

      trace!(
        "WX: {} WY: {} WINDOW: {} BGM: {} MAP_LINE: {:x} OFFSET: {:x} MO: {:x}",
        self.wx,
        self.wy,
        window,
        background_map_selected,
        map_line,
        map_line_offset,
        map_offset
      );

      let mut line_offset = (self.wx >> 3) as u16;
      let mut tile = self.fetch_tile(map_offset + line_offset, bgtile, mem);

      let mut x = 0;
      let y = ((self.current_line - self.wy) & 7) as u16;

      for i in 0..GB_SCREEN_WIDTH {
        let val = self.tile_value(tile, x as u16, y as u16, mem);

        if val != 0 {
          hit[i as usize] = true;
        }

        Self::write_px(
          pixels,
          i as u8,
          self.current_line,
          Self::pal(val, PAL_BG_REG, mem),
        );

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
        if Sprite::visible(i, mem) {
          let sprite = Sprite::fetch(i, mem);

          let current_line_signed = self.current_line.into();

          let hits_line_y = sprite.y <= current_line_signed
            && sprite.y + if big_sprites { 16 } else { 8 } > current_line_signed;
          if hits_line_y {
            let tile_y = if sprite.yflip {
              7 - (self.current_line - sprite.y as u8)
            } else {
              self.current_line - sprite.y as u8
            };

            for x in 0..8u8 {
              let tile_x: u8 = if sprite.xflip { 7 - x } else { x };

              let color = self.tile_value(sprite.tile as u16, tile_x as u16, tile_y as u16, mem);
              let low_x = sprite.x + x as i32;
              if low_x > 0 && low_x < 160 && color != 0 && (sprite.prio || !hit[low_x as usize]) {
                let pval = Self::pal(
                  color,
                  if sprite.palette {
                    PAL_OBJ1_REG
                  } else {
                    PAL_OBJ0_REG
                  },
                  mem,
                );
                Self::write_px(pixels, low_x as u8, self.current_line, pval);
              }
            }
          }
        }
      }
    }
  }

  fn update_stat_lyc(&self, mem: &mut GameboyState, registers: &Registers) {
    let lyc = Self::lyc(mem);

    let current_stat = stat(mem);
    let coincidence_triggered = lyc == self.current_line;

    debug!("STAT int values {:b}", current_stat);

    if coincidence_triggered {
      Self::try_fire(current_stat, STAT_INTERRUPT_LYC_EQUALS_LC, mem, registers);
    }

    let new_stat = if coincidence_triggered {
      STAT_LYC_EQUALS_LCDC
    } else {
      0
    };

    debug!("STAT to {:b}", new_stat);

    let new_stat = new_stat
      | match self.mode {
        Mode::OAM => STAT_OAM,
        Mode::VRAM => STAT_TRANSFERRING_TO_LCD,
        Mode::HBLANK => STAT_H_BLANK,
        Mode::VBLANK => STAT_V_BLANK,
      };

    util::update_stat_flags(new_stat, mem);
  }

  fn update_scanline(&mut self, mem: &mut GameboyState, registers: &Registers) {
    debug!("SCANLINE: {:x}", self.current_line);
    mem.write_special_register(CURRENT_SCANLINE, self.current_line as u8);
    self.update_stat_lyc(mem, registers);
  }

  fn change_scanline(&mut self, new_scanline: u8, mem: &mut GameboyState, registers: &Registers) {
    self.current_line = new_scanline;
    self.update_scanline(mem, registers);
  }

  pub fn step(
    &mut self,
    cpu: &mut Cpu,
    mem: &mut GameboyState,
    draw: &mut [u8],
    disable_framebuffer: bool,
  ) -> PpuStepState {
    self.cycles_in_mode += cpu.registers.last_clock;
    trace!(
      "Ppu mode {:?} step by {} to {}",
      self.mode,
      cpu.registers.last_clock,
      self.cycles_in_mode
    );

    let current_mode = self.mode;
    match current_mode {
      Mode::OAM => {
        if self.cycles_in_mode >= 80 {
          self.enter_mode(Mode::VRAM, mem, &cpu.registers);
        }
        PpuStepState::None
      }
      Mode::VRAM => {
        if self.cycles_in_mode >= 172 {
          self.render_line(mem, draw, disable_framebuffer);
          self.enter_mode(Mode::HBLANK, mem, &cpu.registers);
        }
        PpuStepState::None
      }
      Mode::HBLANK => {
        if self.cycles_in_mode >= 204 {
          self.change_scanline(self.current_line + 1, mem, &cpu.registers);
          if self.current_line == 145 {
            self.enter_mode(Mode::VBLANK, mem, &cpu.registers);
          } else {
            self.enter_mode(Mode::OAM, mem, &cpu.registers);
          }
        }
        PpuStepState::HBlank
      }
      Mode::VBLANK => {
        if self.cycles_in_mode >= 114 {
          self.change_scanline(self.current_line + 1, mem, &cpu.registers);
          self.cycles_in_mode = 0;
        }

        // It seems that the LYC coincidence interrupt can first one full line before the line zero
        // We hack this in by changing to line zero twice, but the first time staying in VBLANK
        // mode and only moving to OAM the second time.
        if self.current_line == 153 {
          self.change_scanline(0, mem, &cpu.registers);
          self.enter_mode(Mode::OAM, mem, &cpu.registers);
          PpuStepState::VBlank
        } else {
          PpuStepState::None
        }
      }
    }
  }
}
