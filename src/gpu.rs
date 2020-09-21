use std::rc::Rc;

use sdl2;
use sdl2::EventPump;
use sdl2::video::{WindowContext};
use sdl2::render::{WindowCanvas, TextureCreator};
use sdl2::rect::Rect;
use sdl2::pixels::Color;
use sdl2::pixels::PixelFormatEnum;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use crate::cpu::CPU;
use crate::memory::MemoryChunk;

use log::trace;

const GB_SCREEN_WIDTH: u32 = 160;
const GB_SCREEN_HEIGHT: u32 = 144;
const BYTES_PER_PIXEL: u32 = 3;
const BYTES_PER_ROW: u32 = GB_SCREEN_WIDTH * BYTES_PER_PIXEL;

const TILESET_ONE_ADDR: u16 = 0x8000;
const TILESET_TWO_ADDR: u16 = 0x8800;

const SCY: u16 = 0xFF42;
const SCX: u16 = 0xFF43;
const LCD_CONTROL: u16 = 0xFF40;
const MAP: u16 = 0x1800;
const BGMAP: u16 = 0x1C00;

const CURRENT_SCANLINE: u16 = 0xFF44;

#[derive(Debug, Copy, Clone)]
enum Mode {
  OAM,
  VRAM,
  HBLANK,
  VBLANK,
}

pub struct GPU {
  events: EventPump,
  canvas: WindowCanvas,
  texture_creator: Rc<TextureCreator<WindowContext>>,
  pixels: Vec<u8>,
  cycles_in_mode: u16,
  current_line: usize,
  mode: Mode,
}

impl GPU {

  pub fn new() -> GPU {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem.window("rustGameboy", GB_SCREEN_WIDTH, GB_SCREEN_HEIGHT)
        .position_centered()
        .build()
        .unwrap();
    let canvas = window.into_canvas().present_vsync().build().unwrap();
    let event_pump = sdl_context.event_pump().unwrap();
    let pixels = vec![0; GB_SCREEN_WIDTH as usize * GB_SCREEN_HEIGHT as usize * 3];
    trace!("GPU initialized");
    GPU {
      events: event_pump,
      texture_creator: Rc::new(canvas.texture_creator()),
      canvas: canvas,
      pixels,
      cycles_in_mode: 0,
      current_line: 0,
      mode: Mode::OAM,
    }
  }

  fn tile_value(&self, tileset: u8, id: u16, x: u16, y: u16, mem: &mut Box<dyn MemoryChunk>) -> u8 {
    const TILE_SIZE: u16 = 16;
    let addr = if tileset == 1 { TILESET_ONE_ADDR } else { TILESET_TWO_ADDR };
    let tile_addr = addr + (TILE_SIZE * id);
    let y_addr = tile_addr + (y * 2);
    let mask_x = 1 << (7 - x);
    mem.read_u8(y_addr) & mask_x + mem.read_u8(y_addr + 1) & mask_x
  }

  fn lcd_control(&mut self, mem: &mut Box<dyn MemoryChunk>) -> u8 {
    mem.read_u8(LCD_CONTROL)
  }

  fn bgmap(&mut self, mem: &mut Box<dyn MemoryChunk>) -> bool {
    self.lcd_control(mem) & 0x08 != 0
  }

  fn bgtile(&mut self, mem: &mut Box<dyn MemoryChunk>) -> bool {
    self.lcd_control(mem) & 0x10 != 0
  }

  fn enter_mode(&mut self, mode: Mode) {
    self.cycles_in_mode = 0;
    self.mode = mode;
  }

  fn render_line(&mut self, mem: &mut Box<dyn MemoryChunk>) {
    trace!("Rendering full line {}", self.current_line);
    let scx = mem.read_u8(SCX);
    let scy = mem.read_u8(SCY);
    let map_offset = if self.bgmap(mem) { BGMAP } else { MAP };
    let map_offset = map_offset + ((scy as u16 + self.current_line as u16) >> 3);
    let mut line_offset = scx as u16 >> 3;
    let y = (self.current_line as u16 + scy as u16) & 7;
    let mut x = scx & 7;
    let mut tile = mem.read_u8(map_offset + line_offset) as u16;
    let canvas_offset = (GB_SCREEN_WIDTH * 3) as usize * self.current_line;
    for i in 0..160 {
      self.pixels[(i * 3) + canvas_offset] = 255;
      self.pixels[(i * 3) + canvas_offset + 1] = 255;
      self.pixels[(i * 3) + canvas_offset + 2] = 255;
      let val = self.tile_value(1, tile as u16, x as u16, y as u16, mem);
      if val != 0 {
        self.pixels[(i * 3) + canvas_offset] = 0;
      }
      x += 1;
      if x == 8 {
        x = 0;
        line_offset = (line_offset + 1) & 31;
        tile = mem.read_u8(map_offset + line_offset) as u16;
        if self.bgtile(mem) && tile < 128 {
          tile += 256;
        }
      }
    }
  }

  fn update_scanline(&mut self, mem: &mut Box<dyn MemoryChunk>) {
    mem.write_u8(CURRENT_SCANLINE, self.current_line as u8);
  }

  pub fn step(&mut self, cpu: &mut CPU, mem: &mut Box<dyn MemoryChunk>) {
    self.cycles_in_mode += cpu.registers.last_clock;
    trace!("GPU mode {:?} step by {} to {}", self.mode, cpu.registers.last_clock, self.cycles_in_mode);
    let current_mode = self.mode;
    match current_mode {
      Mode::OAM => {
        if self.cycles_in_mode >= 80 {
          self.enter_mode(Mode::VRAM);
        }
      },
      Mode::VRAM => {
        if self.cycles_in_mode >= 172 {
          self.render_line(mem);
          self.redraw();
          self.enter_mode(Mode::HBLANK);
        }
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
      },
      Mode::VBLANK => {
        if self.cycles_in_mode % 204 == 0 {
          self.current_line += 1;
          self.update_scanline(mem);
          trace!("TODO: Increment LY");
        }

        if self.cycles_in_mode >= 4560 {
          self.current_line = 0;
          //TODO: flip interrupt
          self.enter_mode(Mode::OAM);
          self.redraw();
        }
      },
    };
  }

  fn redraw(&mut self) {
    trace!("Redrawing screen");

    for event in self.events.poll_iter() {
      match event {
        Event::Quit {..} | Event::KeyDown {keycode: Some(Keycode::Escape), ..} => {
          unimplemented!();
        },
        _ => {}
      }
    }

    let screen_dims = Rect::new(0, 0, GB_SCREEN_WIDTH, GB_SCREEN_HEIGHT);

    // Render the pixels to a texture
    // TODO: Ideally we would cache a single texture
    // and update pixels to it in a streaming format.
    // Unfortunately, Rust SDL2 borrow checking is a POS
    // and this is not easy.
    let texture_creator = &self.texture_creator;

    let mut texture = texture_creator.create_texture_static(
      PixelFormatEnum::RGB24,
      GB_SCREEN_WIDTH,
      GB_SCREEN_HEIGHT
    ).unwrap();

    texture.update(
      screen_dims,
      &self.pixels,
      BYTES_PER_ROW as usize
    ).unwrap();

    // Now render the texture to the canvas
    self.canvas.set_draw_color(Color::RGB(255, 255, 255));
    self.canvas.clear();
    self.canvas.copy(&texture, screen_dims, screen_dims).unwrap();
    self.canvas.present();
  }

}
