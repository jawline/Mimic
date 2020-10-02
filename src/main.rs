mod cpu;
mod gpu;
mod instruction;
mod machine;
mod memory;

use std::env;
use std::io;
use std::time::Instant;

use sdl2;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{Texture, WindowCanvas};
use sdl2::EventPump;

use cpu::{CPU, JOYPAD};
use gpu::{GpuStepState, BYTES_PER_ROW, GB_SCREEN_HEIGHT, GB_SCREEN_WIDTH, GPU};
use log::{info, trace};
use memory::{GameboyState, RomChunk};

fn events(state: &mut GameboyState, events: &mut EventPump) {
  let mut fired = false;
  for event in events.poll_iter() {
    match event {
      Event::Quit { .. }
      | Event::KeyDown {
        keycode: Some(Keycode::Escape),
        ..
      } => {
        unimplemented!();
      }
      Event::KeyDown {
        keycode: Some(Keycode::A),
        ..
      } => {
        fired = true;
        state.a = true;
        info!("A");
      }
      Event::KeyUp {
        keycode: Some(Keycode::A),
        ..
      } => {
        state.a = false;
      }
      Event::KeyDown {
        keycode: Some(Keycode::B),
        ..
      } => {
        fired = true;
        info!("B");
        state.b = true;
      }
      Event::KeyUp {
        keycode: Some(Keycode::B),
        ..
      } => {
        state.b = false;
      }
      Event::KeyDown {
        keycode: Some(Keycode::N),
        ..
      } => {
        fired = true;
        info!("START");
        state.start = true;
      }
      Event::KeyUp {
        keycode: Some(Keycode::N),
        ..
      } => {
        state.start = false;
      }
      Event::KeyDown {
        keycode: Some(Keycode::M),
        ..
      } => {
        fired = true;
        info!("SELECT");
        state.select = true;
      }
      Event::KeyUp {
        keycode: Some(Keycode::M),
        ..
      } => {
        state.select = false;
      }
      Event::KeyDown {
        keycode: Some(Keycode::Left),
        ..
      } => {
        fired = true;
        info!("LEFT");
        state.left = true;
      }
      Event::KeyUp {
        keycode: Some(Keycode::Left),
        ..
      } => {
        state.left = false;
      }
      Event::KeyDown {
        keycode: Some(Keycode::Right),
        ..
      } => {
        fired = true;
        info!("RIGHT");
        state.right = true;
      }
      Event::KeyUp {
        keycode: Some(Keycode::Right),
        ..
      } => {
        state.right = false;
      }
      Event::KeyDown {
        keycode: Some(Keycode::Up),
        ..
      } => {
        fired = true;
        info!("UP");
        state.up = true;
      }
      Event::KeyUp {
        keycode: Some(Keycode::Up),
        ..
      } => {
        state.up = false;
      }
      Event::KeyDown {
        keycode: Some(Keycode::Down),
        ..
      } => {
        fired = true;
        info!("DOWN");
        state.down = true;
      }
      Event::KeyUp {
        keycode: Some(Keycode::Down),
        ..
      } => {
        state.down = false;
      }
      _ => {}
    }
  }

  if fired {
    println!("fire of interrupt");
    CPU::set_interrupt_happened(state, JOYPAD);
  }
}

fn redraw(canvas: &mut WindowCanvas, texture: &mut Texture, pixels: &[u8]) {
  trace!("Redrawing screen");

  let screen_dims = Rect::new(0, 0, GB_SCREEN_WIDTH, GB_SCREEN_HEIGHT);
  let out_dims = Rect::new(0, 0, GB_SCREEN_WIDTH * 4, GB_SCREEN_HEIGHT * 4);

  // Now render the texture to the canvas
  texture
    .update(screen_dims, pixels, BYTES_PER_ROW as usize)
    .unwrap();
  canvas.copy(&texture, screen_dims, out_dims).unwrap();
  canvas.present();
}

fn main() -> io::Result<()> {
  env_logger::init();

  let mut args = env::args();

  info!("ARGS: {:?}", args);

  args.next();
  let bios_file = args.next().unwrap();
  let rom_file = args.next().unwrap();

  info!("loading BIOS: {} TEST: {}", bios_file, rom_file);

  info!("preparing initial state");

  let boot_rom = RomChunk::from_file(&bios_file)?;
  let gb_test = RomChunk::from_file(&rom_file)?;

  let root_map = GameboyState::new(boot_rom, gb_test);

  let mut gameboy_state = machine::Machine {
    cpu: CPU::new(),
    gpu: GPU::new(),
    memory: root_map,
  };

  // Skip boot
  use crate::memory::MemoryChunk;
  gameboy_state.cpu.registers.set_pc(0x100);
  gameboy_state.memory.write_u8(0xFF50, 1);

  info!("preparing screen");

  let sdl_context = sdl2::init().unwrap();
  let video_subsystem = sdl_context.video().unwrap();
  let window = video_subsystem
    .window("rustGameboy", GB_SCREEN_WIDTH * 4, GB_SCREEN_HEIGHT * 4)
    .position_centered()
    .build()
    .unwrap();
  let mut canvas = window.into_canvas().present_vsync().build().unwrap();
  let mut event_pump = sdl_context.event_pump().unwrap();
  let texture_creator = canvas.texture_creator();

  let mut texture = texture_creator
    .create_texture_static(PixelFormatEnum::RGB24, GB_SCREEN_WIDTH, GB_SCREEN_HEIGHT)
    .unwrap();

  info!("starting core loop");

  let mut pixel_buffer = vec![0; GB_SCREEN_WIDTH as usize * GB_SCREEN_HEIGHT as usize * 3];

  let now = Instant::now();
  let mut steps = 0;
  let mut redraws = 0;

  loop {
    let state = gameboy_state.step(&mut pixel_buffer);
    match state {
      GpuStepState::VBlank => {
        events(&mut gameboy_state.memory, &mut event_pump);
        redraw(&mut canvas, &mut texture, &pixel_buffer);
        redraws += 1;
      }
      _ => {}
    }
    steps += 1;
    if steps % 100000 == 0 {
      let time_running = now.elapsed().as_secs_f64();
      info!(
        "Average step rate of {}/s with a redraw rate of {}/s",
        steps as f64 / time_running,
        redraws as f64 / time_running
      );
    }
  }
}
