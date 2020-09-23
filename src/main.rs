mod memory;
mod cpu;
mod gpu;
mod machine;
mod instruction;

use std::io;

use sdl2;
use sdl2::EventPump;
use sdl2::render::{WindowCanvas, Texture};
use sdl2::rect::Rect;
use sdl2::pixels::PixelFormatEnum;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use memory::{RomChunk, RamChunk, MemoryMapEntry, MemoryMap};
use log::{info, trace};
use gpu::{GPU, GpuStepState, GB_SCREEN_WIDTH, GB_SCREEN_HEIGHT, BYTES_PER_ROW};

fn events(events: &mut EventPump) {
  for event in events.poll_iter() {
    match event {
      Event::Quit {..} | Event::KeyDown {keycode: Some(Keycode::Escape), ..} => {
        unimplemented!();
      },
      _ => {}
    }
  }
}

fn redraw(canvas: &mut WindowCanvas, texture: &mut Texture, pixels: &[u8]) {
  trace!("Redrawing screen");

  let screen_dims = Rect::new(0, 0, GB_SCREEN_WIDTH, GB_SCREEN_HEIGHT);
  let out_dims = Rect::new(0, 0, GB_SCREEN_WIDTH * 4, GB_SCREEN_HEIGHT * 4);

  // Now render the texture to the canvas
  texture.update(screen_dims, pixels, BYTES_PER_ROW as usize).unwrap();
  canvas.copy(&texture, screen_dims, out_dims).unwrap();
  canvas.present();
}

fn main() -> io::Result<()> {
  env_logger::init();
  info!("preparing initial state");

  let gpu = GPU::new();

  let gb_test = RomChunk::from_file("/home/blake/gb/test2.gb")?;
  let boot_rom = RomChunk::from_file("/home/blake/gb/bios.gb")?;
  let high_ram = RamChunk::new(0x100);

  // Ram for the sprite buffers etc
  let sprite_ram = RamChunk::new(0x100);
  let vram = RamChunk::new(0x2000);

  // Work ram units
  let work_ram_one = RamChunk::new(0x1000);
  let work_ram_two = RamChunk::new(0x1000);

  // Cartridge might have RAM
  let cart_ram = RamChunk::new(0x2000);

  let root_map = MemoryMap::from(
    vec![
//      MemoryMapEntry::from(Box::new(boot_rom), (0, 0xFF)),
      MemoryMapEntry::from(Box::new(gb_test), (0, 0x7FFF)),
      MemoryMapEntry::from(Box::new(vram), (0x8000, 0x9FFF)),
      MemoryMapEntry::from(Box::new(cart_ram), (0xA000, 0xBFFF)),
      MemoryMapEntry::from(Box::new(sprite_ram), (0xFE00, 0xFEFF)),
      MemoryMapEntry::from(Box::new(work_ram_one), (0xC000, 0xCFFF)),
      MemoryMapEntry::from(Box::new(work_ram_two), (0xD000, 0xDFFF)),
      MemoryMapEntry::from(Box::new(high_ram), (0xFF00, 0xFFFF)),
    ]
  );

  let mut gameboy_state = machine::Machine {
    cpu: cpu::CPU::new(),
    gpu: gpu,
    memory: Box::new(root_map)
  };

  gameboy_state.cpu.registers.set_pc(0x100);

  info!("prearing screen");

  let sdl_context = sdl2::init().unwrap();
  let video_subsystem = sdl_context.video().unwrap();
  let window = video_subsystem.window("rustGameboy", GB_SCREEN_WIDTH * 4, GB_SCREEN_HEIGHT * 4)
      .position_centered()
      .build()
      .unwrap();
  let mut canvas = window.into_canvas().build().unwrap();
  let mut event_pump = sdl_context.event_pump().unwrap();
  let texture_creator = canvas.texture_creator();

  let mut texture = texture_creator.create_texture_static(
    PixelFormatEnum::RGB24,
    GB_SCREEN_WIDTH,
    GB_SCREEN_HEIGHT
  ).unwrap();

  info!("starting core loop");

  let mut pixel_buffer = vec![0; GB_SCREEN_WIDTH as usize * GB_SCREEN_HEIGHT as usize * 3];

  use std::time::{Duration, Instant};
  let now = Instant::now();
  let mut steps = 0;
  let mut redraws = 0;

  loop {
    let state = gameboy_state.step(&mut pixel_buffer);
    match state {
      GpuStepState::VBlank => {
        events(&mut event_pump);
        redraw(&mut canvas, &mut texture, &pixel_buffer);
        redraws += 1;
      },
      _ => {}
    }
    steps += 1;
    if steps % 100000 == 0 {
      let time_running = now.elapsed().as_secs_f64();
      println!(
        "Average step rate of {}/s with a redraw rate of {}/s",
        steps as f64 / time_running,
        redraws as f64 / time_running
      );
    }
  }
}
