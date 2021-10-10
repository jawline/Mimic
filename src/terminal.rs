use console_engine::events::Event;
use console_engine::Color;
use console_engine::ConsoleEngine;
use console_engine::KeyCode;

use console_engine::pixel;
use drawille::{Canvas, PixelColor};
use std::io::{self};
use std::time::SystemTime;

use crate::cpu::{Cpu, JOYPAD};
use crate::frame_timer::FrameTimer;
use crate::machine::Machine;
use crate::ppu::{PpuStepState, GB_SCREEN_HEIGHT, GB_SCREEN_WIDTH};

const KEY_INTERVAL: u128 = 200;

pub fn run(
  mut gameboy_state: Machine,
  save_path: &str,
  frameskip_rate: u32,
  greyscale: bool,
  invert: bool,
  threshold: bool,
) -> io::Result<()> {
  let mut pixel_buffer = vec![0; GB_SCREEN_WIDTH as usize * GB_SCREEN_HEIGHT as usize * 3];
  let mut canvas = Canvas::new(GB_SCREEN_WIDTH, GB_SCREEN_HEIGHT);

  let frame_width = canvas.frame().lines().next().unwrap().len();
  let frame_height = canvas.frame().lines().count();
  let mut engine = ConsoleEngine::init(frame_width as u32, frame_height as u32, 60)?;

  // Simulate long keypresses because the terminal does not send
  // keyup and keydown events
  let mut last_a = None;
  let mut last_b = None;
  let mut last_left = None;
  let mut last_right = None;
  let mut last_up = None;
  let mut last_down = None;
  let mut redrawing = true;

  fn is_key(key: Option<SystemTime>) -> bool {
    match key {
      Some(time) => SystemTime::now().duration_since(time).unwrap().as_millis() < KEY_INTERVAL,
      None => false,
    }
  }

  // Screen redraw is controlled by the frame timer
  // But we don't want the game to get too ahead of
  // the screen so we stop stepping the emulator after
  // the vblank period but before the next frame redraw
  let mut emulator_timer = FrameTimer::new(1);
  let mut emulator_running_fast = false;
  let mut frame_timer = FrameTimer::new(frameskip_rate);

  loop {
    if emulator_running_fast && emulator_timer.should_redraw() {
      emulator_running_fast = false;
    }

    if !emulator_running_fast {
      let state = gameboy_state.step(&mut pixel_buffer);

      if let PpuStepState::VBlank = state {
        let mut save = false;
        let mut state = &mut gameboy_state.state.memory;

        state.start = false;
        state.select = false;

        match engine.poll() {
          Event::Key(event) => {
            let mut fired = false;
            match event.code {
              KeyCode::Left => {
                fired |= !is_key(last_left);
                last_left = Some(SystemTime::now());
              }
              KeyCode::Right => {
                fired |= !is_key(last_right);
                last_right = Some(SystemTime::now());
              }
              KeyCode::Up => {
                fired |= !is_key(last_up);
                last_up = Some(SystemTime::now());
              }
              KeyCode::Down => {
                fired |= !is_key(last_down);
                last_down = Some(SystemTime::now());
              }
              KeyCode::Char('a') => {
                fired |= !is_key(last_a);
                last_a = Some(SystemTime::now());
              }
              KeyCode::Char('b') => {
                fired |= !is_key(last_b);
                last_b = Some(SystemTime::now());
              }
              KeyCode::Char('n') => {
                state.start = true;
                fired = true;
              }
              KeyCode::Char('m') => {
                state.select = true;
                fired = true;
              }
              KeyCode::Char('p') => {
                redrawing = !redrawing;
              }
              KeyCode::Char('s') => {
                save = true;
              }
              KeyCode::Char('q') => {
                break;
              }
              _ => {}
            }
            if fired {
              Cpu::set_interrupt_happened(state, JOYPAD);
            }
          }
          Event::Frame => {
            // Redraw only every other frame to help with flashing
            if frame_timer.should_redraw() {
              canvas.clear();
              engine.clear_screen();

              pub const WHITE: u8 = 255;
              pub const MID: u8 = 128;

              fn print_greyscale(
                canvas: &mut Canvas,
                threshold: bool,
                x: usize,
                y: usize,
                shade: u8,
              ) {
                if (!threshold & (shade > 0)) | (threshold & (shade >= MID)) {
                  canvas.set_colored(
                    x as u32,
                    y as u32,
                    PixelColor::TrueColor {
                      r: shade,
                      g: shade,
                      b: shade,
                    },
                  );
                }
              }

              fn print_black_or_white(canvas: &mut Canvas, x: usize, y: usize, shade: u8) {
                if shade >= MID {
                  canvas.set(x as u32, y as u32);
                }
              }

              for y in 0..GB_SCREEN_HEIGHT as usize {
                for x in 0..GB_SCREEN_WIDTH as usize {
                  let pixel = (y * GB_SCREEN_WIDTH as usize) + x;
                  let pixel_offset = pixel * 3;
                  let pval = pixel_buffer[pixel_offset];

                  let shade = if invert { WHITE - pval } else { pval };
                  if greyscale {
                    print_greyscale(&mut canvas, threshold, x, y, shade);
                  } else {
                    print_black_or_white(&mut canvas, x, y, shade);
                  }
                }
              }

              let frame = canvas.frame();

              for (idx, line) in frame.lines().enumerate() {
                let mut x = 0;
                let mut color = Color::Reset;
                // dirty hack to remove the ANSI characters and convert them to pixels
                for chr in line
                  .replace("\u{1b}[38;2;255;255;255m", "1")
                  .replace("\u{1b}[38;2;192;192;192m", "2")
                  .replace("\u{1b}[0m", "3")
                  .chars()
                {
                  if chr == '1' || chr == '2' || chr == '3' {
                    color = if chr == '1' {
                      Color::White
                    } else if chr == '2' {
                      Color::Rgb {
                        r: 192,
                        g: 192,
                        b: 192,
                      }
                    } else {
                      Color::Reset
                    };
                  } else {
                    engine.set_pxl(x, idx as i32, pixel::pxl_fg(chr, color));
                    x += 1;
                  }
                }
              }

              engine.draw();
            }
          }
          _ => {}
        }

        state.a = is_key(last_a);
        state.b = is_key(last_b);
        state.left = is_key(last_left);
        state.right = is_key(last_right);
        state.up = is_key(last_up);
        state.down = is_key(last_down);

        if save {
          gameboy_state.save_state(save_path).unwrap();
        }

        // Stop processing until the frame timer tells us we're ok again
        emulator_running_fast = true;
      }
    }
  }
  Ok(())
}
