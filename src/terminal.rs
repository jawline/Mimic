use crossterm::{
  cursor::{Hide, MoveTo},
  event::{poll, read, Event, KeyCode},
  execute, queue,
  style::Print,
  terminal::{
    disable_raw_mode, enable_raw_mode, size, Clear, ClearType, EnterAlternateScreen,
    LeaveAlternateScreen, SetSize,
  },
};
use drawille::{Canvas, PixelColor};
use std::io::{self, stdout, Write};
use std::time::Duration;
use std::time::SystemTime;

use crate::cpu::{CPU, JOYPAD};
use crate::gpu::{GpuStepState, GB_SCREEN_HEIGHT, GB_SCREEN_WIDTH};
use crate::machine::Machine;

pub fn run(mut gameboy_state: Machine, greyscale: bool, invert: bool, threshold: bool) -> io::Result<()> {
  let mut pixel_buffer = vec![0; GB_SCREEN_WIDTH as usize * GB_SCREEN_HEIGHT as usize * 3];
  let mut canvas = Canvas::new(GB_SCREEN_WIDTH, GB_SCREEN_HEIGHT);

  let (cols, rows) = size()?;

  let frame_width = canvas.frame().lines().next().unwrap().len();
  let frame_height = canvas.frame().lines().count();

  execute!(
    stdout(),
    SetSize(frame_width as u16, frame_height as u16),
    EnterAlternateScreen,
    Hide,
  )?;

  // Simulate long keypresses because the terminal does not send
  // keyup and keydown events
  let mut last_a = None;
  let mut last_b = None;
  let mut last_left = None;
  let mut last_right = None;
  let mut last_up = None;
  let mut last_down = None;

  let mut frame = 0;

  fn is_key(key: Option<SystemTime>) -> bool {
    match key {
      Some(time) => SystemTime::now().duration_since(time).unwrap().as_millis() < 400,
      None => false,
    }
  }

  enable_raw_mode()?;

  loop {
    let state = gameboy_state.step(&mut pixel_buffer);

    match state {
      GpuStepState::VBlank => {
        let mut state = &mut gameboy_state.memory;

        state.start = false;
        state.select = false;

        if poll(Duration::from_millis(20))? {
          // It's guaranteed that the `read()` won't block when the `poll()`
          // function returns `true`
          match read()? {
            Event::Key(event) => {
              let mut fired = false;
              match event.code {
                KeyCode::Left => {
                  fired = fired | !is_key(last_left);
                  last_left = Some(SystemTime::now());
                }
                KeyCode::Right => {
                  fired = fired | !is_key(last_right);
                  last_right = Some(SystemTime::now());
                }
                KeyCode::Up => {
                  fired = fired | !is_key(last_up);
                  last_up = Some(SystemTime::now());
                }
                KeyCode::Down => {
                  fired = fired | !is_key(last_down);
                  last_down = Some(SystemTime::now());
                }
                KeyCode::Char('a') => {
                  fired = fired | !is_key(last_a);
                  last_a = Some(SystemTime::now());
                }
                KeyCode::Char('b') => {
                  fired = fired | !is_key(last_b);
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
                KeyCode::Char('q') => {
                  break;
                }
                _ => {}
              }
              if fired {
                CPU::set_interrupt_happened(state, JOYPAD);
              }
            }
            _ => {}
          }
        }

        state.a = is_key(last_a);
        state.b = is_key(last_b);
        state.left = is_key(last_left);
        state.right = is_key(last_right);
        state.up = is_key(last_up);
        state.down = is_key(last_down);

        // Redraw only every other frame to help with flashing
        if frame % 3 == 0 {
          canvas.clear();

          pub const WHITE: u8 = 255;
          pub const MID: u8 = 128;
          pub const LOW: u8 = 96;

          fn print_greyscale(canvas: &mut Canvas, threshold: bool, x: usize, y: usize, shade: u8) {
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

          queue!(stdout(), MoveTo(0, 0), Clear(ClearType::All))?;

          let mut idx = 0;

          for line in frame.lines() {
            queue!(stdout(), MoveTo(0, idx), Print(line))?;
            idx += 1;
          }

          stdout().flush()?;
        }

        frame += 1;
      }
      _ => {}
    }
  }

  disable_raw_mode()?;
  execute!(stdout(), SetSize(cols, rows), LeaveAlternateScreen)?;

  Ok(())
}
