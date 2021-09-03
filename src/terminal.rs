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

use crate::cpu::{CPU, JOYPAD};
use crate::gpu::{GpuStepState, GB_SCREEN_HEIGHT, GB_SCREEN_WIDTH};
use crate::machine::Machine;

pub fn run(mut gameboy_state: Machine, greyscale: bool) -> io::Result<()> {
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

  enable_raw_mode()?;

  loop {
    let state = gameboy_state.step(&mut pixel_buffer);

    match state {
      GpuStepState::VBlank => {
        let mut state = &mut gameboy_state.memory;
        state.left = false;
        state.right = false;
        state.up = false;
        state.down = false;
        state.a = false;
        state.b = false;
        state.start = false;
        state.select = false;
        if poll(Duration::from_millis(16))? {
          // It's guaranteed that the `read()` won't block when the `poll()`
          // function returns `true`
          match read()? {
            Event::Key(event) => {
              let mut fired = false;
              match event.code {
                KeyCode::Left => {
                  state.left = true;
                  fired = true;
                }
                KeyCode::Right => {
                  state.right = true;
                  fired = true;
                }
                KeyCode::Up => {
                  state.up = true;
                  fired = true;
                }
                KeyCode::Down => {
                  state.down = true;
                  fired = true;
                }
                KeyCode::Char('a') => {
                  state.a = true;
                  fired = true;
                }
                KeyCode::Char('b') => {
                  state.b = true;
                  fired = true;
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

        canvas.clear();

        pub const WHITE: u8 = 255;
        pub const MID: u8 = 128;
        pub const LOW: u8 = 96;

        fn print_greyscale(canvas: &mut Canvas, x: usize, y: usize, shade: u8) {
          if shade <= LOW {
            let shade = WHITE - shade;
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
          if shade <= MID {
            canvas.set(x as u32, y as u32);
          }
        }

        for y in 0..GB_SCREEN_HEIGHT as usize {
          for x in 0..GB_SCREEN_WIDTH as usize {
            let pixel = (y * GB_SCREEN_WIDTH as usize) + x;
            let pixel_offset = pixel * 3;
            let pval = pixel_buffer[pixel_offset];

            if greyscale {
              print_greyscale(&mut canvas, x, y, pval);
            } else {
              print_black_or_white(&mut canvas, x, y, pval);
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
      _ => {}
    }
  }

  disable_raw_mode()?;
  execute!(stdout(), SetSize(cols, rows), LeaveAlternateScreen)?;

  Ok(())
}
