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
use std::error::Error;
use std::io::stdout;
use std::time::Duration;
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
) -> Result<(), Box<dyn Error>> {
  // TODO: select this on/off
  let (_device, _stream, sample_rate, sound_tx) = crate::sound::open_device()?;

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
    Clear(ClearType::All)
  )?;

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

  enable_raw_mode()?;

  // Screen redraw is controlled by the frame timer
  // But we don't want the game to get too ahead of
  // the screen so we stop stepping the emulator after
  // the vblank period but before the next frame redraw
  let mut emulator_timer = FrameTimer::new(1);
  let mut emulator_running_fast = false;
  let mut frame_timer = FrameTimer::new(frameskip_rate);

  loop {
    if emulator_running_fast {
      if emulator_timer.should_redraw() {
        emulator_running_fast = false;
      }
    }

    if !emulator_running_fast {
      let state = gameboy_state.step(&mut pixel_buffer, sample_rate, &sound_tx);

      match state {
        PpuStepState::VBlank => {
          let mut save = false;
          let mut state = &mut gameboy_state.state.memory;

          state.start = false;
          state.select = false;

          if poll(Duration::from_millis(0))? {
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
              _ => {}
            }
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
        _ => {}
      }
    }

    // Redraw only every other frame to help with flashing
    if frame_timer.should_redraw() {
      canvas.clear();

      pub const WHITE: u8 = 255;
      pub const MID: u8 = 128;

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

      queue!(stdout(), MoveTo(0, 0))?;

      let mut idx = 0;

      for line in frame.lines() {
        queue!(stdout(), MoveTo(0, idx), Print(line))?;
        idx += 1;
      }

      execute!(stdout())?;
    }
  }

  disable_raw_mode()?;
  execute!(stdout(), SetSize(cols, rows), LeaveAlternateScreen)?;

  Ok(())
}
