
  use crossterm::{
    cursor::{ Hide, MoveTo},
    execute,
    queue,
    event::{poll, read, Event, KeyCode},
    style::Print,
    terminal::{size, Clear, ClearType, SetSize, enable_raw_mode, disable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    
  };
  use std::io::{self, stdout, Write};
  use std::time::Duration;
  use drawille::Canvas;
  
use crate::cpu::{CPU, JOYPAD};
use crate::gpu::{GpuStepState, GB_SCREEN_HEIGHT, GB_SCREEN_WIDTH};
use crate::machine::Machine;

pub fn run(mut gameboy_state: Machine) -> io::Result<()> {

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

        if poll(Duration::from_millis(10))? {
            // It's guaranteed that the `read()` won't block when the `poll()`
            // function returns `true`
            match read()? {
                Event::Key(event) => {
                    let mut fired = false;
                    match event.code {
                        KeyCode::Left => { state.left = true; fired = true; },
                        KeyCode::Right => { state.right = true; fired = true; },
                        KeyCode::Up => { state.up = true; fired = true; },
                        KeyCode::Down => { state.down = true; fired = true; },
                        KeyCode::Char('a') => { state.a = true; fired = true;},
                        KeyCode::Char('b') => { state.b = true; fired = true; },
                        KeyCode::Char('n') => { state.start = true; fired = true;},
                        KeyCode::Char('m') => { state.select = true; fired = true; },
                        KeyCode::Char('q') => { break; }
                        _ => {}
                    }
                    if fired {
                        CPU::set_interrupt_happened(state, JOYPAD);
                    }
                },
                _ => {}
            }
        }

        canvas.clear();

        for y in 0..GB_SCREEN_HEIGHT as usize {
          for x in 0..GB_SCREEN_WIDTH as usize {
            let pixel = (y * GB_SCREEN_WIDTH as usize) + x;
            let pixel_offset = pixel * 3;
            let pval = pixel_buffer[pixel_offset];
            if pval <= 128 {
              canvas.set(x as u32, y as u32);
            }
          }
        }

        let frame = canvas.frame();

        queue!(
          stdout(),
          MoveTo(0, 0),
          Clear(ClearType::All))?;

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

