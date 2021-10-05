use std::time::{Duration, SystemTime};

pub const FRAME_TIME: Duration = Duration::from_nanos(16666666);

pub struct FrameTimer {
  /// This can be increased by a factor of FRAME_TIME to introduce frame skip
  frame_interval: Duration,
  /// Time of last frame draw
  last_frame: SystemTime,
}

impl FrameTimer {

    pub fn new(skip_rate: u32) -> Self {
        FrameTimer {
            frame_interval: FRAME_TIME * skip_rate,
            last_frame: SystemTime::now(),
        }
    }

    pub fn should_redraw(&mut self) -> bool {
      let elapsed = self.last_frame.elapsed().expect("We expect SystemTime.elapsed to always return a result");
      if elapsed > self.frame_interval {
          self.last_frame = SystemTime::now();
          true
      } else {
          false
      }
    }
}

