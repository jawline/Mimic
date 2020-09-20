use sdl2;
use sdl2::video::Window;
use sdl2::render::WindowCanvas;
use sdl2::pixels::Color;
use sdl2::pixels::PixelFormatEnum;
use sdl2::surface::Surface;

const GB_SCREEN_WIDTH: u32 = 160;
const GB_SCREEN_HEIGHT: u32 = 144;

pub struct GPU<'a> {
  canvas: WindowCanvas,
  surface: Surface<'a>,
}

impl<'a> GPU<'a> {

  pub fn new() -> GPU<'a> {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem.window("rustGameboy", GB_SCREEN_WIDTH, GB_SCREEN_HEIGHT)
        .position_centered()
        .build()
        .unwrap();
    let canvas = window.into_canvas().present_vsync().build().unwrap();
    let surface = Surface::new(GB_SCREEN_WIDTH, GB_SCREEN_HEIGHT, PixelFormatEnum::RGB24).unwrap();
    GPU {
      canvas: canvas,
      surface: surface,
    }
  }

  pub fn step(&mut self) {
    self.redraw();
  }

  fn redraw(&mut self) {
    self.canvas.set_draw_color(Color::RGB(0, 0, 0));
    self.canvas.clear();
    self.canvas.present();
  }

}
