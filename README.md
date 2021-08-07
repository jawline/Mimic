## RustGameboy

A Gameboy emulator written in Rust. Can launch and play simple games.

![Screenshot](/screenshots/screenshot1.png?raw=true "Screenshot 1")
![Screenshot](/screenshots/screenshot2.png?raw=true "Screenshot 2")
![Screenshot](/screenshots/screenshot3.png?raw=true "Screenshot 3")
![Screenshot](/screenshots/screenshot4.png?raw=true "Screenshot 4")
![Screenshot](/screenshots/screenshot5.png?raw=true "Screenshot 5")
![Screenshot](/screenshots/screenshot6.png?raw=true "Screenshot 6")

### Usage

cargo run --release "PATH_TO_BIOS" "PATH_TO_ROM"

### Working

- System memory map
- Core instruction set
- Pixel-processing unit
- Background rendering
- Sprite rendering
- Interrupts
- Input

### TODO

- Sound
- Memory Banking
- Timing
- The emulator will not run in Debug mode because Rust errors on unsigned integer arithmetic over/underflow. This can be alleviated using the Wrapping<t> struct but requires explicit wrapping of every literal value. wrapping_add and wrapping_sub could be used on every instruction, but neither solution leads to readable code and I would prefer a solution that leaves the base logic understandable. 
