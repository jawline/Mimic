## Mimic

An open source Gameboy emulator written in Rust that can use a command line interface as a screen and input device. The project is an attempt to make an approachable emulator for the Gameboy that can be used to explain the concepts required in emulating a system without overwhelming the reader. The core logic is all in safe Rust, there is no JIT recompiler, and screen / IO logic is kept separate from the emulation core to reduce complexity. As such, it does not perform ideally, but the Gameboy is an old system so ideal performance is not necessary to run games at full speed.

### Usage

cargo run --release -- --rom PATH --cli-mode

<p float="left">
  <img src="/screenshots/zelda1.png" width="49%" />
  <img src="/screenshots/pokemon.png" width="49%" />
  <img src="/screenshots/screenshot4.png" width="49%" />
  <img src="/screenshots/screenshot5.png" width="49%" />
</p>

### Overview

We split our design into a set of components that roughly map to the physical components on the original hardware. These components are:
- CPU: This structure models the fetch, decode, and execute cycle of the CPU and owns the machine registers. Hardware interrupts are also modelled in this structure.
- Registers: A structure that stores the CPU registers and some hidden registers for emulation. This does not store the special memory registers.
- Instructions: An implementation of each CPU opcode plus a table mapping opcodes to their implementation.
- Clock: A structure that models the Gameboy clock.
- PPU (Pixel-processing unit): A structure that models the Gameboy PPU, a component which takes sprite maps and data from memory and draws them to the screen. All video output on the Gameboy travels through the PPU.
- Memory: An implementation of the hardware memory bus and cartridge emulation. Special memory registers are also stored here.

In addition to these components we have the Machine structure which brings all of these components together under a single structure with a central step instruction that moves the emulation forward by precisely one instruction (meaning a single instruction is executed and then all other components are updated so that they are up to date with the new state). The remaining files in the project, main.rs, terminal.rs, and sdl.rs provide the launch sequence and screen / terminal backends for interacting with the emulated machine.

### CPU

The Gameboy uses a modified CISC (complex instruction set computer) Z80 Zilog processor at a frequency of 4.19Mhz. It features an 8-bit instruction set with a 16-bit address bus and some limited support for 16 bit arithmetic. The CPU has a complex instruction set with variable length opcodes. The first byte of every opcode indicates which of the instructions it is. As there are more than 256 instructions, there is also an extended opcode which swaps the CPU to a second instruction set for an instruction starting at the next byte.

The CPU is represented in Mimic through two jump tables that are constructed at startup, one for the main opcode set and one for the extended set. Each entry in the jump table contains a pointer to an execute function which takes the current registers and memory and implements the opcode. A secret register is used to keep track of whether the previous instruction executed was the extend instruction that moves execution to the extended opcode set. The processor steps (executes an instruction) by selecting the next execute function from either the main or extended table depending on the hidden register and then executing it.

Each entry in the table also has metadata to indicate how many cycles that instruction takes to emulate, and the total number of cycles executed is tracked in hidden registers. We need to track this because different components in the Gameboy operate at fixed cycle rates and keeping them in sync with the executed instructions is crucial to accurate emulation.

### Registers

There are 8 general purpose registers 8-bit registers B, C, A, F, D, E, H, L on the device. These registers can also be addressed as 16-bit registers for some instructions as BC, AF, DE, and HL. There are also special register PC and SP for the program counter (the memory address of the current instruction) and the stack pointer. Not all registers can be used in all operations, and some are used to store side effects of opcodes. The A register is used as the accumulator, and is usually the destination register for the result of arithmetic operations. The F register stores the flags after some opcodes, encoded as a bit vector that tells the program is the previous opcode carried, half carried, was negative, or was zero.

### Memory

The Gameboy uses an 8-bit Z80 CPU with a 16-bit memory addressing scheme. The address space is used to access system ROM, cartridge ROM, system RAM, cartridge RAM and to interface with other systems on the device through special memory registers. The console includes a small 256 byte ROM containing the boot up sequence code which scrolls a Nintendo logo across the screen and then does some primitive error checking on the cartridge. This ROM is unmapped from the address space after the initial boot sequence. There is also 8kb of addressable internal RAM and 8kb of video ram for sprites on the device. There are both mapped into fixed locations in the address space.

Programs are read from cartridges that are physically connected to the device and are directly addressable rather than being loaded into memory. The cartridge memory is access through reads to specific regions of memory which the device will automatically treat as a cartridge read. For example a read to 0x0 will access memory 0x0 in the first bank of the cartridge ROM while a write to 0x8000 will access the first byte of the on-board video RAM. The cartridge based design is useful because it allows the available RAM to be used only for program memory, while a system that has to load the program into memory would have reduced capability due to the reduction in usable RAM for program state.

The cartridge based design can be used to expand the available ROM and RAM though this increased the price of physical cartridges. Since the combined ROM and RAM of expanded cartridges cannot be addressed in 16 bits ROM banking where addressable parts of the cartridge ROM or RAM are remapped through writes to specific memory addresses. This requires careful programming since the program code being executed or some data required could be located in a bank that is remapped. This can be used to increase the ROM or RAM on system to 32kb.

Memory is represented in Mimic through a GameboyState structure which tracks the memory map, plus a series of ROM or RAM structures. The special memory registers are hardcoded into the top level structure at their given addresses and with the given read/write rules. The GameboyState routes ROM/RAM read and writes to corresponding structures for processing. ROM banking is also tracked in the this top level structure.

### Interrupts

The Gameboy uses CPU interrupts to notify the running software of hardware events. These events include screen events, timer events, and input events. When an interrupt is raised, a bit in the interrupted memory register is set to indicate it. If interrupts are enabled (toggled with a dedicated instruction) then the CPU will check if the corresponding bit for that interrupt is set in the interrupts enabled memory register. If both bits are set then the current PC will be pushed to the stack and the PC will be set to a fixed interrupt-specific location and interrupts will be disabled. The code at that location is then run (similar to a call instruction) and interrupts are re-enabled upon return.

### PPU

The pixel processing unit (PPU) finds the sprites referenced in the sprite, window, and background maps and draws them to the device screen. The PPU operates concurrently with the CPU and is the only component that can draw to the screen. Interaction with the PPU comes only through updates to the sprite memory, sprite maps, or special memory registers. Feedback for the CPU comes through writes to special registers and hardware interrupts.

### Clock

The Gameboy clock interacts with the CPU through special memory registers or through CPU interrupts. There are two clocks, one which ticks at a constant frequency and another which can be configured through writes to a special register. Since the clock is tied to the cycle rate of the CPU, and not the actual time, we implement the clock in Mimic through a structure that tracks the number of cycles the CPU has performed and updates it's own values accordingly.

### Screenshots

![Screenshot](/screenshots/screenshot1.png?raw=true "Screenshot 1")
![Screenshot](/screenshots/screenshot2.png?raw=true "Screenshot 2")
![Screenshot](/screenshots/screenshot3.png?raw=true "Screenshot 3")
![Screenshot](/screenshots/screenshot6.png?raw=true "Screenshot 6")

### Working

- System memory map
- Core instruction set
- Pixel-processing unit
- Background rendering
- Sprite rendering
- Interrupts
- Input
- Clock
- Memory Banking (Rudimentry)

### TODO

- Sound
- The emulator will not run in Debug mode because Rust errors on unsigned integer arithmetic over/underflow. This can be alleviated using the Wrapping<t> struct but requires explicit wrapping of every literal value. wrapping_add and wrapping_sub could be used on every instruction, but neither solution leads to readable code and I would prefer a solution that leaves the base logic understandable.
- Improve compatibility
