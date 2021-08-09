## Mimic

An open source Gameboy emulator written in Rust that can use a command line interface as a screen and input device.

![Screenshot](/screenshots/screenshot4.png?raw=true "Screenshot 4")
![Screenshot](/screenshots/screenshot5.png?raw=true "Screenshot 5")
![Screenshot](/screenshots/screenshot1.png?raw=true "Screenshot 1")
![Screenshot](/screenshots/screenshot2.png?raw=true "Screenshot 2")
![Screenshot](/screenshots/screenshot3.png?raw=true "Screenshot 3")
![Screenshot](/screenshots/screenshot6.png?raw=true "Screenshot 6")

### Usage

cargo run --release -- --rom PATH --cli-mode

### CPU

The Gameboy uses a modified Z80 Zilog processor, an 8-bit processor with a 16-bit address bus. The CPU has a complex instruction set with variable length opcodes. The first byte of every opcode indicates which of the instructions it is. Because there are more than 256 instructions there is also an extended opcode which swaps the CPU to a second instruction set for the next instruction.

The CPU is represented in Mimic through two jump tables that are constructed at startup, one for the main opcode set and one for the extended set. Each entry in the jump table contains a pointer to an execute function which takes the current registers and memory and implements the opcode. A secret register is used to keep track of whether the previous instruction executed was the extend instruction that moves execution to the extended opcode set. The processor steps (executes an instruction) by selecting the next execute function from either the main or extended table depending on the hidden register and then executing it.

Each entry in the table also has metadata to indicate how many cycles that instruction takes to emulate, and the total number of cycles executed is tracked in hidden registers. We need to track this because different components in the Gameboy operate at fixed cycle rates and keeping them in sync with the executed instructions is crucial to accurate emulation.

### Registers

There are 8 general purpose registers 8-bit registers B, C, A, F, D, E, H, L on the device. These registers can also be addressed as 16-bit registers for some instructions as BC, AF, DE, and HL. There are also special register PC and SP for the program counter (the memory address of the current instruction) and the stack pointer. Not all registers can be used in all operations, and some are used to store side effects of opcodes. The A register is used as the accumulator, and is usually the destination register for the result of arithmetic operations. The F register stores the flags after some opcodes, encoded as a bit vector that tells the program is the previous opcode carried, half carried, was negative, or was zero.

### Memory

The Gameboy uses an 8-bit Z80 CPU with a 16-bit memory addressing scheme. The address space is used to access system ROM, cartridge ROM, system RAM, cartridge RAM and to interface with other systems on the device through special memory registers. The console includes a small 256 byte ROM containing the boot up sequence code which scrolls a Nintendo logo across the screen and then does some primitive error checking on the cartridge. This ROM is unmapped from the address space after the initial boot sequence. There is also 8kb of addressable internal RAM and 8kb of video ram for sprites on the device. There are both mapped into fixed locations in the address space.

Programs are read from cartridges that are physically connected to the device and are directly addressable rather than being loaded into memory. The cartridge memory is access through reads to specific regions of memory which the device will automatically treat as a cartridge read. For example a read to 0x0 will access memory 0x0 in the first bank of the cartridge ROM while a write to 0x8000 will access the first byte of the on-board video RAM. The cartridge based design is useful because it allows the available RAM to be used only for program memory, while a system that has to load the program into memory would have reduced capability due to the reduction in usable RAM for program state.

The cartridge based design can be used to expand the available ROM and RAM though this increased the price of physical cartridges. Since the combined ROM and RAM of expanded cartridges cannot be addressed in 16 bits ROM banking where addressable parts of the cartridge ROM or RAM are remapped through writes to specific memory addresses. This requires careful programming since the program code being executed or some data required could be located in a bank that is remapped. This can be used to increase the ROM or RAM on system to 32kb.

Memory is represented in Mimic through a GameboyState structure which tracks the memory map, plus a series of ROM or RAM structures. The special memory registers are hardcoded into the top level structure at their given addresses and with the given read/write rules. The GameboyState routes ROM/RAM read and writes to corresponding structures for processing. ROM banking is also tracked in the this top level structure.

### Clock

The Gameboy clock interacts with the CPU through special memory registers or through CPU interrupts. There are two clocks, one which ticks at a constant frequency and another which can be configured through writes to a special register. Since the clock is tied to the cycle rate of the CPU, and not the actual time, we implement the clock in Mimic through a structure that tracks the number of cycles the CPU has performed and updates it's own values accordingly.

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
