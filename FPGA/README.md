# Conjugate Gradient with FPGA
## Modules
```bash
$ module load ifpgasdk 520nmx
```
## How to compile
First compile the kernel:
For emulator:
```bash
$ cd device
$ ./buildEmulator.sh
```
For physical board:
```bash
$ cd device
$ ./buildBoard.sh
```

The previous phase can take several hours if you are compiling for the physical board.
Next compile the host program:
```bash
$ make
```

## How to run
Emulator:
```bash
$ CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./CG.exe
```

Physical board:
```bash
$ ./CG.exe
```

