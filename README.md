## Available Simulation Modes
- **Functional Mode**  
  Performs a basic simulation of your CUDA program, focusing on correctness and output verification.  
  Fastest simulation option, ideal for testing program logic and debugging.

- **Cycle Accurate Mode**  
  Provides detailed architectural simulation by modeling the GPU hardware cycle-by-cycle.  
  Slower, but offers performance insights and helps predict actual GPU performance.

---

## Supported CUDA Features
- Constant memory  
- Global memory  
- Shared memory  
- CUDA Streams  
- CUDA Events  
- Atomic operations  

---

## Current Limitations
- Texture memory  
- Dynamic parallelism  
- CUDA Graphs  
- Warp-level primitives (shuffle, vote, etc.)  

---

## Acknowledgments
Special thanks to the **Accel-Sim team** for their incredible work.  

📖 Reference:  
Mahmoud Khairy, Jason Shen, Tor M. Aamodt, and Timothy G. Rogers  
*"Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling"*  
In *The 47th International Symposium on Computer Architecture*, May 2020  

---

## Commands

```sh
leetgpu run FILE.cu         # Execute CUDA code on remote GPU
leetgpu cuda-version        # Show CUDA version for specified mode
leetgpu list-gpus           # Show available GPU models
leetgpu upgrade             # Upgrade the CLI to the latest version
```

---

## Utilities

```sh
leetgpu --version           # Show CLI version
leetgpu --help              # Show this help message
```

---

## Run Options

```sh
[--mode | -m] MODE          # Simulation mode (functional or cycle-accurate)
[--gpu  | -g] GPU           # GPU (required for cycle-accurate mode)
```

---

## Examples

```sh
leetgpu run kernel.cu                                           # Run in functional mode (default)
leetgpu run kernel.cu --mode functional                         # Run in functional mode
leetgpu run kernel.cu --mode cycle-accurate --gpu NVIDIA GV100  # Run in cycle-accurate mode
leetgpu cuda-version --mode functional                          # Show CUDA version for functional mode
leetgpu cuda-version --mode cycle-accurate                      # Show CUDA version for cycle-accurate mode
leetgpu list-gpus                                               # Show available GPU models
leetgpu --version                                               # Show CLI version
leetgpu upgrade                                                 # Upgrade CLI to latest version
```
