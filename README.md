# Plank Universe

A CUDA-based simulation of a Planck-scale network aiming to model emergent time, bidirectional causality, and singularity-free physics. This project explores a universe built from quantized Planck volumes (\( V_P \approx 1.767 \times 10^{-104} \, \text{m}^3 \)), with time emerging from an imaginary clock (\(\tau_i = i (N - n) t_P\)) and the potential for present-to-past interactions. Inspired by loop quantum gravity, Guth’s “universe from nothing,” and a rejection of singularities, it targets applications like finite-density black hole cores and a non-collapsing early universe.

## Status
- **Current**: Simulates up to 2 billion Planck units (`Plank` structs, ~84 bytes) with probabilistic growth, boson-mediated interactions, and particle types. Reaches 21–22 ticks but hits connection loss after tick 18/19 due to GPU memory limits (16GB on RTX 4060 Ti).
- **Challenges**: Heavy logging overhead (dominates runtime), memory bottlenecks when exceeding GPU capacity, and missing imaginary clock/retrocausality features.
- **Goals**: Implement \(\tau_i = i (N - n) t_P\), \(t = \sqrt{(N_{\text{max}} - (N - n)) t_P}\), add retrocausality, improve network structure (e.g., 3D lattice), and leverage dual GPUs (4060 Ti + 3050).

## Features
- **Network**: Up to 2B Planck units, each with energy, particle types, and up to 6 neighbors connected via bosons.
- **Dynamics**: Probabilistic growth (2.5x factor), energy updates, and particle interactions, mimicking quantized space-time expansion.
- **Physics**: Finite quantities (Planck-as-zero), aiming for no singularities or infinities.
- **Hardware**: Optimized for a high-end workstation (dual E5-2667v4 CPUs, 256 GB DDR4 ECC RAM, RTX 4060 Ti 16GB, RTX 3050 8GB, 10 TB SSD/NVMe/SAS storage).
- **Outputs**: CSV logs (`tick_summary.csv`, `cell_history.csv`, `map_info.csv`) for plank states and connections.

## Requirements
- **OS**: Ubuntu 22.04 (tested on WSL under Windows 10)
- **Compiler**: NVIDIA CUDA Toolkit (NVCC, tested with CUDA 11.8 or later)
- **Hardware**:
- GPU: NVIDIA RTX 4060 Ti (16GB) or similar; RTX 3050 (8GB) supported but unused.
- CPU: Dual Intel Xeon E5-2667v4 or equivalent.
- RAM: 256 GB DDR4 ECC (uses ~200 GB at peak).
- Storage: 10 TB (SSD/NVMe/SAS RAID-1), ~5 TB free for logs.
- **Libraries**: Standard C (stdio, stdlib), CUDA runtime.

## Build and Run
1. **Clone the Repository**:
```bash
git clone https://github.com/pnoliveir/plank_universe.git
cd plank_universe
```
2. **Compile**:
```bash
nvcc -o sim sim.cu
```
3. **Run**:
```bash
./sim
```

## Run notes
• Outputs: tick_summary.csv (tick counts), cell_history.csv (plank states), map_info.csv (connections).

• Logs consume significant I/O; ensure ~350/400GB free storage.

• Simulation runs ~21–22 ticks, using ~200 GB RAM and 16GB GPU memory.


## Known Issues
• Connection Loss: Neighbor connections fail after tick 18/19 when plankCount exceeds GPU memory (~184M planks), reducing growth rate.

• Logging Overhead: Writing CSVs for all planks/connections dominates runtime.

• Memory: Host-GPU transfers slow down when batches exceed 16GB.

• Missing Features: Imaginary clock (\tau_i) and retrocausality not yet implemented.

## Contributing

Join the revolution to build a singularity-free universe! We need help with:

• Multi-GPU support (RTX 4060 Ti + 3050).

• Optimizing logging (e.g., buffering, async I/O).

• Implementing imaginary clock (\tau_i = i (N - n) t_P) and retrocausality.

• Designing a 3D lattice network for realistic space-time.

• Adding black hole cores and cosmological inflation scenarios.

## How to Contribute:

• Fork the repo and submit pull requests.

• Share ideas on X (@peterblood850) or open GitHub issues.

• Test on similar hardware (Ubuntu WSL, CUDA GPUs).

• Check docs/model.md (coming soon) for the full theoretical model.

## Hardware Details
• CPUs: Dual Intel Xeon E5-2667v4 (8 cores each, 3.2 GHz).

• RAM: 256 GB DDR4 ECC 2400 MHz.

• GPUs: NVIDIA RTX 4060 Ti (16GB, primary), RTX 3050 (8GB, unused).

• Storage: 10 TB total (SSD, NVMe, SAS RAID-1), ~5 TB free.

• OS: Ubuntu 22.04 on WSL (Windows 10 host).

## Roadmap

• Short-Term:

• Fix connection loss after tick 18/19 (global ID tracking, relaxed cleanup).

• Reduce logging overhead (buffering, selective output).

• Enable RTX 3050 for ~276M planks.

• Mid-Term:

• Add imaginary clock (\tau_i, t) to Plank struct.

• Implement retrocausality (present-to-past energy updates).

• Long-Term:

• Model black hole cores (\rho \leq 1.13 \times 10^{135} \, \text{kg/m}^3).

• Simulate early universe expansion (Guth’s model).

• Collaborate with X’s LQG, gravastar, and cosmology communities.

## License
MIT License

Copyright (c) 2025 Pedro Oliveira

Contact

Ping me on X (@peterblood850) or open a GitHub issue to discuss emergent time, retrocausality, or how to scale this to 2B planks without crashing my GPUs! 
