# Benchmarking FFTRegGPU

This benchmark suite is deterministic (fixed synthetic data seed) and reusable across commits.

## Setup

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
```

## Run

From the repository root:

```bash
julia --project=benchmark benchmark/run_benchmarks.jl --backend=cpu --samples=30 --output=benchmark/results/cpu.csv
```

To run CUDA too:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.add("CUDA")'
julia --project=benchmark benchmark/run_benchmarks.jl --backend=both --samples=20 --output=benchmark/results/both.csv
```

If CUDA is not functional on the machine, CUDA cases are skipped automatically.

## Useful options

- `--backend=cpu|cuda|both`
- `--samples=N`
- `--evals=N`
- `--seed=N`
- `--sizes=128x128,256x256`
- `--stack=256x256x64`
- `--noise=0.02`
- `--output=PATH`

Use `--help` for all options:

```bash
julia --project=benchmark benchmark/run_benchmarks.jl --help
```
