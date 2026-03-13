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

## Compare `reg_stack_translate!` across two commits

Run this standalone commit comparison benchmark (CUDA only):

```bash
julia --project=benchmark benchmark/compare_reg_stack_translate_commits.jl \
  --current-ref=HEAD \
  --previous-ref=HEAD~1 \
  --stack=256x256x64,256x256x256 \
  --samples=20
```

You can manually provide any two refs (branches, tags, or SHAs):

```bash
julia --project=benchmark benchmark/compare_reg_stack_translate_commits.jl \
  --current-ref=3a1b2c4 \
  --previous-ref=f09d8e7
```

The script prints:
- Median timing comparison (`current_ms`, `prev_ms`, ratio, percent delta)
- Output equivalence (`isapprox` + max absolute difference) per stack size

## Useful options

- `--backend=cpu|cuda|both`
- `--samples=N`
- `--evals=N`
- `--seed=N`
- `--sizes=128x128,256x256`
- `--stack=256x256x64`
- `--noise=0.02`
- `--output=PATH`
- `--current-ref=REF`
- `--previous-ref=REF`

Use `--help` for all options:

```bash
julia --project=benchmark benchmark/run_benchmarks.jl --help
```
