# FFTRegGPU.jl
[![CI](https://github.com/urlicht/FFTRegGPU.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/urlicht/FFTRegGPU.jl/actions/workflows/ci.yml)

Fast FFT GPU registration using [phase correlation](https://en.wikipedia.org/wiki/Phase_correlation). This GPU version is based on [SubpixelRegistration.jl](https://github.com/romainFr/SubpixelRegistration.jl)  
Currently it only supports translation.


## Usage and example  
![before vs after registration showing a slice (xy)](docs/img/before_after_xy_slice.png)
![before vs after registration showing xz MIP for a volume](docs/img/before_after_xz_MIP.png)

### Backend setup
Load FFTRegGPU with the backend package you want to use:

```julia
using FFTRegGPU
using FFTW    # CPU backend
# using CUDA  # CUDA backend
```

Backend selection is determined by array type at call time:
- `CuArray` inputs use the CUDA extension (CUFFT/CUDA kernels).
- `Array`/`StridedArray` inputs use the CPU extension (FFTW).
- You can load both `CUDA` and `FFTW`; dispatch still follows the input array types.

### Registering a set of 2D images
```julia
# allocate GPU memory
img1_g = CuArray{Float32}(undef, size_x, size_y)
img2_g = CuArray{Float32}(undef, size_x, size_y)
img1_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
img2_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
CC_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
N_g = CuArray{Float32}(undef, size_x, size_y)
img2_reg_g = CuArray{Float32}(undef, size_x, size_y)

# copy to GPU
copyto!(img1_g, Float32.(img1))
copyto!(img2_g, Float32.(img2))

# perform FFT using the active backend
img1_f_g .= fft(img1_g)
img2_f_g .= fft(img2_g)

# register (find the optimal translation)
error, shift, diffphase = dftreg!(img1_f_g, img2_f_g, CC_g)

# resample the moving image (in-place)
dftreg_resample!(img2_reg_g, img2_f_g, N_g, shift, diffphase)

# copy to CPU
Array(img2_reg_g)
```
### Registering a set of 2D images (subpixel registration)
Use the function `dftreg_subpix!`. For the argument `CC2x_g`, the array size should be 2x of the image size.
```julia
# allocate GPU memory
img1_g = CuArray{Float32}(undef, size_x, size_y)
img2_g = CuArray{Float32}(undef, size_x, size_y)
img1_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
img2_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
CC2x_g = CuArray{Complex{Float32}}(undef, 2 * size_x, 2 * size_y)
N_g = CuArray{Float32}(undef, size_x, size_y)
img2_reg_g = CuArray{Float32}(undef, size_x, size_y)

# copy to GPU
copyto!(img1_g, Float32.(img1))
copyto!(img2_g, Float32.(img2))

# perform FFT using the active backend
img1_f_g .= fft(img1_g)
img2_f_g .= fft(img2_g)

# register (find the optimal translation)
error, shift, diffphase = dftreg_subpix!(img1_f_g, img2_f_g, CC2x_g)

# resample the moving image (in-place)
dftreg_resample!(img2_reg_g, img2_f_g, N_g, shift, diffphase)

# copy to CPU
Array(img2_reg_g)
```

### Registering z-stack
- Moving targets on the stage can cause shearing in z-stack. To correct this, the images within the stack are registered together.
- `reg_stack_translate!` is a memory-efficient and convenient function to register the frames in each z-stack. Here in the example, the script loads the z-stack at each time point, registers it, and then saves the registered z-stack.  
- To ensure this runs on CUDA: make all working arrays `CuArray`, check `CUDA.functional()`, and optionally set `CUDA.allowscalar(false)` to catch accidental scalar fallback.
- For repeated large host<->device transfers, use a long-lived pinned host buffer (`CUDA.pin`) for better copy throughput.
```julia
size_x, size_y, size_z = 256, 256, 94
img_stack_h = CUDA.pin(Array{Float32}(undef, size_x, size_y, size_z))
img_stack_reg_g = CuArray{Float32}(undef, size_x, size_y, size_z)
img1_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
img2_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
CC2x_g = CuArray{Complex{Float32}}(undef, 2 * size_x, 2 * size_y)
N_g = CuArray{Float32}(undef, size_x, size_y)

@showprogress for t = 1:100
    get_zstack!(img_stack_h, t) # load data to pinned host buffer
    copyto!(img_stack_reg_g, img_stack_h) # copy data to GPU
    reg_stack_translate!(img_stack_reg_g, img1_f_g, img2_f_g, CC2x_g, N_g) # register
    copyto!(img_stack_h, img_stack_reg_g) # copy result to CPU
    save_zstack(t, img_stack_h)
end
```

### Registering z-stack in batches (CUDA, higher throughput)
If you have many stacks (`t = 1:nt`), process them in micro-batches to improve GPU utilization and overlap CPU I/O with GPU work.

```julia
using FFTRegGPU
using CUDA
using ProgressMeter
using Base.Threads

CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)

size_x, size_y, size_z = 256, 256, 94
nt = 100
B = 4  # micro-batch size; tune 2, 4, 8...

# One independent workspace per batch slot.
h_in = [CUDA.pin(Array{Float32}(undef, size_x, size_y, size_z)) for _ in 1:B]
h_out = [CUDA.pin(Array{Float32}(undef, size_x, size_y, size_z)) for _ in 1:B]
d_stack = [CuArray{Float32}(undef, size_x, size_y, size_z) for _ in 1:B]
img1_f = [CuArray{ComplexF32}(undef, size_x, size_y) for _ in 1:B]
img2_f = [CuArray{ComplexF32}(undef, size_x, size_y) for _ in 1:B]
CC2x = [CuArray{ComplexF32}(undef, 2 * size_x, 2 * size_y) for _ in 1:B]
N = [CuArray{Float32}(undef, size_x, size_y) for _ in 1:B]
reg_param = [Dict{Int,Any}() for _ in 1:B]

@showprogress for t0 in 1:B:nt
    k = min(B, nt - t0 + 1)

    # Load stacks on CPU.
    @threads for i in 1:k
        get_zstack!(h_in[i], t0 + i - 1)
    end

    # Process stacks concurrently on GPU.
    @sync for i in 1:k
        @spawn begin
            empty!(reg_param[i])
            CUDA.@sync begin
                copyto!(d_stack[i], h_in[i]) # H2D
                reg_stack_translate!(d_stack[i], img1_f[i], img2_f[i], CC2x[i], N[i]; reg_param=reg_param[i])
                copyto!(h_out[i], d_stack[i]) # D2H
            end
        end
    end

    # Save stacks on CPU.
    @threads for i in 1:k
        save_zstack(t0 + i - 1, h_out[i])
    end
end
```

Notes:
- Run Julia with multiple threads (e.g. `julia -t auto`) to benefit from `@threads`/`@spawn`.
- `B` is the main tuning parameter; increase it until throughput stops improving or GPU memory becomes limiting.

## Performance
### Benchmark command
```bash
julia --project=benchmark benchmark/run_benchmarks.jl --backend=both --samples=20 --sizes=64x64,128x128,256x256,512x512,2048x2048 --stack=128x128x128,256x256x256 --output=benchmark/results/both.csv
```

### Benchmark results
Time in ms

| backend | case | dims | median | mean | min | memory | allocs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| cpu | dftreg | 64x64 | 0.059 | 0.066 | 0.059 | 400 | 7 |
| cpu | dftreg_subpix | 64x64 | 0.403 | 0.419 | 0.400 | 492568 | 263 |
| cpu | dftreg | 128x128 | 0.203 | 0.214 | 0.202 | 400 | 7 |
| cpu | dftreg_subpix | 128x128 | 4.515 | 4.310 | 3.885 | 1755864 | 263 |
| cpu | dftreg | 256x256 | 2.944 | 2.951 | 2.912 | 400 | 7 |
| cpu | dftreg_subpix | 256x256 | 96.249 | 86.929 | 19.879 | 6640136 | 285 |
| cpu | dftreg | 512x512 | 17.250 | 17.164 | 15.607 | 400 | 7 |
| cpu | dftreg_subpix | 512x512 | 202.291 | 194.513 | 76.812 | 25846040 | 295 |
| cpu | dftreg | 2048x2048 | 282.298 | 283.155 | 278.946 | 400 | 7 |
| cpu | dftreg_subpix | 2048x2048 | 3731.667 | 3731.667 | 3575.166 | 405360272 | 295 |
| cpu | reg_stack_translate | 128x128x128 | 584.774 | 590.850 | 582.729 | 224239024 | 38492 |
| cpu | reg_stack_translate | 256x256x256 | 29150.903 | 29150.903 | 29150.903 | 1697874352 | 83396 |
| cuda | dftreg | 64x64 | 0.613 | 0.631 | 0.577 | 15120 | 488 |
| cuda | dftreg_subpix | 64x64 | 1.487 | 1.487 | 1.084 | 128000 | 2406 |
| cuda | dftreg | 128x128 | 0.219 | 0.240 | 0.216 | 15088 | 476 |
| cuda | dftreg_subpix | 128x128 | 1.560 | 1.575 | 1.077 | 176576 | 2406 |
| cuda | dftreg | 256x256 | 0.220 | 0.241 | 0.216 | 14976 | 481 |
| cuda | dftreg_subpix | 256x256 | 1.631 | 1.659 | 1.203 | 273152 | 2434 |
| cuda | dftreg | 512x512 | 0.223 | 0.241 | 0.217 | 15264 | 499 |
| cuda | dftreg_subpix | 512x512 | 2.090 | 2.034 | 1.396 | 467408 | 2575 |
| cuda | dftreg | 2048x2048 | 1.845 | 1.853 | 1.619 | 24432 | 1028 |
| cuda | dftreg_subpix | 2048x2048 | 13.527 | 13.277 | 12.191 | 1632336 | 3147 |
| cuda | reg_stack_translate | 128x128x128 | 198.666 | 228.952 | 182.387 | 25326976 | 369055 |
| cuda | reg_stack_translate | 256x256x256 | 410.275 | 413.359 | 394.183 | 76830816 | 751483 |

### Configuration
```
CUDA toolchain: 
- runtime 12.9, artifact installation
- driver 550.107.2 for 12.4
- compiler 12.9

CUDA libraries: 
- CUBLAS: 12.9.1
- CURAND: 10.3.10
- CUFFT: 11.4.1
- CUSOLVER: 11.7.5
- CUSPARSE: 12.5.10
- CUPTI: 2025.2.1 (API 12.9.1)
- NVML: 12.0.0+550.107.2

Julia packages: 
- CUDA: 5.10.1
- GPUArrays: 11.4.1
- GPUCompiler: 1.8.2
- KernelAbstractions: 0.9.40
- CUDA_Driver_jll: 13.2.0+0
- CUDA_Compiler_jll: 0.4.2+0
- CUDA_Runtime_jll: 0.20.1+0

Toolchain:
- Julia: 1.12.5
- LLVM: 18.1.7

1 device:
  0: NVIDIA GeForce RTX 3060 (sm_86, 11.752 GiB / 12.000 GiB available)
```
