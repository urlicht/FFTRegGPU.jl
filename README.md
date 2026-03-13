# FFTRegGPU.jl
Fast FFT GPU registration using [phase correlation](https://en.wikipedia.org/wiki/Phase_correlation). This GPU version is based on [SubpixelRegistration.jl](https://github.com/romainFr/SubpixelRegistration.jl)  
Currently it only supports translation.


## Usage and example  
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
```julia
size_x, size_y, size_z = 256, 256, 94
img_stack_reg = zeros(Float32, size_x, size_y, size_z)
img_stack_reg_g = CuArray{Float32}(undef, size_x, size_y, size_z)
img1_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
img2_f_g = CuArray{Complex{Float32}}(undef, size_x, size_y)
CC2x_g = CuArray{Complex{Float32}}(undef, 2 * size_x, 2 * size_y)
N_g = CuArray{Float32}(undef, size_x, size_y)

@showprogress for t = 1:100
    copyto!(img_stack_reg_g, get_zstack(t)) # copy data to GPU
    reg_stack_translate!(img_stack_reg_g, img1_f_g, img2_f_g, CC2x_g, N_g) # register
    copyto!(img_stack_reg, img_stack_reg_g) # copy result to CPU
    save_zstack(t, img_stack_reg)
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
| cpu | dftreg | 64x64 | 0.090 | 0.115 | 0.088 | 400 | 7 |
| cpu | dftreg_subpix | 64x64 | 0.864 | 0.860 | 0.631 | 492568 | 263 |
| cpu | dftreg | 128x128 | 0.326 | 0.342 | 0.313 | 400 | 7 |
| cpu | dftreg_subpix | 128x128 | 4.465 | 4.393 | 3.797 | 1755928 | 263 |
| cpu | dftreg | 256x256 | 2.545 | 2.571 | 2.328 | 400 | 7 |
| cpu | dftreg_subpix | 256x256 | 101.268 | 104.862 | 15.536 | 6640424 | 285 |
| cpu | dftreg | 512x512 | 11.170 | 11.478 | 10.875 | 400 | 7 |
| cpu | dftreg_subpix | 512x512 | 198.892 | 194.983 | 67.973 | 25846328 | 295 |
| cpu | dftreg | 2048x2048 | 276.238 | 277.944 | 272.715 | 400 | 7 |
| cpu | dftreg_subpix | 2048x2048 | 2555.666 | 2555.666 | 2415.997 | 405360432 | 295 |
| cpu | reg_stack_translate | 128x128x128 | 868.029 | 861.984 | 828.896 | 290972976 | 41278 |
| cpu | reg_stack_translate | 256x256x256 | 37213.014 | 37213.014 | 37213.014 | 2233026768 | 88998 |
| cuda | dftreg | 64x64 | 0.965 | 0.976 | 0.650 | 14816 | 471 |
| cuda | dftreg_subpix | 64x64 | 2.499 | 2.511 | 2.278 | 127856 | 2397 |
| cuda | dftreg | 128x128 | 0.371 | 0.389 | 0.341 | 14848 | 472 |
| cuda | dftreg_subpix | 128x128 | 2.635 | 2.608 | 1.907 | 176448 | 2398 |
| cuda | dftreg | 256x256 | 0.383 | 0.397 | 0.340 | 14880 | 475 |
| cuda | dftreg_subpix | 256x256 | 2.741 | 2.738 | 2.082 | 273072 | 2428 |
| cuda | dftreg | 512x512 | 0.372 | 0.397 | 0.346 | 15184 | 494 |
| cuda | dftreg_subpix | 512x512 | 3.355 | 3.342 | 2.637 | 467232 | 2564 |
| cuda | dftreg | 2048x2048 | 2.382 | 2.383 | 1.952 | 22256 | 892 |
| cuda | dftreg_subpix | 2048x2048 | 13.917 | 13.958 | 12.945 | 1626624 | 2834 |
| cuda | reg_stack_translate | 128x128x128 | 341.053 | 341.947 | 337.756 | 26691536 | 405533 |
| cuda | reg_stack_translate | 256x256x256 | 739.277 | 739.101 | 734.337 | 79566528 | 824039 |

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
