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
    copyto!(img_stack_reg, img_stack_reg_g) # copy result to GPU
    save_zstack(t, img_stack_reg)
end
```

## Performance
### Benchmark command
```bash
julia --project=benchmark benchmark/run_benchmarks.jl --backend=both --samples=20 --sizes=64x64,128x128,256x256,512x512,2048x2048 --stack=128x128x128,256x256x256 --output=benchmark/results/both.csv
```

### Benchmark results


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