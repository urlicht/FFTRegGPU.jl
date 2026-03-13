"""
    FFTRegGPUCUDAExt

CUDA extension for FFTRegGPU backend hooks.

When `CUDA` is loaded and inputs are `CuArray` (or views over `CuArray`), FFT
and shift operations dispatch to CUFFT/CUDA implementations.
"""
module FFTRegGPUCUDAExt

using FFTRegGPU
using CUDA

import FFTRegGPU: _fft, _ifft, _ifft!, _fftshift, _ifftshift, _scalar_at
import FFTRegGPU: _plan_fft_inplace, _fft_inplace!
import FFTRegGPU: _findmax_abs2_loc

_fft(inp::CUDA.CuArray) = CUDA.CUFFT.fft(inp)
_ifft(inp::CUDA.CuArray) = CUDA.CUFFT.ifft(inp)
_ifft!(inp::CUDA.CuArray{<:Complex}) = CUDA.CUFFT.ifft!(inp)
_plan_fft_inplace(inp::CUDA.CuArray{<:Complex}) = CUDA.CUFFT.plan_fft!(inp)
_fft_inplace!(inp::CUDA.CuArray{<:Complex}) = CUDA.CUFFT.fft!(inp)
_fft_inplace!(inp::CUDA.CuArray{<:Complex}, ::Nothing) = CUDA.CUFFT.fft!(inp)
_fft_inplace!(inp::CUDA.CuArray{<:Complex}, plan) = (plan * inp)
_fftshift(inp::CUDA.CuArray) = CUDA.CUFFT.fftshift(inp)
_ifftshift(inp::CUDA.CuArray) = CUDA.CUFFT.ifftshift(inp)
_scalar_at(inp::CUDA.CuArray, idx) = CUDA.@allowscalar inp[idx]
_findmax_abs2_loc(inp::CUDA.CuArray{<:Complex}, ::Union{Nothing,AbstractArray{<:Real}}=nothing) = findmax(abs2, inp)[2]

_fft(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.fft(inp)
_ifft(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.ifft(inp)
_fftshift(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.fftshift(inp)
_ifftshift(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.ifftshift(inp)
_scalar_at(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}, idx) = CUDA.@allowscalar inp[idx]
_findmax_abs2_loc(
    inp::SubArray{<:Any,<:Any,<:CUDA.CuArray},
    ::Union{Nothing,AbstractArray{<:Real}}=nothing,
) = findmax(abs2, inp)[2]

end
