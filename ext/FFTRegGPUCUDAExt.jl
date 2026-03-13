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

_fft(inp::CUDA.CuArray) = CUDA.CUFFT.fft(inp)
_ifft(inp::CUDA.CuArray) = CUDA.CUFFT.ifft(inp)
_ifft!(inp::CUDA.CuArray{<:Complex}) = CUDA.CUFFT.ifft!(inp)
_fftshift(inp::CUDA.CuArray) = CUDA.CUFFT.fftshift(inp)
_ifftshift(inp::CUDA.CuArray) = CUDA.CUFFT.ifftshift(inp)
_scalar_at(inp::CUDA.CuArray, idx) = CUDA.@allowscalar inp[idx]

_fft(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.fft(inp)
_ifft(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.ifft(inp)
_fftshift(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.fftshift(inp)
_ifftshift(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.ifftshift(inp)
_scalar_at(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}, idx) = CUDA.@allowscalar inp[idx]

end
