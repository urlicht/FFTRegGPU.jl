module FFTRegGPUCUDAExt

using FFTRegGPU
using CUDA

import FFTRegGPU: _fft, _ifft, _fftshift, _ifftshift

_fft(inp::CUDA.CuArray) = CUDA.CUFFT.fft(inp)
_ifft(inp::CUDA.CuArray) = CUDA.CUFFT.ifft(inp)
_fftshift(inp::CUDA.CuArray) = CUDA.CUFFT.fftshift(inp)
_ifftshift(inp::CUDA.CuArray) = CUDA.CUFFT.ifftshift(inp)

_fft(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.fft(inp)
_ifft(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.ifft(inp)
_fftshift(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.fftshift(inp)
_ifftshift(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.ifftshift(inp)

end
