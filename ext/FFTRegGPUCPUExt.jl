"""
    FFTRegGPUCPUExt

CPU extension for FFTRegGPU backend hooks.

When `FFTW` is loaded and inputs are `StridedArray`s, FFTRegGPU internal hooks
dispatch to FFTW and AbstractFFTs shift operations.
"""
module FFTRegGPUCPUExt

using FFTRegGPU
using FFTW
using AbstractFFTs

import FFTRegGPU: _fft, _ifft, _fftshift, _ifftshift
import FFTRegGPU: _ifft!

_fft(inp::StridedArray) = FFTW.fft(inp)
_ifft(inp::StridedArray) = FFTW.ifft(inp)
_ifft!(inp::StridedArray{<:Complex}) = FFTW.ifft!(inp)
_fftshift(inp::StridedArray) = AbstractFFTs.fftshift(inp)
_ifftshift(inp::StridedArray) = AbstractFFTs.ifftshift(inp)

end
