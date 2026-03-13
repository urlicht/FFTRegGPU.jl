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
