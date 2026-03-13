"""
    FFTRegGPU

Phase-correlation based translation registration for 2D images with array-type
driven backend dispatch (CPU via FFTW, GPU via CUDA extensions).

The core API expects Fourier-domain images (`fft(img)`) and exposes:
- `dftreg!` for integer-pixel translation estimation,
- `dftreg_subpix!` for subpixel translation estimation,
- `subpix_shift!` and `dftreg_resample!` for Fourier-domain shifting/resampling,
- `reg_stack_translate!` for sequential z-stack registration.
"""
module FFTRegGPU
using AbstractFFTs

include("dftreg_translate.jl")

export dftreg!,
    dftreg_subpix!,
    subpix_shift!,
    dftreg_resample,
    dftreg_resample!,
    reg_stack_translate!

end # module
