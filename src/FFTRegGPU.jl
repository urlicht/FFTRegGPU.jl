module FFTRegGPU
using AbstractFFTs

include("dftreg_translate.jl")

export dftreg!,
    dftreg_subpix!,
    subpix_shift!,
    dftreg_resample,
    dftreg_resample!,
    reg_stack_translate!,
    dftreg_gpu!,
    dftreg_subpix_gpu!,
    subpix_shift_gpu!,
    dftreg_resample_gpu!

# Backward-compatible wrappers around the backend-agnostic API.
dftreg_gpu!(args...) = dftreg!(args...)
dftreg_subpix_gpu!(args...) = dftreg_subpix!(args...)
subpix_shift_gpu!(args...) = subpix_shift!(args...)
dftreg_resample_gpu(args...; kwargs...) = dftreg_resample(args...; kwargs...)
dftreg_resample_gpu!(args...; kwargs...) = dftreg_resample!(args...; kwargs...)

end # module
