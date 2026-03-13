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

struct _EachIndex{T,N,IS} <: AbstractArray{T,N}
    dims::NTuple{N,Int}
    indices::IS
end
_EachIndex(A::AbstractArray) =
    _EachIndex{typeof(firstindex(A)), ndims(A), typeof(eachindex(A))}(size(A), eachindex(A))
Base.size(ei::_EachIndex) = ei.dims
Base.getindex(ei::_EachIndex, i::Int) = ei.indices[i]
Base.IndexStyle(::Type{<:_EachIndex}) = Base.IndexLinear()

function _findmax_abs2_loc_cuda(inp::AbstractArray{<:Complex})
    isempty(inp) && throw(ArgumentError("input must be non-empty"))
    indices = _EachIndex(inp)
    dummy_index = firstindex(inp)
    Treal = float(real(eltype(inp)))

    function reduction(t1, t2)
        (x, i), (y, j) = t1, t2
        if isless(x, y)
            return t2
        elseif isequal(x, y)
            return (x, min(i, j))
        else
            return t1
        end
    end

    res = mapreduce(
        (x, i) -> (abs2(x), i),
        reduction,
        inp,
        indices;
        init=(typemin(Treal), dummy_index),
    )
    ndims(inp) == 1 ? res[2] : CartesianIndices(inp)[res[2]]
end

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
_findmax_abs2_loc(inp::CUDA.CuArray{<:Complex}, ::Union{Nothing,AbstractArray{<:Real}}=nothing) =
    _findmax_abs2_loc_cuda(inp)

_fft(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.fft(inp)
_ifft(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.ifft(inp)
_fftshift(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.fftshift(inp)
_ifftshift(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CUFFT.ifftshift(inp)
_scalar_at(inp::SubArray{<:Any,<:Any,<:CUDA.CuArray}, idx) = CUDA.@allowscalar inp[idx]
_findmax_abs2_loc(
    inp::SubArray{<:Any,<:Any,<:CUDA.CuArray},
    ::Union{Nothing,AbstractArray{<:Real}}=nothing,
) = _findmax_abs2_loc_cuda(inp)

end
