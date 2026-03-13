_fft(inp::AbstractArray) = fft(inp)
_ifft(inp::AbstractArray) = ifft(inp)
_fftshift(inp::AbstractArray) = fftshift(inp)
_ifftshift(inp::AbstractArray) = ifftshift(inp)

function dftups(inp::AbstractArray{T,N}, no, usfac::Int=1, offset=zeros(N)) where {T,N}
    sz = [size(inp)...]
    permV = 1:N
    for i in permV
        inp = permutedims(inp, [i; deleteat!(collect(permV), i)])
        kern = exp.(
            (-1im * 2 * pi / (sz[i] * usfac)) *
            ((0:(no - 1)) .- offset[i]) *
            transpose(ifftshift(0:(sz[i] - 1)) .- floor(sz[i] / 2)),
        )
        d = size(inp)[2:N]
        inp = kern * reshape(inp, Val(2))
        inp = reshape(inp, (no, d...))
    end
    permutedims(inp, collect(ndims(inp):-1:1))
end

function dftreg!(
    img1_f::AbstractMatrix{<:Complex},
    img2_f::AbstractMatrix{<:Complex},
    CC::AbstractMatrix{<:Complex},
)
    size(img1_f) == size(img2_f) || throw(DimensionMismatch("img1_f and img2_f must have the same size"))
    size(img1_f) == size(CC) || throw(DimensionMismatch("CC must have the same size as img1_f"))

    L = length(img1_f)
    CC .= _ifft(img1_f .* conj.(img2_f))
    loc = argmax(abs.(CC))
    CCmax = Array(CC)[loc]
    rfzero = sum(abs2, img1_f) / L
    rgzero = sum(abs2, img2_f) / L
    error = abs(1 - CCmax * conj(CCmax) / (rgzero * rfzero))
    diffphase = atan(imag(CCmax), real(CCmax))

    indi = size(img1_f)
    ind2 = tuple([div(x, 2) for x in indi]...)
    locI = Tuple(loc)
    shift = zeros(Float64, length(locI))

    for i in eachindex(locI)
        if locI[i] > ind2[i]
            shift[i] = locI[i] - indi[i] - 1
        else
            shift[i] = locI[i] - 1
        end
    end

    error, shift, diffphase
end

function dftreg_subpix!(
    img1_f::AbstractMatrix{<:Complex},
    img2_f::AbstractMatrix{<:Complex},
    CC2x::AbstractMatrix{<:Complex},
    up_fac::Int=10,
)
    size(img1_f) == size(img2_f) || throw(DimensionMismatch("img1_f and img2_f must have the same size"))
    up_fac > 0 || throw(ArgumentError("up_fac must be positive"))

    # initial estimate by 2x upsample
    dim_input = collect(size(img1_f))
    ranges = [(x + 1 - div(x, 2)):(x + 1 + div(x - 1, 2)) for x in dim_input]
    expected_cc_size = tuple((2 .* dim_input)...)
    size(CC2x) == expected_cc_size ||
        throw(DimensionMismatch("CC2x must have size $(expected_cc_size), got $(size(CC2x))"))

    CC2x .= 0
    CC2x[ranges...] .= _fftshift(img1_f) .* conj.(_fftshift(img2_f))

    # compute cross-correlation and locate the peak
    CC2x_corr = _ifft(_ifftshift(CC2x))
    loc = argmax(abs.(CC2x_corr))

    indi = size(CC2x_corr)
    locI = collect(Tuple(loc))
    CC2x_max = Array(CC2x_corr)[loc]

    # obtain shift in original pixel grid
    ind2 = indi ./ 2
    shift = zeros(Float64, length(locI))
    for i in eachindex(locI)
        if locI[i] > ind2[i]
            shift[i] = locI[i] - indi[i] - 1
        else
            shift[i] = locI[i] - 1
        end
    end
    shift = shift / 2

    # refine subpixel estimation
    if up_fac > 2
        shift = round.(Int, shift * up_fac) / up_fac # initial shift estimate
        dft_shift = ceil(up_fac * 1.5) / 2 # center of output at dft_shift + 1

        CC_refine = dftups(
            Array(img2_f .* conj.(img1_f)),
            ceil(Int, up_fac * 1.5),
            up_fac,
            dft_shift .- shift .* up_fac,
        ) / (prod(ind2) * up_fac ^ 2)

        loc = argmax(abs.(CC_refine))
        locI = Tuple(loc)
        CC_refine_max = CC_refine[loc]
        locI = locI .- dft_shift .- 1
        shift = shift .+ locI ./ up_fac

        img1_00 = dftups(Array(img1_f .* conj.(img1_f)), 1, up_fac)[1] / (prod(ind2) * up_fac ^ 2)
        img2_00 = dftups(Array(img2_f .* conj.(img2_f)), 1, up_fac)[1] / (prod(ind2) * up_fac ^ 2)
        CC_max = CC_refine_max
    else
        img1_00 = sum(img1_f .* conj.(img1_f)) / prod(indi)
        img2_00 = sum(img2_f .* conj.(img2_f)) / prod(indi)
        CC_max = CC2x_max
    end

    error = 1 - CC_max * conj(CC_max) / (img1_00 * img2_00)
    error = sqrt(abs(error))
    diffphase = atan(imag(CC_max), real(CC_max))

    error, shift, diffphase
end

function subpix_shift!(
    img_f::AbstractMatrix{<:Complex},
    N::AbstractMatrix{<:Real},
    shift,
    diffphase,
)
    size(img_f) == size(N) || throw(DimensionMismatch("N must have the same size as img_f"))

    sz = [size(img_f)...]
    T = float(real(eltype(img_f)))
    N_ = zero(T)

    for i in eachindex(sz)
        shifti = ifftshift((-div(sz[i], 2)):(ceil(Int, sz[i] / 2) - 1)) * shift[i] / sz[i]
        resh = (ntuple(_ -> 1, i - 1)..., length(shifti))
        N_ = N_ .- T.(reshape(shifti, resh))
    end

    copyto!(N, N_)
    phase = cis(T(diffphase))
    twopi_im = complex(zero(T), T(2 * pi))
    phase .* (img_f .* exp.(twopi_im .* N))
end

function dftreg_resample!(
    img_f::AbstractMatrix{<:Complex},
    N::AbstractMatrix{<:Real},
    shift,
    diffphase,
)
    real(_ifft(subpix_shift!(img_f, N, shift, diffphase)))
end

function reg_stack_translate!(
    img_stack_reg::AbstractArray{<:Real,3},
    img1_f::AbstractMatrix{<:Complex},
    img2_f::AbstractMatrix{<:Complex},
    CC2x::AbstractMatrix{<:Complex},
    N::AbstractMatrix{<:Real};
    reg_param::AbstractDict{<:Integer}=Dict{Int,Any}(),
)
    _, _, size_z = size(img_stack_reg)
    CC2x .= zero(eltype(CC2x))
    N .= zero(eltype(N))

    for z = 2:size_z
        z1, z2 = z - 1, z
        img1 = view(img_stack_reg, :, :, z1)
        img2 = view(img_stack_reg, :, :, z2)
        img1_f .= _fft(img1)
        img2_f .= _fft(img2)

        if !haskey(reg_param, z)
            error, shift, diffphase = dftreg_subpix!(img1_f, img2_f, CC2x)
            reg_param[z] = (error, shift, diffphase)
        else
            error, shift, diffphase = reg_param[z]
        end

        img_stack_reg[:, :, z] .= dftreg_resample!(img2_f, N, shift, diffphase)
        CC2x .= zero(eltype(CC2x))
        N .= zero(eltype(N))
    end

    nothing
end
