_fft(inp::AbstractArray) = fft(inp)
_ifft(inp::AbstractArray) = ifft(inp)
_ifft!(inp::AbstractArray) = copyto!(inp, _ifft(inp))
_fftshift(inp::AbstractArray) = fftshift(inp)
_ifftshift(inp::AbstractArray) = ifftshift(inp)
_scalar_at(inp::AbstractArray, idx) = inp[idx]

_backend_template(inp::AbstractArray) = inp
_backend_template(inp::SubArray) = _backend_template(parent(inp))
_backend_template(inp::Base.ReshapedArray) = _backend_template(parent(inp))
_backend_template(inp::Base.PermutedDimsArray) = _backend_template(parent(inp))

function _to_backend(ref::AbstractArray, src::AbstractArray)
    tmpl = _backend_template(ref)
    out = similar(tmpl, eltype(src), size(src))
    copyto!(out, src)
    out
end

function dftups(inp::AbstractArray{T,N}, no::Integer, usfac::Int=1, offset=nothing) where {T<:Number,N}
    no > 0 || throw(ArgumentError("no must be positive"))
    usfac > 0 || throw(ArgumentError("usfac must be positive"))

    Treal = float(real(T))
    offset_vals = offset === nothing ? ntuple(_ -> zero(Treal), N) : ntuple(i -> Treal(offset[i]), N)

    sz = ntuple(i -> size(inp, i), N)
    permV = 1:N
    out = inp
    for i in permV
        out = permutedims(out, [i; deleteat!(collect(permV), i)])

        row = Treal.(0:(no - 1)) .- offset_vals[i]
        col = Treal.(ifftshift(0:(sz[i] - 1)) .- div(sz[i], 2))
        phase = -complex(zero(Treal), Treal(2 * pi / (sz[i] * usfac)))
        kern_host = exp.(phase .* (row * transpose(col)))
        kern = _to_backend(out, convert.(eltype(out), kern_host))

        d = size(out)[2:N]
        out = kern * reshape(out, Val(2))
        out = reshape(out, (no, d...))
    end
    permutedims(out, collect(ndims(out):-1:1))
end

function dftreg!(
    img1_f::AbstractMatrix{T1},
    img2_f::AbstractMatrix{T2},
    CC::AbstractMatrix{TC},
) where {T1<:Complex,T2<:Complex,TC<:Complex}
    size(img1_f) == size(img2_f) || throw(DimensionMismatch("img1_f and img2_f must have the same size"))
    size(img1_f) == size(CC) || throw(DimensionMismatch("CC must have the same size as img1_f"))

    Treal = float(real(promote_type(T1, T2, TC)))
    L = Treal(length(img1_f))

    @. CC = img1_f * conj(img2_f)
    _ifft!(CC)

    _, loc = findmax(abs, CC)
    CCmax = _scalar_at(CC, loc)
    rfzero = sum(abs2, img1_f) / L
    rgzero = sum(abs2, img2_f) / L
    error = abs(one(Treal) - CCmax * conj(CCmax) / (rgzero * rfzero))
    diffphase = Treal(atan(imag(CCmax), real(CCmax)))

    indi = size(img1_f)
    ind2 = ntuple(i -> div(indi[i], 2), length(indi))
    locI = Tuple(loc)
    shift = zeros(Treal, length(locI))

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
    img1_f::AbstractMatrix{T1},
    img2_f::AbstractMatrix{T2},
    CC2x::AbstractMatrix{TC},
    up_fac::Int=10,
) where {T1<:Complex,T2<:Complex,TC<:Complex}
    size(img1_f) == size(img2_f) || throw(DimensionMismatch("img1_f and img2_f must have the same size"))
    up_fac > 0 || throw(ArgumentError("up_fac must be positive"))
    Treal = float(real(promote_type(T1, T2, TC)))

    # initial estimate by 2x upsample
    dim_input = size(img1_f)
    ranges = [(x + 1 - div(x, 2)):(x + 1 + div(x - 1, 2)) for x in dim_input]
    expected_cc_size = ntuple(i -> 2 * dim_input[i], length(dim_input))
    size(CC2x) == expected_cc_size ||
        throw(DimensionMismatch("CC2x must have size $(expected_cc_size), got $(size(CC2x))"))

    CC2x .= zero(eltype(CC2x))
    img1_shift = _fftshift(img1_f)
    img2_shift = _fftshift(img2_f)
    @views @. CC2x[ranges...] = img1_shift * conj(img2_shift)

    # compute cross-correlation and locate the peak
    copyto!(CC2x, _ifftshift(CC2x))
    _ifft!(CC2x)
    _, loc = findmax(abs, CC2x)

    indi = size(CC2x)
    locI = collect(Tuple(loc))
    CC2x_max = _scalar_at(CC2x, loc)

    # obtain shift in original pixel grid
    ind2 = Treal.(indi) ./ Treal(2)
    shift = zeros(Treal, length(locI))
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
        up = Treal(up_fac)
        shift = round.(Treal, shift .* up) ./ up # initial shift estimate
        dft_shift = Treal(ceil(up_fac * 1.5) / 2) # center of output at dft_shift + 1
        denom = prod(ind2) * up ^ 2

        CC_refine = dftups(
            img2_f .* conj.(img1_f),
            ceil(Int, up_fac * 1.5),
            up_fac,
            dft_shift .- shift .* up,
        ) / denom

        _, loc = findmax(abs, CC_refine)
        locI_ref = Treal.(Tuple(loc))
        CC_refine_max = _scalar_at(CC_refine, loc)
        shift = shift .+ (locI_ref .- dft_shift .- one(Treal)) ./ up

        img1_00 = _scalar_at(dftups(img1_f .* conj.(img1_f), 1, up_fac), 1) / denom
        img2_00 = _scalar_at(dftups(img2_f .* conj.(img2_f), 1, up_fac), 1) / denom
        CC_max = CC_refine_max
    else
        img1_00 = sum(img1_f .* conj.(img1_f)) / prod(indi)
        img2_00 = sum(img2_f .* conj.(img2_f)) / prod(indi)
        CC_max = CC2x_max
    end

    error = 1 - CC_max * conj(CC_max) / (img1_00 * img2_00)
    error = sqrt(abs(error)) |> Treal
    diffphase = Treal(atan(imag(CC_max), real(CC_max)))

    error, shift, diffphase
end

function subpix_shift!(
    out::AbstractMatrix{<:Complex},
    img_f::AbstractMatrix{<:Complex},
    N::AbstractMatrix{<:Real},
    shift,
    diffphase,
)
    size(out) == size(img_f) || throw(DimensionMismatch("out must have the same size as img_f"))
    size(img_f) == size(N) || throw(DimensionMismatch("N must have the same size as img_f"))
    length(shift) == ndims(img_f) || throw(DimensionMismatch("shift must match image dimensionality"))

    sz = [size(img_f)...]
    Tphase = float(real(eltype(out)))
    TN = eltype(N)
    fill!(N, zero(TN))

    for i in eachindex(sz)
        shifti = ifftshift((-div(sz[i], 2)):(ceil(Int, sz[i] / 2) - 1)) * shift[i] / sz[i]
        resh = (ntuple(_ -> 1, i - 1)..., length(shifti))
        shifti_backend = _to_backend(N, reshape(TN.(shifti), resh))
        N .-= shifti_backend
    end

    phase = cis(Tphase(diffphase))
    twopi_im = complex(zero(Tphase), Tphase(2 * pi))
    @. out = phase * (img_f * exp(twopi_im * N))
    out
end

function subpix_shift!(
    img_f::AbstractMatrix{<:Complex},
    N::AbstractMatrix{<:Real},
    shift,
    diffphase,
)
    out = similar(img_f)
    subpix_shift!(out, img_f, N, shift, diffphase)
end

function dftreg_resample!(
    out::AbstractMatrix{<:Real},
    img_f::AbstractMatrix{<:Complex},
    N::AbstractMatrix{<:Real},
    shift,
    diffphase,
    ;
    work_f::AbstractMatrix{<:Complex}=similar(img_f),
)
    size(out) == size(img_f) || throw(DimensionMismatch("out must have the same size as img_f"))
    size(work_f) == size(img_f) || throw(DimensionMismatch("work_f must have the same size as img_f"))

    subpix_shift!(work_f, img_f, N, shift, diffphase)
    _ifft!(work_f)
    out .= real.(work_f)
    out
end

function dftreg_resample(
    img_f::AbstractMatrix{<:Complex},
    N::AbstractMatrix{<:Real},
    shift,
    diffphase,
    ;
    work_f::AbstractMatrix{<:Complex}=similar(img_f),
)
    out = similar(N, float(real(eltype(img_f))), size(img_f))
    dftreg_resample!(out, img_f, N, shift, diffphase; work_f=work_f)
end

function dftreg_resample!(
    img_f::AbstractMatrix{<:Complex},
    N::AbstractMatrix{<:Real},
    shift,
    diffphase,
)
    dftreg_resample(img_f, N, shift, diffphase)
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

        dftreg_resample!(view(img_stack_reg, :, :, z), img2_f, N, shift, diffphase; work_f=img2_f)
        CC2x .= zero(eltype(CC2x))
        N .= zero(eltype(N))
    end

    nothing
end
