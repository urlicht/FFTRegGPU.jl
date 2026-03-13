using Random
using Test

function make_blob_image(nx::Int, ny::Int, rng::AbstractRNG; nspots::Int=10, sigma::Float32=2.8f0)
    x = reshape(Float32.(0:(nx - 1)), nx, 1)
    y = reshape(Float32.(0:(ny - 1)), 1, ny)
    img = zeros(Float32, nx, ny)

    two_sigma2 = 2f0 * sigma^2
    for _ in 1:nspots
        cx = rand(rng, Float32) * Float32(nx - 1)
        cy = rand(rng, Float32) * Float32(ny - 1)
        amp = 0.2f0 + 0.8f0 * rand(rng, Float32)
        img .+= amp .* exp.(-((x .- cx).^2 .+ (y .- cy).^2) ./ two_sigma2)
    end

    img ./= maximum(img)
    img
end

function stack_adjacent_mse(stack::AbstractArray{<:Real,3})
    _, _, nz = size(stack)
    nz <= 1 && return 0.0
    acc = 0.0
    for z in 2:nz
        a = @view stack[:, :, z - 1]
        b = @view stack[:, :, z]
        acc += sum((a .- b) .^ 2) / length(a)
    end
    acc / (nz - 1)
end

@testset "synthetic noisy stack registration" begin
    rng = MersenneTwister(20260313)
    nx, ny = 64, 64
    known_rel = [(1, -1), (2, 1), (-1, 2), (0, -2), (1, 1)]
    nz = length(known_rel) + 1

    base = make_blob_image(nx, ny, rng)
    clean_stack = zeros(Float32, nx, ny, nz)
    clean_stack[:, :, 1] .= base
    for z in 2:nz
        clean_stack[:, :, z] .= circshift(@view(clean_stack[:, :, z - 1]), known_rel[z - 1])
    end

    noise_sigma = 0.03f0
    noisy_stack = clean_stack .+ noise_sigma .* randn(rng, Float32, nx, ny, nz)
    reg_stack = copy(noisy_stack)

    img1_f = zeros(ComplexF32, nx, ny)
    img2_f = zeros(ComplexF32, nx, ny)
    cc2x = zeros(ComplexF32, 2 * nx, 2 * ny)
    nbuf = zeros(Float32, nx, ny)
    reg_param = Dict{Int,Any}()

    mse_before = stack_adjacent_mse(noisy_stack)
    reg_stack_translate!(reg_stack, img1_f, img2_f, cc2x, nbuf; reg_param=reg_param)
    mse_after = stack_adjacent_mse(reg_stack)

    @test length(reg_param) == nz - 1
    cum_shift = zeros(Float32, 2)
    for z in 2:nz
        cum_shift .+= Float32.(collect(known_rel[z - 1]))
        _, shift, _ = reg_param[z]
        expected = -cum_shift
        @test length(shift) == 2
        @test all(isapprox.(shift, expected; atol=0.35f0))
    end

    @test mse_after < 0.75 * mse_before
end
