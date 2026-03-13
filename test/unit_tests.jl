using Random
using Test

@testset "FFTRegGPU unit tests" begin
    rng = MersenneTwister(42)
    nx, ny = 32, 24

    img = rand(rng, Float32, nx, ny)
    img_f = fft(img)

    @testset "argument validation" begin
        bad_cc = zeros(ComplexF32, nx + 1, ny)
        @test_throws DimensionMismatch dftreg!(img_f, img_f, bad_cc)

        cc2x = zeros(ComplexF32, 2 * nx, 2 * ny)
        @test_throws ArgumentError dftreg_subpix!(img_f, img_f, cc2x, 0)

        bad_cc2x = zeros(ComplexF32, 2 * nx + 1, 2 * ny)
        @test_throws DimensionMismatch dftreg_subpix!(img_f, img_f, bad_cc2x, 10)

        @test_throws ArgumentError FFTRegGPU.dftups(img_f, 0, 1)
        @test_throws ArgumentError FFTRegGPU.dftups(img_f, 4, 0)

        out_bad = zeros(Float32, nx + 1, ny)
        nbuf = zeros(Float32, nx, ny)
        @test_throws DimensionMismatch dftreg_resample!(out_bad, img_f, nbuf, (0f0, 0f0), 0f0)
    end

    @testset "identity registration" begin
        cc = similar(img_f)
        err, shift, phase = dftreg!(img_f, img_f, cc)

        @test err ≤ 1f-5
        @test isapprox(phase, 0f0; atol=1f-5)
        @test eltype(shift) == Float32
        @test all(isapprox.(shift, zero(eltype(shift)); atol=1f-5))
    end

    @testset "integer shift recovery" begin
        shifted = circshift(img, (3, -2))
        shifted_f = fft(shifted)
        cc = similar(img_f)
        _, shift, _ = dftreg!(img_f, shifted_f, cc)
        @test all(isapprox.(shift, Float32[-3, 2]; atol=1f-5))
    end

    @testset "resampling API" begin
        shift = Float32[2.0, -1.0]
        phase = 0f0
        nbuf1 = zeros(Float32, nx, ny)
        nbuf2 = zeros(Float32, nx, ny)

        moved = circshift(img, (2, -1))
        moved_f = fft(moved)

        alloc_out = dftreg_resample(moved_f, nbuf1, shift, phase)
        inpl_out = similar(img)
        work_f = similar(moved_f)
        ret = dftreg_resample!(inpl_out, moved_f, nbuf2, shift, phase; work_f=work_f)

        @test ret === inpl_out
        @test alloc_out ≈ inpl_out atol = 1f-5

        phase_shifted = similar(moved_f)
        ret2 = subpix_shift!(phase_shifted, moved_f, nbuf2, (0f0, 0f0), 0f0)
        @test ret2 === phase_shifted
        @test phase_shifted ≈ moved_f atol = 1f-6
    end
end
