using BenchmarkTools
using FFTRegGPU
using FFTW
using Random
using Printf

const DEFAULT_SEED = 20260313

struct BenchConfig
    backend::Symbol
    samples::Int
    evals::Int
    seed::Int
    sizes::Vector{Tuple{Int,Int}}
    stack_size::NTuple{3,Int}
    noise_sigma::Float32
    output::Union{Nothing,String}
end

function parse_dim2(token::AbstractString)
    m = match(r"^(\d+)x(\d+)$", lowercase(strip(token)))
    m === nothing && error("Invalid 2D size '$token'. Use NxM (e.g. 256x256).")
    parse(Int, m.captures[1]), parse(Int, m.captures[2])
end

function parse_dim3(token::AbstractString)
    m = match(r"^(\d+)x(\d+)x(\d+)$", lowercase(strip(token)))
    m === nothing && error("Invalid 3D size '$token'. Use NxMxZ (e.g. 256x256x64).")
    parse(Int, m.captures[1]), parse(Int, m.captures[2]), parse(Int, m.captures[3])
end

function parse_sizes(token::AbstractString)
    [parse_dim2(s) for s in split(token, ",")]
end

function parse_backend(token::AbstractString)
    b = Symbol(lowercase(strip(token)))
    b in (:cpu, :cuda, :both) || error("backend must be cpu, cuda, or both")
    b
end

function parse_args(args::Vector{String})
    cfg = Dict(
        :backend => :both,
        :samples => 20,
        :evals => 1,
        :seed => DEFAULT_SEED,
        :sizes => [(128, 128), (256, 256)],
        :stack_size => (256, 256, 64),
        :noise_sigma => 0.02f0,
        :output => nothing,
    )

    for arg in args
        if arg == "--help" || arg == "-h"
            println("""
Usage:
  julia --project=benchmark benchmark/run_benchmarks.jl [options]

Options:
  --backend=cpu|cuda|both      Benchmark backend(s). Default: both
  --samples=N                  BenchmarkTools samples per case. Default: 20
  --evals=N                    BenchmarkTools evals per sample. Default: 1
  --seed=N                     RNG seed for synthetic data. Default: $(DEFAULT_SEED)
  --sizes=128x128,256x256      2D pair benchmark sizes. Default: 128x128,256x256
  --stack=256x256x64           3D stack benchmark size. Default: 256x256x64
  --noise=0.02                 Added Gaussian noise sigma. Default: 0.02
  --output=PATH                Optional CSV output path.
""")
            exit(0)
        elseif startswith(arg, "--backend=")
            cfg[:backend] = parse_backend(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--samples=")
            cfg[:samples] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--evals=")
            cfg[:evals] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--seed=")
            cfg[:seed] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sizes=")
            cfg[:sizes] = parse_sizes(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--stack=")
            cfg[:stack_size] = parse_dim3(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--noise=")
            cfg[:noise_sigma] = Float32(parse(Float64, split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--output=")
            cfg[:output] = split(arg, "=", limit=2)[2]
        else
            error("Unknown option: $arg")
        end
    end

    cfg[:samples] > 0 || error("samples must be positive")
    cfg[:evals] > 0 || error("evals must be positive")

    BenchConfig(
        cfg[:backend],
        cfg[:samples],
        cfg[:evals],
        cfg[:seed],
        cfg[:sizes],
        cfg[:stack_size],
        cfg[:noise_sigma],
        cfg[:output],
    )
end

function make_base_image(nx::Int, ny::Int, rng::AbstractRNG)
    x = reshape(collect(LinRange(0f0, 1f0, nx)), nx, 1)
    y = reshape(collect(LinRange(0f0, 1f0, ny)), 1, ny)
    img = @. 0.55f0 * sin(2f0 * π * (3f0 * x + 5f0 * y)) + 0.35f0 * cos(2f0 * π * (7f0 * x - 2f0 * y))
    img .+= 0.10f0 .* randn(rng, Float32, nx, ny)
    img .-= minimum(img)
    img ./= maximum(img)
    img
end

function make_integer_pair(nx::Int, ny::Int, rng::AbstractRNG, noise_sigma::Float32)
    img1 = make_base_image(nx, ny, rng)
    img2 = circshift(img1, (3, -2))
    img2 .+= noise_sigma .* randn(rng, Float32, nx, ny)
    fft(img1), fft(img2)
end

function make_subpixel_pair(nx::Int, ny::Int, rng::AbstractRNG, noise_sigma::Float32)
    img1 = make_base_image(nx, ny, rng)
    img1_f = fft(img1)

    shift = Float32[2.25, -1.75]
    nbuf = zeros(Float32, nx, ny)
    shifted_f = similar(img1_f)
    subpix_shift!(shifted_f, img1_f, nbuf, shift, 0f0)
    img2 = real(ifft(shifted_f))
    img2 .+= noise_sigma .* randn(rng, Float32, nx, ny)
    img1_f, fft(img2)
end

function make_stack(nx::Int, ny::Int, nz::Int, rng::AbstractRNG, noise_sigma::Float32)
    stack = zeros(Float32, nx, ny, nz)
    stack[:, :, 1] .= make_base_image(nx, ny, rng)

    for z in 2:nz
        stack[:, :, z] .= circshift(@view(stack[:, :, z - 1]), (1, -1))
    end
    stack .+= noise_sigma .* randn(rng, Float32, nx, ny, nz)
    stack
end

function push_summary!(rows, backend::String, case_name::String, dims::String, cfg::BenchConfig, trial::BenchmarkTools.Trial)
    med = BenchmarkTools.median(trial)
    mn = BenchmarkTools.mean(trial)
    mnm = BenchmarkTools.minimum(trial)
    push!(
        rows,
        (
            backend = backend,
            case = case_name,
            dims = dims,
            median_ms = med.time / 1e6,
            mean_ms = mn.time / 1e6,
            min_ms = mnm.time / 1e6,
            memory_bytes = med.memory,
            allocs = med.allocs,
            samples = cfg.samples,
            evals = cfg.evals,
        ),
    )
end

function run_cpu_benchmarks(cfg::BenchConfig, rows)
    println("Running CPU benchmarks...")

    for (i, (nx, ny)) in enumerate(cfg.sizes)
        rng = MersenneTwister(cfg.seed + i)
        dims = "$(nx)x$(ny)"

        img1_f, img2_f = make_integer_pair(nx, ny, rng, cfg.noise_sigma)
        cc = similar(img1_f)
        cc_abs2_work = similar(cc, float(real(eltype(cc))))
        dftreg!(img1_f, img2_f, cc; cc_abs2_work=cc_abs2_work) # warmup
        trial = run(
            @benchmarkable dftreg!($img1_f, $img2_f, $cc; cc_abs2_work=$cc_abs2_work) samples=cfg.samples evals=cfg.evals
        )
        push_summary!(rows, "cpu", "dftreg", dims, cfg, trial)

        img1_f_sub, img2_f_sub = make_subpixel_pair(nx, ny, rng, cfg.noise_sigma)
        cc2x = zeros(eltype(img1_f_sub), 2 * nx, 2 * ny)
        cc2x_abs2_work = similar(cc2x, float(real(eltype(cc2x))))
        dftreg_subpix!(img1_f_sub, img2_f_sub, cc2x; cc2x_abs2_work=cc2x_abs2_work) # warmup
        trial = run(
            @benchmarkable dftreg_subpix!($img1_f_sub, $img2_f_sub, $cc2x; cc2x_abs2_work=$cc2x_abs2_work) samples=cfg.samples evals=cfg.evals
        )
        push_summary!(rows, "cpu", "dftreg_subpix", dims, cfg, trial)
    end

    nx, ny, nz = cfg.stack_size
    rng = MersenneTwister(cfg.seed + 100)
    dims = "$(nx)x$(ny)x$(nz)"
    stack_template = make_stack(nx, ny, nz, rng, cfg.noise_sigma)
    stack_work = similar(stack_template)
    img1_f = zeros(ComplexF32, nx, ny)
    img2_f = zeros(ComplexF32, nx, ny)
    cc2x = zeros(ComplexF32, 2 * nx, 2 * ny)
    nbuf = zeros(Float32, nx, ny)
    reg_param = Dict{Int,Any}()

    copyto!(stack_work, stack_template)
    reg_stack_translate!(stack_work, img1_f, img2_f, cc2x, nbuf; reg_param=reg_param) # warmup

    trial = run(
        @benchmarkable begin
            copyto!($stack_work, $stack_template)
            empty!($reg_param)
            reg_stack_translate!($stack_work, $img1_f, $img2_f, $cc2x, $nbuf; reg_param=$reg_param)
        end samples=cfg.samples evals=1
    )
    push_summary!(rows, "cpu", "reg_stack_translate", dims, cfg, trial)
end

function maybe_load_cuda()
    Base.find_package("CUDA") === nothing && return nothing
    try
        @eval using CUDA
        # `@eval using CUDA` inside a function can create a newer-world binding.
        cuda = Base.invokelatest(() -> getfield(@__MODULE__, :CUDA))
        Base.invokelatest(cuda.functional) || return nothing
        cuda
    catch
        nothing
    end
end

function run_cuda_benchmarks(cfg::BenchConfig, rows)
    CUDA = maybe_load_cuda()
    if CUDA === nothing
        println("Skipping CUDA benchmarks (CUDA not installed or no functional CUDA device).")
        return
    end
    println("Running CUDA benchmarks...")
    Base.invokelatest(_run_cuda_benchmarks_loaded, CUDA, cfg, rows)
end

function _run_cuda_benchmarks_loaded(CUDA, cfg::BenchConfig, rows)
    for (i, (nx, ny)) in enumerate(cfg.sizes)
        rng = MersenneTwister(cfg.seed + i)
        dims = "$(nx)x$(ny)"

        img1_f_cpu, img2_f_cpu = make_integer_pair(nx, ny, rng, cfg.noise_sigma)
        img1_f = CUDA.CuArray(img1_f_cpu)
        img2_f = CUDA.CuArray(img2_f_cpu)
        cc = similar(img1_f)
        cc_abs2_work = similar(cc, float(real(eltype(cc))))

        dftreg!(img1_f, img2_f, cc; cc_abs2_work=cc_abs2_work) # warmup
        CUDA.synchronize()
        trial = run(
            @benchmarkable begin
                dftreg!($img1_f, $img2_f, $cc; cc_abs2_work=$cc_abs2_work)
                CUDA.synchronize()
            end samples=cfg.samples evals=cfg.evals
        )
        push_summary!(rows, "cuda", "dftreg", dims, cfg, trial)

        img1_f_sub_cpu, img2_f_sub_cpu = make_subpixel_pair(nx, ny, rng, cfg.noise_sigma)
        img1_f_sub = CUDA.CuArray(img1_f_sub_cpu)
        img2_f_sub = CUDA.CuArray(img2_f_sub_cpu)
        cc2x = CUDA.zeros(eltype(img1_f_sub), 2 * nx, 2 * ny)
        cc2x_abs2_work = similar(cc2x, float(real(eltype(cc2x))))

        dftreg_subpix!(img1_f_sub, img2_f_sub, cc2x; cc2x_abs2_work=cc2x_abs2_work) # warmup
        CUDA.synchronize()
        trial = run(
            @benchmarkable begin
                dftreg_subpix!($img1_f_sub, $img2_f_sub, $cc2x; cc2x_abs2_work=$cc2x_abs2_work)
                CUDA.synchronize()
            end samples=cfg.samples evals=cfg.evals
        )
        push_summary!(rows, "cuda", "dftreg_subpix", dims, cfg, trial)
    end

    nx, ny, nz = cfg.stack_size
    rng = MersenneTwister(cfg.seed + 100)
    dims = "$(nx)x$(ny)x$(nz)"
    stack_template = CUDA.CuArray(make_stack(nx, ny, nz, rng, cfg.noise_sigma))
    stack_work = similar(stack_template)
    img1_f = CUDA.zeros(ComplexF32, nx, ny)
    img2_f = CUDA.zeros(ComplexF32, nx, ny)
    cc2x = CUDA.zeros(ComplexF32, 2 * nx, 2 * ny)
    nbuf = CUDA.zeros(Float32, nx, ny)
    reg_param = Dict{Int,Any}()

    copyto!(stack_work, stack_template)
    reg_stack_translate!(stack_work, img1_f, img2_f, cc2x, nbuf; reg_param=reg_param) # warmup
    CUDA.synchronize()

    trial = run(
        @benchmarkable begin
            copyto!($stack_work, $stack_template)
            empty!($reg_param)
            reg_stack_translate!($stack_work, $img1_f, $img2_f, $cc2x, $nbuf; reg_param=$reg_param)
            CUDA.synchronize()
        end samples=cfg.samples evals=1
    )
    push_summary!(rows, "cuda", "reg_stack_translate", dims, cfg, trial)
end

function print_rows(rows)
    println()
    println("Benchmark results (times in ms):")
    println(rpad("backend", 8), rpad("case", 20), rpad("dims", 14), lpad("median", 12), lpad("mean", 12), lpad("min", 12), lpad("memory", 12), lpad("allocs", 10))
    for r in rows
        @printf(
            "%-8s%-20s%-14s%12.3f%12.3f%12.3f%12d%10d\n",
            r.backend,
            r.case,
            r.dims,
            r.median_ms,
            r.mean_ms,
            r.min_ms,
            r.memory_bytes,
            r.allocs,
        )
    end
end

function write_csv(path::String, rows, cfg::BenchConfig)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "backend,case,dims,median_ms,mean_ms,min_ms,memory_bytes,allocs,samples,evals,seed")
        for r in rows
            @printf(
                io,
                "%s,%s,%s,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d\n",
                r.backend,
                r.case,
                r.dims,
                r.median_ms,
                r.mean_ms,
                r.min_ms,
                r.memory_bytes,
                r.allocs,
                cfg.samples,
                cfg.evals,
                cfg.seed,
            )
        end
    end
end

function main()
    cfg = parse_args(ARGS)
    rows = Vector{NamedTuple}()

    println("FFTRegGPU benchmark configuration:")
    println("  backend:   ", cfg.backend)
    println("  samples:   ", cfg.samples)
    println("  evals:     ", cfg.evals)
    println("  seed:      ", cfg.seed)
    println("  sizes:     ", join(["$(x)x$(y)" for (x, y) in cfg.sizes], ", "))
    println("  stack:     ", "$(cfg.stack_size[1])x$(cfg.stack_size[2])x$(cfg.stack_size[3])")
    println("  noise:     ", cfg.noise_sigma)

    if cfg.backend in (:cpu, :both)
        run_cpu_benchmarks(cfg, rows)
    end
    if cfg.backend in (:cuda, :both)
        run_cuda_benchmarks(cfg, rows)
    end

    print_rows(rows)
    if cfg.output !== nothing
        write_csv(cfg.output, rows, cfg)
        println()
        println("Wrote CSV: ", cfg.output)
    end
end

main()
