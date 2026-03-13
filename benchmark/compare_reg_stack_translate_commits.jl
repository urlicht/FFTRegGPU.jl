using Printf
using Serialization

const DEFAULT_SEED = 20260313

struct CompareConfig
    repo_path::String
    current_ref::String
    previous_ref::String
    stack_sizes::Vector{NTuple{3,Int}}
    samples::Int
    evals::Int
    seed::Int
    noise_sigma::Float32
end

function parse_dim3(token::AbstractString)
    m = match(r"^(\d+)x(\d+)x(\d+)$", lowercase(strip(token)))
    m === nothing && error("Invalid 3D size '$token'. Use NxMxZ (e.g. 256x256x64).")
    parse(Int, m.captures[1]), parse(Int, m.captures[2]), parse(Int, m.captures[3])
end

function parse_stack_sizes(token::AbstractString)
    [parse_dim3(s) for s in split(token, ",")]
end

function parse_args(args::Vector{String})
    cfg = Dict(
        :repo_path => abspath(joinpath(@__DIR__, "..")),
        :current_ref => "HEAD",
        :previous_ref => "HEAD~1",
        :stack_sizes => [(256, 256, 64), (256, 256, 256)],
        :samples => 20,
        :evals => 1,
        :seed => DEFAULT_SEED,
        :noise_sigma => 0.02f0,
    )

    for arg in args
        if arg == "--help" || arg == "-h"
            println("""
Usage:
  julia --project=benchmark benchmark/compare_reg_stack_translate_commits.jl [options]

Options:
  --repo=PATH                     Path to FFTRegGPU repository. Default: repo root
  --current-ref=REF               Current/reference A git ref. Default: HEAD
  --previous-ref=REF              Baseline/reference B git ref. Default: HEAD~1
  --stack=256x256x64,256x256x256  Stack sizes to benchmark. Default: 256x256x64,256x256x256
  --samples=N                     BenchmarkTools samples per case. Default: 20
  --evals=N                       BenchmarkTools evals per sample. Default: 1
  --seed=N                        RNG seed for synthetic data. Default: $(DEFAULT_SEED)
  --noise=0.02                    Added Gaussian noise sigma. Default: 0.02
""")
            exit(0)
        elseif startswith(arg, "--repo=")
            cfg[:repo_path] = abspath(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--current-ref=")
            cfg[:current_ref] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--previous-ref=")
            cfg[:previous_ref] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--stack=")
            cfg[:stack_sizes] = parse_stack_sizes(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--samples=")
            cfg[:samples] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--evals=")
            cfg[:evals] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--seed=")
            cfg[:seed] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--noise=")
            cfg[:noise_sigma] = Float32(parse(Float64, split(arg, "=", limit=2)[2]))
        else
            error("Unknown option: $arg")
        end
    end

    isdir(cfg[:repo_path]) || error("Repo path does not exist: $(cfg[:repo_path])")
    cfg[:samples] > 0 || error("samples must be positive")
    cfg[:evals] > 0 || error("evals must be positive")
    isempty(cfg[:current_ref]) && error("current-ref must not be empty")
    isempty(cfg[:previous_ref]) && error("previous-ref must not be empty")
    isempty(cfg[:stack_sizes]) && error("stack must include at least one size")

    CompareConfig(
        cfg[:repo_path],
        cfg[:current_ref],
        cfg[:previous_ref],
        cfg[:stack_sizes],
        cfg[:samples],
        cfg[:evals],
        cfg[:seed],
        cfg[:noise_sigma],
    )
end

const CHILD_CODE = raw"""
using Pkg
using Random
using Serialization

repo_checkout = ARGS[1]
result_file = ARGS[2]
stack_arg = ARGS[3]
samples = parse(Int, ARGS[4])
evals = parse(Int, ARGS[5])
seed = parse(Int, ARGS[6])
noise_sigma = Float32(parse(Float64, ARGS[7]))

function parse_stack_sizes(token::AbstractString)
    out = NTuple{3,Int}[]
    for s in split(token, ';')
        isempty(s) && continue
        p = split(s, 'x')
        length(p) == 3 || error("Invalid stack size: $s (expected NxMxZ)")
        push!(out, (parse(Int, p[1]), parse(Int, p[2]), parse(Int, p[3])))
    end
    out
end

stack_sizes = parse_stack_sizes(stack_arg)
isempty(stack_sizes) && error("No stack sizes provided")

env_dir = mktempdir()
Pkg.activate(env_dir)
Pkg.develop(PackageSpec(path=repo_checkout))
Pkg.instantiate()
Pkg.add(["BenchmarkTools", "FFTW", "CUDA"])

using BenchmarkTools
using FFTRegGPU
using FFTW
using CUDA

CUDA.functional() || error("CUDA is not functional on this machine")

_safe_str(f, default="unknown") = try
    string(f())
catch
    default
end

function _cuda_info()
    dev = CUDA.device()
    (
        functional = true,
        device_name = _safe_str(() -> CUDA.name(dev), string(dev)),
        device_index = _safe_str(() -> CUDA.deviceid(dev)),
        driver_version = _safe_str(CUDA.driver_version),
        runtime_version = _safe_str(CUDA.runtime_version),
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

function make_stack(nx::Int, ny::Int, nz::Int, rng::AbstractRNG, noise_sigma::Float32)
    stack = zeros(Float32, nx, ny, nz)
    stack[:, :, 1] .= make_base_image(nx, ny, rng)
    for z in 2:nz
        stack[:, :, z] .= circshift(@view(stack[:, :, z - 1]), (1, -1))
    end
    stack .+= noise_sigma .* randn(rng, Float32, nx, ny, nz)
    stack
end

function _resolve_reg_fn()
    if isdefined(FFTRegGPU, :reg_stack_translate_gpu!)
        return getfield(FFTRegGPU, :reg_stack_translate_gpu!)
    elseif isdefined(FFTRegGPU, :reg_stack_translate!)
        return getfield(FFTRegGPU, :reg_stack_translate!)
    else
        error("Neither reg_stack_translate_gpu! nor reg_stack_translate! exists in this commit")
    end
end

function call_reg_stack_translate!(stack_work, img1_f, img2_f, cc2x, nbuf, reg_param)
    f = _resolve_reg_fn()
    try
        return f(stack_work, img1_f, img2_f, cc2x, nbuf; reg_param=reg_param)
    catch err
        if err isa MethodError
            return f(stack_work, img1_f, img2_f, cc2x, nbuf)
        end
        rethrow(err)
    end
end

function run_case_rows(stack_sizes, samples, evals, seed, noise_sigma)
    rows = NamedTuple[]
    all_buffers_on_cuda = true
    for (i, (nx, ny, nz)) in enumerate(stack_sizes)
        rng = MersenneTwister(seed + 100 + i)
        dims = "$(nx)x$(ny)x$(nz)"

        stack_template = CUDA.CuArray(make_stack(nx, ny, nz, rng, noise_sigma))
        stack_work = similar(stack_template)
        img1_f = CUDA.zeros(ComplexF32, nx, ny)
        img2_f = CUDA.zeros(ComplexF32, nx, ny)
        cc2x = CUDA.zeros(ComplexF32, 2 * nx, 2 * ny)
        nbuf = CUDA.zeros(Float32, nx, ny)
        reg_param = Dict{Int,Any}()
        all_buffers_on_cuda = all_buffers_on_cuda && (
            stack_template isa CUDA.CuArray &&
            stack_work isa CUDA.CuArray &&
            img1_f isa CUDA.CuArray &&
            img2_f isa CUDA.CuArray &&
            cc2x isa CUDA.CuArray &&
            nbuf isa CUDA.CuArray
        )

        copyto!(stack_work, stack_template)
        call_reg_stack_translate!(stack_work, img1_f, img2_f, cc2x, nbuf, reg_param)
        CUDA.synchronize()
        stack_out = Array(stack_work)

        trial = run(@benchmarkable begin
            copyto!($stack_work, $stack_template)
            empty!($reg_param)
            call_reg_stack_translate!($stack_work, $img1_f, $img2_f, $cc2x, $nbuf, $reg_param)
            CUDA.synchronize()
        end samples=samples evals=evals)

        med = BenchmarkTools.median(trial)
        mn = BenchmarkTools.mean(trial)
        mnm = BenchmarkTools.minimum(trial)
        push!(rows, (
            dims = dims,
            median_ms = med.time / 1e6,
            mean_ms = mn.time / 1e6,
            min_ms = mnm.time / 1e6,
            memory_bytes = med.memory,
            allocs = med.allocs,
            stack_out = stack_out,
        ))
    end

    return rows, all_buffers_on_cuda
end

rows, all_buffers_on_cuda = run_case_rows(stack_sizes, samples, evals, seed, noise_sigma)

serialize(
    result_file,
    (
        rows = rows,
        cuda_info = _cuda_info(),
        all_buffers_on_cuda = all_buffers_on_cuda,
    ),
)
"""

function write_child_script()
    dir = mktempdir()
    path = joinpath(dir, "bench_compare_one_ref.jl")
    write(path, CHILD_CODE)
    path
end

function checkout_ref(repo_path::AbstractString, ref::AbstractString)
    dir = mktempdir()
    run(`git clone --quiet $(String(repo_path)) $dir`)
    run(`git -C $dir checkout --quiet $(String(ref))`)
    dir
end

function run_ref(
    repo_path::AbstractString,
    ref::AbstractString,
    child_file::AbstractString,
    stack_sizes::Vector{NTuple{3,Int}},
    samples::Int,
    evals::Int,
    seed::Int,
    noise_sigma::Float32,
)
    checkout = checkout_ref(repo_path, ref)
    result_dir = mktempdir()
    result_file = joinpath(result_dir, "result.bin")
    stack_arg = join(["$(x)x$(y)x$(z)" for (x, y, z) in stack_sizes], ";")

    cmd = `$(Base.julia_cmd()) $(String(child_file)) $checkout $result_file $stack_arg $samples $evals $seed $noise_sigma`

    try
        run(cmd)
        commit = chomp(read(`git -C $checkout rev-parse --short HEAD`, String))
        payload = deserialize(result_file)
        if payload isa NamedTuple && hasproperty(payload, :rows)
            return (
                commit = commit,
                rows = payload.rows,
                cuda_info = hasproperty(payload, :cuda_info) ? payload.cuda_info : (functional = true,),
                all_buffers_on_cuda = hasproperty(payload, :all_buffers_on_cuda) ? payload.all_buffers_on_cuda : false,
            )
        end

        # Backward compatibility with older payload shape.
        return (
            commit = commit,
            rows = payload,
            cuda_info = (functional = true,),
            all_buffers_on_cuda = false,
        )
    finally
        rm(result_dir; force=true, recursive=true)
        rm(checkout; force=true, recursive=true)
    end
end

function print_cuda_report(label::String, run_result)
    info = run_result.cuda_info
    device_name = hasproperty(info, :device_name) ? info.device_name : "unknown"
    device_index = hasproperty(info, :device_index) ? info.device_index : "unknown"
    driver = hasproperty(info, :driver_version) ? info.driver_version : "unknown"
    runtime = hasproperty(info, :runtime_version) ? info.runtime_version : "unknown"
    functional = hasproperty(info, :functional) ? info.functional : "unknown"
    arrays_on_cuda = run_result.all_buffers_on_cuda ? "yes" : "no"
    pad = repeat(" ", length(label))
    println("  ", label, ": functional=", functional, ", device=", device_name, " (id=", device_index, ")")
    println("  ", pad, "  driver=", driver, ", runtime=", runtime, ", arrays_on_cuda=", arrays_on_cuda)
end

function print_table(curr, prev, current_ref::String, previous_ref::String)
    curr_by_dims = Dict(r.dims => r for r in curr.rows)
    prev_by_dims = Dict(r.dims => r for r in prev.rows)
    dims_all = sort!(collect(intersect(keys(curr_by_dims), keys(prev_by_dims))))
    isempty(dims_all) && error("No overlapping stack dimensions between compared refs.")

    println("CUDA reg_stack_translate! comparison")
    println("current:  $(curr.commit) ($current_ref)")
    println("previous: $(prev.commit) ($previous_ref)")
    println("CUDA verification:")
    print_cuda_report("current", curr)
    print_cuda_report("previous", prev)
    println()

    @printf("%-14s%14s%14s%12s%12s\n", "dims", "current_ms", "prev_ms", "curr/prev", "delta_%")
    for d in dims_all
        c = curr_by_dims[d]
        p = prev_by_dims[d]
        ratio = c.median_ms / p.median_ms
        delta = (c.median_ms - p.median_ms) / p.median_ms * 100
        @printf("%-14s%14.3f%14.3f%12.3f%12.1f\n", d, c.median_ms, p.median_ms, ratio, delta)
    end

    println()
    println("Output equivalence (isapprox):")
    @printf("%-14s%8s%12s\n", "dims", "same", "max_abs")
    for d in dims_all
        a = curr_by_dims[d].stack_out
        b = prev_by_dims[d].stack_out
        max_abs = maximum(abs.(a .- b))
        same = isapprox(a, b; rtol=1f-4, atol=1f-4)
        @printf("%-14s%8s%12.3e\n", d, same ? "yes" : "no", max_abs)
    end
end

function main()
    cfg = parse_args(ARGS)

    println("Commit comparison configuration:")
    println("  repo:         ", cfg.repo_path)
    println("  current-ref:  ", cfg.current_ref)
    println("  previous-ref: ", cfg.previous_ref)
    println("  samples:      ", cfg.samples)
    println("  evals:        ", cfg.evals)
    println("  seed:         ", cfg.seed)
    println("  stack:        ", join(["$(x)x$(y)x$(z)" for (x, y, z) in cfg.stack_sizes], ", "))
    println("  noise:        ", cfg.noise_sigma)
    println()

    child_file = write_child_script()
    try
        curr = run_ref(
            cfg.repo_path,
            cfg.current_ref,
            child_file,
            cfg.stack_sizes,
            cfg.samples,
            cfg.evals,
            cfg.seed,
            cfg.noise_sigma,
        )
        prev = run_ref(
            cfg.repo_path,
            cfg.previous_ref,
            child_file,
            cfg.stack_sizes,
            cfg.samples,
            cfg.evals,
            cfg.seed,
            cfg.noise_sigma,
        )
        print_table(curr, prev, cfg.current_ref, cfg.previous_ref)
    finally
        rm(dirname(child_file); force=true, recursive=true)
    end
end

main()
