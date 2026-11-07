"""Generate Table 1 and its saved accuracy and single-thread timing evidence."""

using DrWatson
@quickactivate "Ptd"
using PrettyTables, Printf, Statistics, JLD2, SHA, LinearAlgebra
using Ptd
import Random
import Ptd: DcEgm

const DETERMINISTIC_BENCHMARK_SCHEMA = "ptd-deterministic-benchmark-v3"
const DETERMINISTIC_TIE_RTOL = 64 * eps(Float64)

"""Deterministic benchmark summaries, raw draws, and their reproducibility record."""
struct DeterministicBenchmarkResult
    data::Matrix{Any}
    winners::Vector{Int} # strict wins plus shared-best ties, retained for callers
    strict_wins::Vector{Int}
    ties::Vector{Int}
    median_loss_bp::Vector{Float64}
    p95_loss_bp::Vector{Float64}
    max_loss_bp::Vector{Float64}
    batch_time_seconds::Vector{Float64}
    specs::Vector{ContestSpec}
    segment_counts::Matrix{Int}
    mean_segments::Float64
    max_segments::Int
    artifact::NamedTuple
end

"""Paths of Table 1 and its accuracy artifact, together with benchmark results."""
struct DeterministicOutputs
    table_path::String
    artifact_path::String
    benchmark::DeterministicBenchmarkResult
end

"""Consumption-equivalent gap against PTD in basis points; negative means a loss."""
function ce_gap_basis_points(value::Float64, value_ptd::Float64;
    beta::Float64, gamma::Float64, t_max::Int)
    budget = sum(beta .^ (0:t_max-1))
    if isapprox(gamma, 1.0; atol=1e-12)
        return 1e4 * expm1((value - value_ptd) / budget)
    end
    numerator = budget + (1 - gamma) * value
    denominator = budget + (1 - gamma) * value_ptd
    return 1e4 * expm1(log(numerator / denominator) / (1 - gamma))
end

"""Classify unique best methods and shared-best ties at the utility-sum scale."""
function deterministic_outcomes(values, absolute_utility_sums)
    all(isfinite, values) && all(isfinite, absolute_utility_sums) ||
        throw(ArgumentError("deterministic welfare comparisons require finite values"))
    length(values) == length(absolute_utility_sums) ||
        throw(ArgumentError("utility values and scales must have matching lengths"))
    tolerance = DETERMINISTIC_TIE_RTOL * max(1.0, maximum(absolute_utility_sums))
    best = maximum(values)
    tied_best = values .>= best - tolerance
    unique_best = count(tied_best) == 1
    return (; strict_wins=tied_best .& unique_best,
        ties=tied_best .& !unique_best, tolerance)
end

"""The three original grids plus a wide log grid with the same 400-node budget."""
function deterministic_asset_grids()
    return [collect(range(0.0, 50.0; length=100)),
        collect(range(0.0, 100.0; length=200)),
        collect(range(0.0, 100.0; length=400)),
        Ptd.iid_dcegm_asset_grid(400; asset_maximum=10_000.0)]
end

function _grid_description(grid)
    linear = all(isapprox.(diff(grid), (grid[end] - grid[1]) / (length(grid) - 1);
        rtol=1e-10, atol=1e-13))
    log_steps = diff(log1p.(grid))
    logarithmic = all(isapprox.(log_steps,
        (log1p(grid[end]) - log1p(grid[1])) / (length(grid) - 1);
        rtol=1e-10, atol=1e-13))
    bounds = @sprintf("[%g,%g]", first(grid), last(grid))
    suffix = linear ? "" : logarithmic ? " (log space)" : " (custom)"
    return "$bounds, $(length(grid))$suffix"
end

function _deterministic_sources()
    root = pkgdir(Ptd)
    paths = [joinpath("scripts", "2_deterministic.jl"), "Project.toml", "Manifest.toml"]
    for (directory, _, files) in walkdir(joinpath(root, "src"))
        append!(paths, [relpath(joinpath(directory, file), root)
            for file in files if endswith(file, ".jl")])
    end
    return [(; path, sha256=bytes2hex(sha256(read(joinpath(root, path)))))
        for path in sort!(paths) if isfile(joinpath(root, path))]
end

"""Evaluate indexed draws and propagate the earliest failed draw after workers stop."""
function _run_deterministic_draws!(work, specs; parallel::Bool=true)
    failures = fill!(Vector{Any}(undef, length(specs)), nothing)
    first_failed = Threads.Atomic{Int}(length(specs) + 1)
    function perform(j)
        j > first_failed[] && return
        try
            work(j, specs[j])
        catch error
            failures[j] = error
            Threads.atomic_min!(first_failed, j)
        end
    end
    if parallel
        Threads.@threads for j in eachindex(specs)
            perform(j)
        end
    else
        for j in eachindex(specs)
            perform(j)
        end
    end
    if first_failed[] <= length(specs)
        j = first_failed[]
        @error "deterministic benchmark stopped" draw=j spec=specs[j]
        throw(failures[j])
    end
    return nothing
end

"""
    randomized_performance(; n_draws=10_000, asset_grids=deterministic_asset_grids(),
        r=0.1, x_floor=3.0, t_max=50, income=1.0, save_path="", artifact_path="",
        parallel=true)

Run the unchanged Latin-hypercube parameter design. Compare policy-path utilities
with a tolerance scaled by the sum of absolute discounted utility terms. Record
strict wins separately from ties, CE-loss quantiles, and segment counts at every
solved date (including the terminal date). Uncertified PTD tails stop the run.
Draws are generated before parallel execution and stored in their original order.
Batch durations remain in the raw evidence. Table 1 uses a separate
single-thread timing run across sampled configurations.
"""
function randomized_performance(; n_draws::Int=10_000,
    asset_grids::Vector{Vector{Float64}}=deterministic_asset_grids(),
    r::Float64=0.1, x_floor::Float64=3.0, t_max::Int=50, income::Float64=1.0,
    save_path::String="", artifact_path::String="", parallel::Bool=true)
    isempty(asset_grids) && throw(ArgumentError("at least one DC-EGM grid is required"))
    timing = nothing
    sources = _deterministic_sources()
    config = ContestDesignConfig(n_contests=n_draws, horizon=t_max, floor=x_floor)
    specs = build_contest_design(config)
    n_methods = 1 + length(asset_grids)
    incomes = fill(income, (1, t_max))
    fusion_options = FusionOptions()

    gaps_bp = zeros(n_draws, n_methods)
    utilities = zeros(n_draws, n_methods)
    utility_scales = zeros(n_draws, n_methods)
    tie_tolerances = zeros(n_draws)
    segment_counts = zeros(Int, n_draws, t_max)
    batch_time_seconds = zeros(n_methods)
    for m in 1:n_methods
        method = m == 1 ? "PTD" : "DC-EGM$(m-1)"
        work = function (j, spec)
            if m == 1
                agent_ptd = DeterministicProblem(β=spec.beta, γ=spec.gamma, r=r,
                    xFloor=x_floor, incomes=incomes)
                deterministic_ptd!(agent_ptd; options=fusion_options)
                utilities[j, m], utility_scales[j, m] =
                    lifetime_utility_and_scale(agent_ptd, spec.initial_cash)
                segment_counts[j, :] .= length.(vec(agent_ptd.policy))
            else
                agent_egm = DcEgm.build_agent(model="deterministic",
                    β=spec.beta, γ=spec.gamma, r=r, xFloor=x_floor,
                    tMax=t_max, a=asset_grids[m-1], incomes=vec(incomes))
                DcEgm.egm!(agent_egm)
                utilities[j, m], utility_scales[j, m] =
                    lifetime_utility_and_scale(agent_egm, spec.initial_cash)
            end
            gaps_bp[j, m] = ce_gap_basis_points(utilities[j, m], utilities[j, 1];
                beta=spec.beta, gamma=spec.gamma, t_max=t_max)
        end
        # Warm this exact worker and threaded dispatch before starting its clock.
        # The timed pass rebuilds every policy and overwrites the warmup row.
        _run_deterministic_draws!(work, specs[1:1]; parallel)
        @info "deterministic accuracy batch starting" method draws=n_draws
        batch_started = time_ns()
        _run_deterministic_draws!(work, specs; parallel)
        batch_time_seconds[m] = (time_ns() - batch_started) / 1e9
        @info "deterministic accuracy batch completed" method draws=n_draws threads=(parallel ? Threads.nthreads() : 1) wall_seconds=batch_time_seconds[m]
    end
    strict_wins, ties = zeros(Int, n_methods), zeros(Int, n_methods)
    for j in eachindex(specs)
        outcomes = deterministic_outcomes(view(utilities, j, :), view(utility_scales, j, :))
        strict_wins .+= outcomes.strict_wins
        ties .+= outcomes.ties
        tie_tolerances[j] = outcomes.tolerance
    end

    losses = max.(-gaps_bp, 0.0)
    median_loss_bp = vec(median(losses; dims=1))
    p95_loss_bp = [quantile(view(losses, :, m), 0.95) for m in 1:n_methods]
    max_loss_bp = vec(maximum(losses; dims=1))
    mean_segments, max_segments = mean(segment_counts), maximum(segment_counts)
    tail_audit = Ptd.fusion_tail_diagnostics(fusion_options)
    summary = (; strict_wins, ties, median_loss_bp, p95_loss_bp, max_loss_bp,
        batch_time_seconds, mean_segments, max_segments)
    spec_records = [(; s.id, s.beta, s.gamma, s.initial_cash) for s in specs]
    artifact = (; schema=DETERMINISTIC_BENCHMARK_SCHEMA,
        configuration=(; n_draws, r, x_floor, t_max, income,
            parameter_seed=config.parameter_seed,
            bounds=(; config.bounds.beta, config.bounds.gamma, config.bounds.initial_cash),
            asset_grids, tie_rtol=DETERMINISTIC_TIE_RTOL),
        provenance=(; julia_version=string(VERSION), threads=Threads.nthreads(),
            parallel_draws=parallel, scope="accuracy_and_method_batch_wall_time",
            parameter_design_fingerprint=stable_fingerprint(spec_records),
            source_files=sources, source_fingerprint=stable_fingerprint(sources)),
        specs=spec_records, utilities, absolute_utility_sums=utility_scales,
        tie_tolerances, ce_gaps_bp=gaps_bp, segment_counts, summary, timing,
        batch_timing=(; scope=:complete_method_accuracy_pass,
            threads=parallel ? Threads.nthreads() : 1,
            included=(:policy_construction, :policy_path_utility, :ce_gap, :ptd_segment_counts),
            excluded=(:draw_generation, :warmup, :cross_method_summaries, :file_io, :plotting)),
        tail_safeguard=(; search_bound=fusion_options.search_bound,
            allow_uncertified_tail=false, unresolved_count=tail_audit.unresolved_count))
    sources == _deterministic_sources() || error(
        "Computational source changed during the deterministic benchmark; rerun before publishing.")
    if !isempty(artifact_path)
        Ptd._validate_data_output(artifact_path)
        mkpath(dirname(artifact_path))
        JLD2.jldsave(artifact_path; artifact, payload_fingerprint=stable_fingerprint(artifact))
    end
    data, _ = _deterministic_table_data(artifact)
    isempty(save_path) || _write_deterministic_table(save_path, artifact)
    _print_deterministic_results(artifact)
    return DeterministicBenchmarkResult(data, strict_wins + ties, strict_wins, ties,
        median_loss_bp, p95_loss_bp, max_loss_bp, batch_time_seconds, specs,
        segment_counts, mean_segments, max_segments, artifact)
end

"""Only heterogeneous timing records can populate Table 1."""
function _table1_timing(artifact)
    timing = artifact.timing
    isnothing(timing) && return nothing
    hasproperty(timing.configuration, :n_draws) &&
        all(row -> hasproperty(row, :mean_seconds) && hasproperty(row, :median_seconds),
            timing.rows) || return nothing
    return timing
end

function _deterministic_table_data(artifact)
    summary, grids = artifact.summary, artifact.configuration.asset_grids
    timing = _table1_timing(artifact)
    data = Matrix{Any}(undef, length(grids) + 1, 7)
    for m in axes(data, 1)
        data[m, :] = [m == 1 ? "PTD" : "DC-EGM$(m-1)",
            m == 1 ? "grid free" : _grid_description(grids[m-1]),
            isnothing(timing) ? "not reported" :
                @sprintf("%.3f", 1e3 * timing.rows[m].mean_seconds),
            isnothing(timing) ? "not reported" :
                @sprintf("%.3f", 1e3 * timing.rows[m].median_seconds),
            summary.strict_wins[m], summary.ties[m],
            @sprintf("%.2f", summary.max_loss_bp[m] / 100)]
    end
    header = ["Method", "Grid of a", "Mean time (ms)", "Median time (ms)",
        "Strict wins", "Ties", "Max. CEG (%)"]
    return data, header
end

"""Print Table 1 and the policy-segment statistics used in its discussion."""
function _print_deterministic_results(artifact)
    data, header = _deterministic_table_data(artifact)
    pretty_table(data; header)
    counts = artifact.segment_counts
    @printf("PTD policy segments over %d draws and %d dates (including terminal): mean = %.6f (%.2f to two decimals), maximum = %d.\n",
        size(counts, 1), size(counts, 2), mean(counts), mean(counts), maximum(counts))
    return nothing
end

"""Write the merged Table 1 and short notes from saved evidence."""
function _write_deterministic_table(path::String, artifact)
    data, _ = _deterministic_table_data(artifact)
    timing = _table1_timing(artifact)
    Ptd._validate_data_output(path)
    mkpath(dirname(path))
    open(path, "w") do f
        write(f, "% Generated by scripts/2_deterministic.jl; do not edit.\n")
        write(f, "\\begin{table}[htbp]\n\\centering\n")
        write(f, "\\caption{Deterministic PTD and DC-EGM performance}\n" *
            "\\label{tab: deterministic performance}\n" *
            "{\\small\\setlength{\\tabcolsep}{4pt}\n")
        println(f, raw"\begin{tabular}{llrrrrr}")
        println(f, raw"\toprule")
        println(f, raw"Method & Grid of $a$ & Mean time & Median time & Strict wins & Ties & Max. CEG", " \\\\")
        println(f, raw" & & (ms) & (ms) & & & (\%)", " \\\\")
        println(f, raw"\midrule")
        for m in axes(data, 1)
            println(f, join(data[m, :], " & "), " \\\\")
        end
        println(f, raw"\bottomrule")
        println(f, raw"\end{tabular}")
        write(f, "}\n\\fnote{Notes: ")
        if isnothing(timing)
            write(f, "Single-path timing is not reported. ")
        else
            n_timing = replace(string(timing.configuration.n_draws), r"(\d)(?=(\d{3})+$)" => s"\1,")
            write(f, "Times cover a fresh solve and one path on a single thread after warmup, " *
                "using a common $n_timing-draw subset. ")
        end
        n_accuracy = replace(string(artifact.configuration.n_draws), r"(\d)(?=(\d{3})+$)" => s"\1,")
        write(f, "Accuracy uses $n_accuracy draws. Strict wins and ties count the best utilities " *
            "within numerical tolerance; a tie can include several methods. " *
            "Max. CEG is the largest consumption-equivalent gap " *
            "(shortfall relative to PTD), in percent.}\n")
        if !isnothing(timing)
            write(f, "% Timing artifact fingerprint: $(timing.artifact_fingerprint)\n")
            write(f, "% Timing source fingerprint: $(timing.source_fingerprint)\n")
        end
        write(f, "\\end{table}\n")
    end
    return nothing
end

"""Authenticate saved benchmark draws and regenerate the paper's table."""
function render_deterministic_benchmark(artifact_path::AbstractString;
    table_path::AbstractString=Ptd.tables_dir("deterministic_performance.tex"),
    timing_artifact_path::Union{Nothing,AbstractString}=nothing)
    saved = JLD2.load(artifact_path)
    artifact = saved["artifact"]
    artifact.schema == DETERMINISTIC_BENCHMARK_SCHEMA ||
        throw(ArgumentError("unsupported deterministic benchmark schema"))
    stable_fingerprint(artifact) == saved["payload_fingerprint"] ||
        throw(ArgumentError("deterministic benchmark payload fingerprint mismatch"))
    if !isnothing(timing_artifact_path)
        timing = load_deterministic_random_timings(timing_artifact_path, artifact)
        artifact = merge(artifact, (; timing))
    end
    _write_deterministic_table(String(table_path), artifact)
    _print_deterministic_results(artifact)
    return String(table_path)
end

"""Evaluate policy-path utility and the absolute summation scale used for ties."""
function lifetime_utility_and_scale(agent::Ptd.DeterministicProblem, x::Float64)
    value, scale = 0.0, 0.0
    for t in 1:agent.tMax
        c = c_x(agent, x; i=1, t)
        term = agent.β^(t - 1) * u(agent, c)
        value += term
        scale += abs(term)
        t < agent.tMax && (x = xNext_a(agent, x - c; i=1, t))
    end
    return value, scale
end

function lifetime_utility_and_scale(agent::DcEgm.AgentD, x::Float64)
    value, scale = 0.0, 0.0
    for t in 1:agent.bf.tMax
        c = DcEgm.c_x(agent, x; t)
        term = agent.bf.β^(t - 1) * DcEgm.u(agent, c)
        value += term
        scale += abs(term)
        t < agent.bf.tMax && (x = DcEgm.xNext_a(agent, x - c; t))
    end
    return value, scale
end

const DETERMINISTIC_RANDOM_TIMING_SCHEMA = "ptd-deterministic-random-timing-v1"
const DETERMINISTIC_RANDOM_TIMING_SEED = UInt64(0x5054445f4454494d)

_random_timing_require(condition, message) = condition || throw(ArgumentError(message))

_deterministic_accuracy_fingerprint(benchmark) =
    stable_fingerprint(merge(benchmark, (; timing=nothing)))

function _deterministic_random_methods(benchmark)
    return (:ptd, (Symbol("dcegm", i) for i in eachindex(benchmark.configuration.asset_grids))...)
end

function _deterministic_random_model(benchmark)
    config = benchmark.configuration
    return (; return_rate=config.r, cash_floor=config.x_floor, horizon=config.t_max,
        income=config.income, asset_grids=config.asset_grids)
end

function _deterministic_random_indices(population::Int, count::Int, seed::UInt64)
    _random_timing_require(1 <= count <= population,
        "timing sample size must be between one and the saved accuracy draw count")
    return Random.randperm(Random.Xoshiro(seed), population)[1:count]
end

function _validate_deterministic_timing_benchmark(benchmark)
    _random_timing_require(benchmark.schema == DETERMINISTIC_BENCHMARK_SCHEMA,
        "unsupported deterministic accuracy artifact schema")
    count = benchmark.configuration.n_draws
    methods = _deterministic_random_methods(benchmark)
    _random_timing_require(count == length(benchmark.specs) && count >= 1 &&
        size(benchmark.utilities) == size(benchmark.absolute_utility_sums) == (count, length(methods)),
        "deterministic accuracy dimensions are inconsistent")
    _random_timing_require(all(isfinite, benchmark.utilities) &&
        all(x -> isfinite(x) && x >= 0, benchmark.absolute_utility_sums),
        "deterministic accuracy utilities and scales must be finite")
    _random_timing_require(length(unique(spec.id for spec in benchmark.specs)) == count,
        "deterministic accuracy draw identities must be unique")
    _random_timing_require(benchmark.provenance.parameter_design_fingerprint ==
        stable_fingerprint(benchmark.specs), "deterministic accuracy design fingerprint differs")
    _random_timing_require(benchmark.tail_safeguard.unresolved_count == 0 &&
        benchmark.tail_safeguard.allow_uncertified_tail === false,
        "deterministic accuracy artifact has uncertified tails")
    return nothing
end

function _build_deterministic_random_policy(method_index, spec, model, incomes, income_vector)
    if method_index == 1
        policy = DeterministicProblem(β=spec.beta, γ=spec.gamma,
            r=model.return_rate, xFloor=model.cash_floor, incomes=incomes)
        deterministic_ptd!(policy; options=FusionOptions())
    else
        policy = DcEgm.build_agent(model="deterministic", β=spec.beta, γ=spec.gamma,
            r=model.return_rate, xFloor=model.cash_floor, tMax=model.horizon,
            a=model.asset_grids[method_index - 1], incomes=income_vector)
        DcEgm.egm!(policy)
    end
    return policy
end

"""A fresh complete solve and one path; all comparison checks follow the clocks."""
function _measure_deterministic_random_path(method_index, spec, model, incomes, income_vector)
    started = time_ns()
    policy = _build_deterministic_random_policy(method_index, spec, model, incomes, income_vector)
    construction_seconds = (time_ns() - started) / 1e9
    started = time_ns()
    utility, absolute_utility_sum = lifetime_utility_and_scale(policy, spec.initial_cash)
    evaluation_seconds = (time_ns() - started) / 1e9
    return (; construction_seconds, evaluation_seconds,
        total_seconds=construction_seconds + evaluation_seconds, utility, absolute_utility_sum)
end

function _validate_deterministic_random_utility(sample, benchmark, index, method_index)
    _random_timing_require(isfinite(sample.utility) && isfinite(sample.absolute_utility_sum) &&
        sample.absolute_utility_sum >= 0, "timed deterministic path has invalid utility")
    scale = max(1.0, maximum(view(benchmark.absolute_utility_sums, index, :)),
        sample.absolute_utility_sum)
    tolerance = 64eps(Float64) * scale
    _random_timing_require(abs(sample.utility - benchmark.utilities[index, method_index]) <= tolerance,
        "timed utility differs from accuracy draw $(benchmark.specs[index].id), method $method_index")
    _random_timing_require(abs(sample.absolute_utility_sum -
        benchmark.absolute_utility_sums[index, method_index]) <= tolerance,
        "timed utility scale differs from accuracy draw $(benchmark.specs[index].id), method $method_index")
    return nothing
end

function _deterministic_random_rows(measurements, methods, count)
    return [begin
        samples = filter(sample -> sample.method === method, measurements)
        (; method, n_draws=count,
            mean_seconds=mean(sample.total_seconds for sample in samples),
            median_seconds=median([sample.total_seconds for sample in samples]))
    end for method in methods]
end

function _validate_deterministic_random_artifact(artifact, benchmark)
    _validate_deterministic_timing_benchmark(benchmark)
    _random_timing_require(artifact.schema == DETERMINISTIC_RANDOM_TIMING_SCHEMA,
        "unsupported deterministic random-timing schema")
    payload = (; (name => getproperty(artifact, name) for name in keys(artifact)
        if name !== :authentication_fingerprint)...)
    _random_timing_require(stable_fingerprint(payload) == artifact.authentication_fingerprint,
        "deterministic random-timing payload fingerprint differs")
    _random_timing_require(artifact.accuracy.artifact_fingerprint == _deterministic_accuracy_fingerprint(benchmark) &&
        artifact.accuracy.source_fingerprint == benchmark.provenance.source_fingerprint &&
        artifact.accuracy.design_fingerprint == benchmark.provenance.parameter_design_fingerprint,
        "deterministic timings were measured against a different accuracy artifact")
    config = artifact.configuration
    methods = _deterministic_random_methods(benchmark)
    _random_timing_require(config.population_size == length(benchmark.specs) &&
        config.methods == methods && config.model == _deterministic_random_model(benchmark),
        "deterministic timing calibration, grids, or methods differ from the accuracy experiment")
    expected_indices = _deterministic_random_indices(config.population_size, config.n_draws, config.seed)
    _random_timing_require(config.selection === :seeded_randperm_without_replacement &&
        config.sample_indices == expected_indices && config.specs == benchmark.specs[expected_indices],
        "deterministic timing sample does not match its seeded accuracy subset")
    _random_timing_require(config.julia_threads == config.blas_threads == config.paths_per_draw == 1 &&
        config.measured_runs_per_draw == 1 && config.warmup === :discarded_full_solve_and_path_per_method &&
        config.method_order === :rotating_by_draw && config.utility_rtol == 64eps(Float64) &&
        config.total === :sum_of_separately_measured_construction_and_path,
        "unsupported deterministic random-timing measurement protocol")
    _random_timing_require(artifact.runtime.julia_threads == artifact.runtime.blas_threads == 1 &&
        !isempty(artifact.runtime.julia_version), "invalid deterministic timing runtime metadata")
    _random_timing_require(!isempty(artifact.source_files) &&
        stable_fingerprint(artifact.source_files) == artifact.source_fingerprint,
        "deterministic timing source fingerprint differs")
    expected_order = [(index, methods[mod1(position + offset, length(methods))])
        for (position, index) in enumerate(expected_indices) for offset in 0:length(methods)-1]
    _random_timing_require(length(artifact.measurements) == length(expected_order),
        "deterministic timing measurements are incomplete")
    for (sample, (index, method)) in zip(artifact.measurements, expected_order)
        _random_timing_require(sample.accuracy_index == index && sample.method === method &&
            sample.spec_id == benchmark.specs[index].id,
            "deterministic timing measurements have different draw identities or method order")
        _random_timing_require(all(x -> isfinite(x) && x >= 0,
            (sample.construction_seconds, sample.evaluation_seconds, sample.total_seconds)) &&
            sample.total_seconds == sample.construction_seconds + sample.evaluation_seconds,
            "invalid deterministic timing duration")
        method_index = findfirst(==(method), methods)
        _validate_deterministic_random_utility(sample, benchmark, index, method_index)
    end
    expected_rows = _deterministic_random_rows(artifact.measurements, methods, config.n_draws)
    _random_timing_require(artifact.rows == expected_rows,
        "deterministic timing mean or median differs from raw measurements")
    return nothing
end

"""
    run_deterministic_random_timings(accuracy_path; n_draws=1000,
        seed=DETERMINISTIC_RANDOM_TIMING_SEED, output_path=nothing)

Measure every method on the same seeded sample without replacement from a saved
accuracy design. Each draw has one fresh construction/full solve and one path.
Warmup, model/grid setup, post-warmup GC, utility comparisons, summary
statistics, progress messages, and file I/O are outside the reported intervals.
Automatic garbage collection during a solve or path remains inside its timer.
The saved accuracy artifact and existing timing files are never overwritten.
"""
function run_deterministic_random_timings(accuracy_path::AbstractString;
    n_draws::Int=1000, seed::Integer=DETERMINISTIC_RANDOM_TIMING_SEED,
    output_path::Union{Nothing,AbstractString}=nothing)
    _random_timing_require(Threads.nthreads() == 1 && LinearAlgebra.BLAS.get_num_threads() == 1,
        "deterministic random timings require one Julia thread and one BLAS thread")
    _random_timing_require(0 <= seed <= typemax(UInt64), "timing seed must fit UInt64")
    source_path = abspath(accuracy_path)
    destination = isnothing(output_path) ? nothing : abspath(output_path)
    if !isnothing(destination)
        _random_timing_require(destination != source_path && !ispath(destination) && !islink(destination),
            "timing output must be a new file distinct from the accuracy artifact")
    end
    accuracy_sha256 = bytes2hex(sha256(read(source_path)))
    saved = JLD2.load(source_path)
    benchmark = saved["artifact"]
    _random_timing_require(stable_fingerprint(benchmark) == saved["payload_fingerprint"],
        "deterministic accuracy payload fingerprint differs")
    _validate_deterministic_timing_benchmark(benchmark)
    indices = _deterministic_random_indices(length(benchmark.specs), n_draws, UInt64(seed))
    methods = _deterministic_random_methods(benchmark)
    model = _deterministic_random_model(benchmark)
    incomes = fill(model.income, 1, model.horizon)
    income_vector = vec(incomes)
    sources = _deterministic_sources()
    measurements = NamedTuple[]
    first_index = first(indices)
    for (method_index, method) in enumerate(methods)
        println("Deterministic random timing warmup: ", method)
        flush(stdout)
        sample = _measure_deterministic_random_path(method_index, benchmark.specs[first_index],
            model, incomes, income_vector)
        _validate_deterministic_random_utility(sample, benchmark, first_index, method_index)
    end
    GC.gc()
    for (position, index) in enumerate(indices)
        spec = benchmark.specs[index]
        for offset in 0:length(methods)-1
            method_index = mod1(position + offset, length(methods))
            sample = _measure_deterministic_random_path(method_index, spec, model, incomes, income_vector)
            _validate_deterministic_random_utility(sample, benchmark, index, method_index)
            push!(measurements, merge((; method=methods[method_index], accuracy_index=index, spec_id=spec.id),
                sample))
        end
        if position % 50 == 0 || position == n_draws
            println("Deterministic random timing: ", position, "/", n_draws, " configurations completed")
            flush(stdout)
        end
    end
    _random_timing_require(sources == _deterministic_sources(),
        "computational source changed during deterministic timing")
    _random_timing_require(accuracy_sha256 == bytes2hex(sha256(read(source_path))),
        "accuracy artifact changed during deterministic timing")
    configuration = (; n_draws, population_size=length(benchmark.specs), seed=UInt64(seed),
        selection=:seeded_randperm_without_replacement, sample_indices=indices,
        specs=benchmark.specs[indices], methods, model, julia_threads=1, blas_threads=1,
        paths_per_draw=1, measured_runs_per_draw=1,
        warmup=:discarded_full_solve_and_path_per_method, method_order=:rotating_by_draw,
        utility_rtol=64eps(Float64), total=:sum_of_separately_measured_construction_and_path,
        inside_timers=(:fresh_policy_construction, :full_backward_solve, :one_path_utility_and_scale,
            :automatic_gc),
        outside_timers=(:model_and_grid_setup, :sample_selection, :warmup, :postwarmup_gc,
            :accuracy_comparison, :ce_and_segment_summaries, :summary_statistics, :progress_logging, :file_io))
    payload = (; schema=DETERMINISTIC_RANDOM_TIMING_SCHEMA, configuration,
        accuracy=(; source_path=Ptd.portable_path(source_path),
            file_sha256=accuracy_sha256,
            artifact_fingerprint=_deterministic_accuracy_fingerprint(benchmark),
            original_payload_fingerprint=saved["payload_fingerprint"],
            source_fingerprint=benchmark.provenance.source_fingerprint,
            design_fingerprint=benchmark.provenance.parameter_design_fingerprint),
        runtime=(; julia_version=string(VERSION), julia_threads=Threads.nthreads(),
            blas_threads=LinearAlgebra.BLAS.get_num_threads(), cpu_name=Sys.CPU_NAME,
            machine=Sys.MACHINE, kernel=string(Sys.KERNEL)),
        source_files=sources, source_fingerprint=stable_fingerprint(sources), measurements,
        rows=_deterministic_random_rows(measurements, methods, n_draws))
    artifact = merge(payload, (; authentication_fingerprint=stable_fingerprint(payload)))
    _validate_deterministic_random_artifact(artifact, benchmark)
    if !isnothing(destination)
        mkpath(dirname(destination))
        temporary = Ptd._temporary_jld_path(destination)
        try
            JLD2.jldsave(temporary; artifact)
            _random_timing_require(!ispath(destination) && !islink(destination),
                "timing output appeared during measurement; refusing to overwrite it")
            Base.rename(temporary, destination)
        finally
            isfile(temporary) && rm(temporary)
        end
    end
    return artifact
end

"""Authenticate raw timing evidence and its exact parent accuracy design."""
function load_deterministic_random_timings(path::AbstractString, benchmark)
    source_path = abspath(path)
    artifact = JLD2.load(source_path, "artifact")
    _validate_deterministic_random_artifact(artifact, benchmark)
    return (; rows=artifact.rows, configuration=artifact.configuration,
        artifact_fingerprint=artifact.authentication_fingerprint,
        source_fingerprint=artifact.source_fingerprint, source_path,
        artifact_sha256=bytes2hex(sha256(read(source_path))))
end

"""Write Table 1 and retain every accuracy draw on disk."""
function deterministic_outputs(; n_draws::Int=10_000,
    table_directory::AbstractString=Ptd.tables_dir(),
    artifact_directory::AbstractString=normpath(table_directory) ==
        normpath(Ptd.tables_dir()) ? projectdir("data") :
        joinpath(dirname(table_directory), "data"))
    Ptd._validate_data_output(joinpath(table_directory, "deterministic_performance.tex"))
    Ptd._validate_data_output(joinpath(artifact_directory, "deterministic-benchmark.jld2"))
    mkpath(table_directory)
    asset_grids = deterministic_asset_grids()
    table_path = joinpath(table_directory, "deterministic_performance.tex")
    artifact_path = joinpath(artifact_directory, "deterministic-benchmark.jld2")
    benchmark = randomized_performance(; n_draws, asset_grids,
        save_path=table_path, artifact_path)
    return DeterministicOutputs(table_path, artifact_path, benchmark)
end

"""
Table 1 commands:
  --accuracy-only                  Generate 10,000 accuracy draws with the selected Julia threads (default).
  --timing-only [ACCURACY]          Time 1,000 sampled draws on one thread and render.
  --render-only [ACCURACY]          Render saved accuracy and heterogeneous timings.
Options: --timing-draws N, --timing-output FILE, --timing-artifact FILE, --table-output FILE.
"""
function deterministic_main(args=ARGS)
    if !isempty(args) && first(args) == "--fixed-timing"
        return Ptd.FixedCalibrationTiming.main(args[2:end]; panels=(:deterministic,))
    end
    mode = :accuracy
    accuracy_path = projectdir("data", "deterministic-benchmark.jld2")
    timing_path = projectdir("data", "deterministic-random-timings.jld2")
    table_path = Ptd.tables_dir("deterministic_performance.tex")
    timing_draws = 1000
    seen_mode = false
    index = 1
    while index <= length(args)
        option = args[index]
        if option in ("--accuracy-only", "--timing-only", "--render-only")
            seen_mode && throw(ArgumentError("choose one Table 1 mode"))
            seen_mode = true
            mode = option == "--accuracy-only" ? :accuracy :
                option == "--timing-only" ? :timing : :render
            if mode != :accuracy && index < length(args) && !startswith(args[index + 1], "--")
                index += 1
                accuracy_path = abspath(args[index])
            end
        elseif option in ("--timing-draws", "--timing-output", "--timing-artifact", "--table-output")
            index < length(args) || throw(ArgumentError("missing value for $option"))
            index += 1
            value = args[index]
            if option == "--timing-draws"
                timing_draws = parse(Int, value)
            elseif option == "--table-output"
                table_path = abspath(value)
            else
                timing_path = abspath(value)
            end
        else
            throw(ArgumentError("unknown Table 1 option: $option"))
        end
        index += 1
    end
    Ptd._validate_data_output(accuracy_path)
    Ptd._validate_data_output(timing_path)
    Ptd._validate_data_output(table_path)
    if mode == :accuracy
        LinearAlgebra.BLAS.set_num_threads(1)
        randomized_performance(; save_path=table_path, artifact_path=accuracy_path)
    elseif mode == :timing
        LinearAlgebra.BLAS.set_num_threads(1)
        run_deterministic_random_timings(accuracy_path;
            n_draws=timing_draws, output_path=timing_path)
        render_deterministic_benchmark(accuracy_path;
            table_path, timing_artifact_path=timing_path)
    else
        isfile(timing_path) || throw(ArgumentError(
            "Run --timing-only first or supply --timing-artifact."))
        render_deterministic_benchmark(accuracy_path;
            table_path, timing_artifact_path=timing_path)
    end
    println(table_path)
    return (; table_path, accuracy_path, timing_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    deterministic_main()
end
