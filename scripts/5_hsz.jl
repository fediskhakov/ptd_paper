"""Run or render HSZ accuracy, retained-policy timing, and grid-refinement experiments."""

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using DrWatson
@quickactivate "Ptd"

using LinearAlgebra
using Printf
using Ptd
using SHA
using JLD2
using Dates
using TOML

"""Locations of both saved HSZ artifacts, the path figure, and the two LaTeX tables."""
struct HSZPaperOutputs
    artifact_path::String
    graph_path::String
    timing_path::String
    utility_artifact_path::String
    utility_table_path::String
end

const HSZ_USAGE = "usage: 5_hsz.jl [--render-only | --run] " *
    "[--example-artifact PATH] [--utility-artifact PATH] [--timing-artifact PATH] [--output-dir DIR]; " *
    "5_hsz.jl --timing-only [--accuracy PATH] [--output PATH] [--resume] [--smoke]; " *
    "5_hsz.jl --fixed-timing [OPTIONS]; 5_hsz.jl --refinement [OPTIONS]"

function _hsz_options(args)
    mode = nothing
    values = Dict{String,String}()
    i = 1
    while i <= length(args)
        option = args[i]
        if option in ("--render-only", "--run")
            isnothing(mode) || throw(ArgumentError(
                "Specify only one execution mode. " * HSZ_USAGE))
            mode = option
        elseif option in ("--example-artifact", "--utility-artifact", "--timing-artifact", "--output-dir")
            haskey(values, option) && throw(ArgumentError(
                "Duplicate option: $option. " * HSZ_USAGE))
            i < length(args) && !startswith(args[i+1], "--") ||
                throw(ArgumentError("Missing path for $option. " * HSZ_USAGE))
            i += 1
            isempty(args[i]) && throw(ArgumentError(
                "Empty path for $option. " * HSZ_USAGE))
            values[option] = abspath(args[i])
        else
            throw(ArgumentError("Unknown option: $option. " * HSZ_USAGE))
        end
        i += 1
    end
    default_timing = projectdir("data", "hsz-path-timings.jld2")
    return (; run=mode == "--run",
        artifact_path=get(values, "--example-artifact",
            Ptd.HSZ.default_hsz_example_artifact_path()),
        utility_artifact_path=get(values, "--utility-artifact",
            Ptd.HSZ.default_hsz_utility_artifact_path()),
        timing_artifact_path=get(values, "--timing-artifact",
            mode != "--run" && isfile(default_timing) &&
                !haskey(values, "--example-artifact") ? default_timing : nothing),
        output_dir=get(values, "--output-dir", nothing))
end

"""
Render both saved HSZ accuracy experiments with separate single-thread timings.

Use `--run` to recompute both experiments; new elapsed times will vary by run.
The path figure goes to `Ptd.graphs_dir()` and both LaTeX tables to
`Ptd.tables_dir()`, the `Graphs/` and `Tables/` folders of the active project.
`--output-dir DIR` sends the path figure and both LaTeX tables to a single directory.
`--timing-artifact PATH` selects the fixed log-agent timing artifact. The default
prefers `data/hsz-path-timings.jld2`: one construction and 100 paths with a shared
policy. The table reports construction time and mean/median time in seconds,
adding the same fixed construction time to each measured path-simulation time.
When the saved timing artifact is unavailable, timing is displayed as `not reported`.
With custom accuracy artifacts, select matching timings explicitly.
Fresh accuracy runs attach earlier timings only when explicitly requested.
"""
function _hsz_accuracy_main(args)
    options = _hsz_options(args)
    artifact_path = options.artifact_path
    utility_artifact_path = options.utility_artifact_path
    output_dir = options.output_dir
    graph_directory = isnothing(output_dir) ? Ptd.graphs_dir() : output_dir
    table_directory = isnothing(output_dir) ? Ptd.tables_dir() : output_dir
    graph_path = Ptd._validate_data_output(
        joinpath(graph_directory, Ptd.HSZ.HSZ_PATHS_FILENAME))
    timing_path = Ptd._validate_data_output(
        joinpath(table_directory, Ptd.HSZ.HSZ_TIMING_FILENAME))
    utility_table_path = Ptd._validate_data_output(
        joinpath(table_directory, "hsz_utility_comparison.tex"))
    if options.run
        Ptd._validate_data_output(artifact_path)
        Ptd._validate_data_output(utility_artifact_path)
    end
    options.run && mkpath(dirname(artifact_path))
    # Check every destination together, including aliases of either artifact.
    Ptd.HSZ._validate_utility_render_paths(artifact_path,
        (utility_artifact_path, graph_path, timing_path,
            utility_table_path))
    isnothing(options.timing_artifact_path) || Ptd.HSZ._validate_utility_render_paths(
        options.timing_artifact_path,
        (artifact_path, utility_artifact_path, graph_path, timing_path, utility_table_path))
    if options.run
        BLAS.set_num_threads(1)
        config = Ptd.HSZ.production_hsz_example_config()
        Ptd.HSZ.run_hsz_example(config; output_path=artifact_path)
        Ptd.HSZ.run_hsz_utility_comparison(artifact_path;
            config, output_path=utility_artifact_path, overwrite=true)
    end
    artifact = Ptd.HSZ.load_hsz_example(artifact_path)
    utility = Ptd.HSZ.load_hsz_utility_comparison(utility_artifact_path)
    utility.provenance.source_sha256 == bytes2hex(SHA.sha256(read(artifact_path))) ||
        throw(ArgumentError("The HSZ utility artifact was produced from a different example artifact."))
    utility.provenance.source_scientific_fingerprint == artifact.scientific_fingerprint ||
        throw(ArgumentError("The HSZ example and utility artifacts have different scientific configurations."))
    if !isnothing(options.timing_artifact_path) &&
            Ptd.HSZ._field(JLD2.load(options.timing_artifact_path, "artifact"), :schema) ==
                Ptd.HSZ.HSZ_PATH_TIMING_SCHEMA
        Ptd.HSZ.load_hsz_path_timing_summary(options.timing_artifact_path,
            utility_artifact_path)
    end
    rendered = Ptd.HSZ.render_hsz_example(
        artifact_path; graph_path, timing_path, timing_artifact_path=options.timing_artifact_path)
    rendered_utility = Ptd.HSZ.render_hsz_utility_comparison(utility_artifact_path;
        table_path=utility_table_path)
    println("HSZ artifact: ", artifact_path)
    println("HSZ path figure: ", rendered.graph_path)
    println("HSZ timing table: ", rendered.timing_path)
    println("Single-thread timing artifact: ", isnothing(options.timing_artifact_path) ?
        "not reported" : options.timing_artifact_path)
    println("HSZ utility artifact: ", rendered_utility.artifact_path)
    println("HSZ utility table: ", rendered_utility.table_path)
    scientific = artifact.scientific_configuration
    println("HSZ unconditional log medical mean: ",
        scientific.process.unconditional_log_medical)
    for method in Ptd.HSZ.HSZ_EXAMPLE_METHODS
        hits = getproperty(artifact.paths, method).floor_hits
        println("  ", method, ": ", count(>(0), hits), "/", length(hits),
            " paths receive a floor transfer; ", sum(hits), "/",
            length(hits) * (scientific.horizon - 1), " transitions")
    end
    return HSZPaperOutputs(artifact_path, rendered.graph_path, rendered.timing_path,
        rendered_utility.artifact_path, rendered_utility.table_path)
end

function _hsz_path_timing_main(args)
    accuracy = projectdir("data", "hsz-utility-medical-1p5.jld2")
    output = projectdir("data", "hsz-path-timings.jld2")
    resume, smoke = false, false
    index = 1
    while index <= length(args)
        flag = args[index]
        if flag in ("--accuracy", "--output")
            index < length(args) && !startswith(args[index + 1], "--") ||
                throw(ArgumentError("$flag requires a path"))
            index += 1
            flag == "--accuracy" ? (accuracy = args[index]) : (output = args[index])
        elseif flag == "--resume"
            resume = true
        elseif flag == "--smoke"
            smoke = true
        else
            throw(ArgumentError("unknown argument: $flag"))
        end
        index += 1
    end
    BLAS.set_num_threads(1)
    output = Ptd._validate_data_output(output)
    return Ptd.HSZ.run_hsz_path_timings(accuracy; output_path=output, resume,
        require_production=!smoke)
end

const HSZ_REFINEMENT_DEFAULT_POINTS = 18
const HSZ_REFINEMENT_TARGET_SECONDS = 600.0
const HSZ_REFINEMENT_MAX_SECONDS = 900.0
const HSZ_REFINEMENT_USAGE = "usage: 5_hsz.jl --refinement " *
    "[--run|--render-only] [UTILITY_ARTIFACT] [REFINEMENT_ARTIFACT] " *
    "[FINER_NODES] [--max-seconds SECONDS] [--include-timings]; " *
    "--render-only without REFINEMENT_ARTIFACT composes the saved " *
    "18- and 19-node artifacts"

struct HSZRefinementBudgetExceeded <: Exception
    seconds::Float64
    log_path::String
    status_path::String
end
function Base.showerror(io::IO, error::HSZRefinementBudgetExceeded)
    print(io, "HSZ refinement exceeded its ", error.seconds,
        " second worker budget (warmup, one finer-grid solve, and replay). ",
        "The worker was stopped; existing artifacts and tables were preserved. ",
        "Diagnostic: ", error.status_path, "; log: ", error.log_path)
end

function _refinement_budget(seconds::Real)
    budget = Float64(seconds)
    isfinite(budget) && 0.0 < budget <= HSZ_REFINEMENT_MAX_SECONDS ||
        throw(ArgumentError("HSZ refinement worker budget must be in (0,900] seconds"))
    return budget
end

"""Terminate this subprocess only; never interrupt the calling Julia task."""
function _stop_refinement_worker(process)
    process_running(process) || return
    try
        kill(process, Base.SIGTERM)
    catch error
        process_running(process) && rethrow(error)
    end
    grace_end = time_ns() + UInt64(1_000_000_000)
    while process_running(process) && time_ns() < grace_end
        sleep(0.01)
    end
    if process_running(process)
        try
            kill(process, Base.SIGKILL)
        catch error
            process_running(process) && rethrow(error)
        end
    end
    return nothing
end

"""Run an isolated worker under a wall-clock deadline and retain its log."""
function _run_refinement_worker(command::Cmd;
    max_seconds::Real=HSZ_REFINEMENT_MAX_SECONDS, log_path::AbstractString)
    budget = _refinement_budget(max_seconds)
    Ptd._validate_data_output(log_path)
    mkpath(dirname(abspath(log_path)))
    started = time_ns()
    timed_out = false
    exit_code = -1
    termination_signal = 0
    open(log_path, "w") do log
        process = run(pipeline(command, stdout=log, stderr=log); wait=false)
        try
            while process_running(process)
                elapsed = (time_ns() - started) / 1e9
                if elapsed >= budget
                    timed_out = true
                    _stop_refinement_worker(process)
                    break
                end
                sleep(min(0.05, max(0.001, budget - elapsed)))
            end
            wait(process)
            exit_code = process.exitcode
            termination_signal = process.termsignal
        finally
            # Also avoid leaving an expensive worker behind if the caller fails.
            _stop_refinement_worker(process)
            wait(process)
        end
    end
    elapsed_seconds = (time_ns() - started) / 1e9
    # A late observed exit is treated conservatively: never publish beyond budget.
    timed_out |= elapsed_seconds > budget
    status = timed_out ? :timed_out :
        exit_code == 0 && termination_signal == 0 ? :completed : :failed
    return (; status, elapsed_seconds, max_seconds=budget, exit_code, termination_signal)
end

"""
    _refinement_arguments(args)

Parse the arguments of `5_hsz.jl --refinement`.

The first positional argument is the mode. `--run` (the default) solves
one finer grid in a worker with a wall-clock limit and renders it.
`--render-only` renders saved artifacts without solving: without a
`REFINEMENT_ARTIFACT` it composes the saved 18- and 19-node artifacts
behind the paper's grid-refinement footnote, otherwise it shows the one
artifact named. `--worker` is the private mode of the `--run` worker.

# Arguments
- `args::AbstractVector{<:AbstractString}`: the arguments after
  `--refinement`. Positional: `[--run|--render-only|--worker]
  [UTILITY_ARTIFACT] [REFINEMENT_ARTIFACT] [FINER_NODES]`, where file
  paths are relative to the working directory. Options, in any position:
  `--max-seconds SECONDS`, the worker's wall-clock limit in seconds, in
  (0, 900]; `--include-timings`, which adds the archived timing rows to
  the table.

# Returns
A `NamedTuple` with fields
- `mode::String`: `"--run"`, `"--render-only"` or `"--worker"`;
- `baseline::String`: absolute path of the saved HSZ utility comparison,
  by default `data/hsz-utility-medical-1p5.jld2` in the package;
- `output::String`: absolute path of the refinement artifact that `--run`
  writes or `--render-only` reads, by default
  `data/hsz-grid-refinement.jld2` in the package;
- `points::Int`: nodes in each exogenous grid of the finer run (18 by
  default);
- `max_seconds::Float64`: the worker's wall-clock limit in seconds;
- `include_timings::Bool`: whether the table shows archived timings;
- `refinement_paths::Vector{String}`: the refinement artifacts that the
  table combines: `[output]`, or for `--render-only` without a
  `REFINEMENT_ARTIFACT` both saved artifacts of the paper.
"""
function _refinement_arguments(args)

    positional = String[]
    max_seconds = HSZ_REFINEMENT_MAX_SECONDS
    include_timings = false
    index = 1
    while index <= length(args)
        if args[index] == "--max-seconds"
            index < length(args) ||
                throw(ArgumentError("--max-seconds needs a value"))
            max_seconds = _refinement_budget(parse(Float64, args[index + 1]))
            index += 2
        elseif args[index] == "--include-timings"
            include_timings = true
            index += 1
        else
            push!(positional, args[index])
            index += 1
        end
    end
    mode = isempty(positional) ? "--run" : first(positional)
    mode in ("--run", "--render-only", "--worker") ||
        throw(ArgumentError(HSZ_REFINEMENT_USAGE))
    # Only the mode starts with "--"; anything else is an unknown option.
    any(startswith("--"), positional[2:end]) &&
        throw(ArgumentError(HSZ_REFINEMENT_USAGE))
    length(positional) <= 4 || throw(ArgumentError(
        "run one finer grid per invocation; the default is 18 nodes"))
    baseline = length(positional) >= 2 ? abspath(positional[2]) :
        Ptd.HSZ.default_hsz_utility_artifact_path()
    output = length(positional) >= 3 ? abspath(positional[3]) :
        Ptd.HSZ.default_hsz_grid_refinement_path()
    points = length(positional) == 4 ? parse(Int, positional[4]) :
        HSZ_REFINEMENT_DEFAULT_POINTS
    points >= 2 || throw(ArgumentError(
        "the finer exogenous grid needs at least two nodes"))
    compose_paper_artifacts = mode == "--render-only" &&
        length(positional) < 3
    refinement_paths = compose_paper_artifacts ?
        Ptd.HSZ.paper_hsz_grid_refinement_paths() : [output]

    return (; mode, baseline, output, points, max_seconds, include_timings,
        refinement_paths)
end

"""
    _print_refinement_rows(composed)

Print where the grid-refinement table went and the numbers in each row.

# Arguments
- `composed::NamedTuple`: the result of
  `Ptd.HSZ.compose_hsz_grid_refinement`, with fields `table_path`,
  `family_size` and `rows`.

# Returns
- `Nothing`.
"""
function _print_refinement_rows(composed)

    println("HSZ refinement table: ", composed.table_path)
    println("HSZ refinement Bonferroni family size: ", composed.family_size)
    for row in composed.rows
        cash = @sprintf("%.2f", row.initial_cash)
        consumption = @sprintf("%.3f", row.initial_consumption)
        println("  ", row.nodes, " nodes, initial cash ", cash,
            ": initial G2EGM consumption ", consumption, "; ",
            row.g2egm_floor_paths, "/", row.n_paths,
            " paths with a G2EGM floor transfer")
    end

    return nothing
end

function _write_refinement_budget_status(path, result, request, log_path)
    Ptd._validate_data_output(path)
    status = Dict("status" => string(result.status),
        "target_construction_seconds" => HSZ_REFINEMENT_TARGET_SECONDS,
        "maximum_worker_seconds" => result.max_seconds,
        "elapsed_worker_seconds" => result.elapsed_seconds,
        "worker_exit_code" => Int(result.exit_code),
        "worker_termination_signal" => Int(result.termination_signal),
        "budget_scope" => "worker launch, compilation warmup, one finer-grid construction, and replay",
        "finer_nodes_per_exogenous_dimension" => request.points,
        "baseline_artifact" => request.baseline,
        "requested_output_artifact" => request.output,
        "worker_log" => abspath(log_path),
        "created_utc" => string(Dates.now(Dates.UTC)))
    open(path, "w") do io
        TOML.print(io, status)
    end
    return nothing
end

"""Route the isolated refinement worker through the HSZ entrypoint."""
function _refinement_worker_command(baseline, output, points)
    return `$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) --threads=$(Threads.nthreads()) --startup-file=no $(@__FILE__) --refinement --worker $baseline $output $points`
end

"""Keep worker diagnostics in runtime scratch, outside the saved-results directory."""
function _refinement_runtime_paths()
    directory = mktempdir(; prefix="ptd-hsz-refinement-", cleanup=false)
    return (; directory, log_path=joinpath(directory, "worker.log"),
        status_path=joinpath(directory, "budget-status.toml"))
end

"""
    _hsz_refinement_main(args)

Run or render the HSZ exogenous-grid refinement check.

`--render-only` solves nothing. Without a `REFINEMENT_ARTIFACT` it
composes the saved 18- and 19-node artifacts behind the paper's
grid-refinement footnote. `--run` solves one finer grid in a worker with a
wall-clock limit, saves its artifact and renders it with the reused
baseline grid. Either way the table goes to
`Ptd.tables_dir("hsz_grid_refinement.tex")`, and the initial G2EGM
consumption and the number of G2EGM floor-transfer paths are printed for
every grid and initial cash.

# Arguments
- `args::AbstractVector{<:AbstractString}`: the arguments after
  `--refinement`, described in `_refinement_arguments`.

# Returns
A `NamedTuple` with fields `output` (the refinement artifact path) and
`status` (`:rendered`, `:completed` for `--worker`, or the worker's
status after `--run`). Except for `--worker` it also has `table`, the
written table; `--run` adds `status_path` and `log_path`, the worker's
budget diagnostic and log.
"""
function _hsz_refinement_main(args)

    request = _refinement_arguments(args)
    (; mode, baseline, output, points, max_seconds, include_timings,
        refinement_paths) = request
    Ptd._validate_data_output(output)
    table_path = Ptd.tables_dir("hsz_grid_refinement.tex")
    if mode == "--render-only"
        composed = Ptd.HSZ.compose_hsz_grid_refinement(baseline,
            refinement_paths; table_path, include_timings)
        _print_refinement_rows(composed)
        return (; output, table=composed.table_path, status=:rendered)
    end
    BLAS.set_num_threads(1)
    if mode == "--worker"
        # This private mode is called only by the deadline-enforcing parent.
        Ptd.HSZ.run_hsz_grid_refinement(baseline; finer_points=[points],
            output_path=output, overwrite=false)
        return (; output, status=:completed)
    end

    Ptd.HSZ._validate_utility_render_paths(baseline, (output,))
    (; log_path, status_path) = _refinement_runtime_paths()
    Ptd.HSZ._validate_utility_render_paths(baseline, (output, log_path, status_path))
    println("HSZ refinement: one ", points, "-node exogenous grid; ",
        max_seconds, " s worker ceiling including warmup and replay.")
    println("HSZ refinement worker log: ", log_path)
    flush(stdout)
    # Stage the numeric artifact so timeout/error cannot overwrite existing results.
    result = mktempdir(; prefix="ptd-hsz-refinement-staging-") do staging
        temporary_output = joinpath(staging, "refinement.jld2")
        command = _refinement_worker_command(baseline, temporary_output, points)
        outcome = _run_refinement_worker(command; max_seconds, log_path)
        _write_refinement_budget_status(status_path, outcome, request, log_path)
        outcome.status === :timed_out &&
            throw(HSZRefinementBudgetExceeded(max_seconds, log_path, status_path))
        outcome.status === :completed || error(
            "HSZ refinement worker failed (exit $(outcome.exit_code)); see $log_path. " *
            "Existing artifacts and tables were preserved.")
        isfile(temporary_output) || error("HSZ refinement worker produced no artifact; see $log_path")
        # Copy from scratch before the same-directory rename, so a failed copy
        # cannot remove the previous artifact even when /tmp is another volume.
        publication = Ptd._temporary_jld_path(output)
        try
            cp(temporary_output, publication; force=true)
            mv(publication, output; force=true)
        finally
            isfile(publication) && rm(publication)
        end
        outcome
    end
    composed = Ptd.HSZ.compose_hsz_grid_refinement(baseline,
        refinement_paths; table_path, include_timings)
    println("HSZ refinement artifact: ", output)
    _print_refinement_rows(composed)
    println("HSZ refinement budget diagnostic: ", status_path)

    return (; output, table=composed.table_path, status=result.status,
        status_path, log_path)
end

"""Dispatch HSZ accuracy, single-thread timing, or optional grid refinement."""
function main(args=ARGS)
    if !isempty(args) && first(args) == "--timing-only"
        return _hsz_path_timing_main(args[2:end])
    elseif !isempty(args) && first(args) == "--fixed-timing"
        return Ptd.FixedCalibrationTiming.main(args[2:end]; panels=(:hsz,))
    elseif !isempty(args) && first(args) == "--refinement"
        return _hsz_refinement_main(args[2:end])
    end
    return _hsz_accuracy_main(args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
