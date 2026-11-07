"""Generate the AR(1) figures, accuracy tables, and timing summary in Section 4.4.2."""

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using DrWatson
@quickactivate "Ptd"

using Dates
using JLD2
using LinearAlgebra
using Ptd
using SHA

function _ar1_preference_timing_authenticate(payload)
    return merge(payload, (; authentication_fingerprint=Ptd.stable_fingerprint(payload)))
end

function _ar1_preference_timing_write(path, artifact; replace::Bool=false)
    destination = Ptd._validate_data_output(path)
    !replace && (ispath(destination) || islink(destination)) &&
        throw(ArgumentError("refusing to overwrite AR(1) timing artifact: $destination"))
    mkpath(dirname(destination))
    temporary = Ptd._temporary_jld_path(destination)
    try
        JLD2.jldsave(temporary; artifact)
        open(temporary, "r") do io
            ccall(:fsync, Cint, (Cint,), fd(io)) == 0 || error("could not sync AR(1) timing file")
        end
        !replace && (ispath(destination) || islink(destination)) &&
            throw(ArgumentError("AR(1) timing destination appeared during measurement"))
        Base.rename(temporary, destination)
    finally
        isfile(temporary) && rm(temporary)
    end
    return destination
end

function _ar1_preference_timing_checkpoint(path, configuration, accuracy,
    source_files, sessions, measurements)
    source_files == Ptd.AR1._ar1_preference_timing_sources() ||
        error("AR(1) timing source inputs changed during measurement")
    payload = (; schema=Ptd.AR1.AR1_PREFERENCE_TIMING_CHECKPOINT_SCHEMA,
        configuration, accuracy, source_files, sessions, measurements)
    return _ar1_preference_timing_write(path,
        _ar1_preference_timing_authenticate(payload); replace=true)
end

function _measure_ar1_preference_path(method, gamma, configuration, grids)
    model = StochasticPTDConfig(beta=configuration.beta, gamma=gamma,
        return_rate=configuration.return_rate, cash_floor=configuration.cash_floor,
        terminal_date=configuration.horizon)
    process = LogAR1Process(rho=configuration.rho, sigma=configuration.sigma)
    rule = gauss_hermite_rule(process, configuration.quadrature_nodes)
    initial = ObservedState(1, configuration.initial_cash,
        LogAR1State(configuration.initial_log_income))
    options = Ptd.AR1.production_run_config()
    GC.gc()
    started = time_ns()
    policy = build_ar1_contest_policy(method, model, process; base_rule=rule,
        g2egm_grids=grids, g2egm_options=options.g2egm_options,
        ptd_osa_options=options.ptd_osa_options)
    construction_seconds = (time_ns() - started) / 1e9
    validate_ar1_contest_policy_alignment(policy, method, model, process)
    started = time_ns()
    evaluation = evaluate_policy_paths(policy, method, model, process,
        initial, configuration.innovations)
    evaluation_seconds = (time_ns() - started) / 1e9
    Ptd.AR1._ar1_preference_timing_require(evaluation.method === method &&
        length(evaluation.utility) == 1 && all(isfinite, evaluation.utility) &&
        all(isfinite, evaluation.total_transfers) && all(>=(0), evaluation.floor_hits),
        "invalid AR(1) single-path evaluation")
    return (; construction_seconds, evaluation_seconds,
        total_seconds=construction_seconds + evaluation_seconds,
        utility=only(evaluation.utility))
end

function _ar1_preference_timing_progress(parts...)
    println(string(now(UTC)), " UTC ", parts...)
    flush(stdout)
end

"""
Time one fresh solve and one common path at gamma=1,1.5,2,2.5,3, beta=.98,
and initial cash=16. One full discarded log-utility warmup precedes each
method's fresh measured solves. Summaries average over preference settings,
not repeated runs.
Resume accepts only unchanged calibration, source files, parent accuracy data,
and execution environment. Each completed sample is checkpointed atomically.
"""
function run_ar1_preference_timings(accuracy_path::AbstractString;
    output_path::AbstractString, resume::Bool=false, smoke::Bool=false,
    require_production::Bool=!smoke)
    require = Ptd.AR1._ar1_preference_timing_require
    require(Threads.nthreads() == BLAS.get_num_threads() == 1,
        "AR(1) preference timings require one Julia thread and one BLAS thread")
    source_path, destination = abspath(accuracy_path), Ptd._validate_data_output(output_path)
    checkpoint_path = destination * ".partial.jld2"
    require(source_path != destination && source_path != checkpoint_path &&
        !ispath(destination) && !islink(destination),
        "AR(1) timing output must be a new file distinct from its accuracy source")
    accuracy_sha = bytes2hex(SHA.sha256(read(source_path)))
    accuracy = Ptd.AR1.load_ar1_contest_welfare(source_path)
    configuration = Ptd.AR1._ar1_preference_timing_configuration(accuracy;
        smoke, require_production)
    accuracy_identity = (; source_path=Ptd.portable_path(source_path),
        file_sha256=accuracy_sha,
        authentication_fingerprint=accuracy["metadata"].authentication_fingerprint,
        result_fingerprint=accuracy["metadata"].result_fingerprint)
    sources = Ptd.AR1._ar1_preference_timing_sources()
    measurements, sessions = NamedTuple[], NamedTuple[]
    if resume
        require(isfile(checkpoint_path), "AR(1) preference timing checkpoint does not exist")
        checkpoint = JLD2.load(checkpoint_path, "artifact")
        payload = (; (name => getproperty(checkpoint, name) for name in keys(checkpoint)
            if name !== :authentication_fingerprint)...)
        require(checkpoint.schema == Ptd.AR1.AR1_PREFERENCE_TIMING_CHECKPOINT_SCHEMA &&
            Ptd.stable_fingerprint(payload) == checkpoint.authentication_fingerprint,
            "AR(1) preference timing checkpoint authentication differs")
        require(checkpoint.configuration == configuration &&
            checkpoint.accuracy == accuracy_identity && checkpoint.source_files == sources,
            "AR(1) preference timing checkpoint inputs changed")
        append!(measurements, checkpoint.measurements)
        append!(sessions, checkpoint.sessions)
        Ptd.AR1._validate_ar1_preference_timing_samples(measurements, configuration; complete=false)
        Ptd.AR1._validate_ar1_preference_timing_sessions(sessions, measurements)
    else
        require(!ispath(checkpoint_path) && !islink(checkpoint_path),
            "AR(1) preference checkpoint exists; use --resume or another output path")
    end
    push!(sessions, Ptd.AR1._ar1_preference_timing_session())
    Ptd.AR1._validate_ar1_preference_timing_sessions(sessions, measurements)
    session_index = length(sessions)
    _ar1_preference_timing_checkpoint(checkpoint_path, configuration,
        accuracy_identity, sources, sessions, measurements)
    order = Ptd.AR1._ar1_preference_timing_order(configuration)
    pending = order[length(measurements)+1:end]
    grids = Ptd.AR1._ar1_preference_timing_grid(; smoke)
    for method in configuration.methods
        any(identity -> identity.method === method, pending) || continue
        for gamma in configuration.warmup_gammas
            _ar1_preference_timing_progress("Warmup gamma=", gamma, " / ", method)
            sample = _measure_ar1_preference_path(method, gamma, configuration, grids)
            _ar1_preference_timing_progress("Completed discarded warmup in ", sample.total_seconds, " s")
        end
    end
    for identity in pending
        _ar1_preference_timing_progress("Timing agent ", identity.agent_index,
            "/", configuration.n_agents, ", gamma=", identity.gamma, " / ", identity.method)
        sample = _measure_ar1_preference_path(identity.method, identity.gamma, configuration, grids)
        push!(measurements, merge(identity, (; beta=configuration.beta,
            initial_cash=configuration.initial_cash,
            shock_fingerprint=configuration.shock_fingerprint, session_index), sample))
        Ptd.AR1._validate_ar1_preference_timing_samples(measurements, configuration; complete=false)
        _ar1_preference_timing_checkpoint(checkpoint_path, configuration,
            accuracy_identity, sources, sessions, measurements)
        _ar1_preference_timing_progress("Completed in ", sample.total_seconds, " s")
    end
    require(sources == Ptd.AR1._ar1_preference_timing_sources(),
        "AR(1) timing source inputs changed during measurement")
    require(accuracy_sha == bytes2hex(SHA.sha256(read(source_path))),
        "AR(1) accuracy artifact changed during timing")
    payload = (; schema=Ptd.AR1.AR1_PREFERENCE_TIMING_SCHEMA, configuration,
        accuracy=accuracy_identity, source_files=sources,
        source_fingerprint=Ptd.stable_fingerprint(sources), sessions, measurements,
        rows=Ptd.AR1._ar1_preference_timing_rows(measurements, configuration))
    artifact = _ar1_preference_timing_authenticate(payload)
    Ptd.AR1._validate_ar1_preference_timing_artifact(artifact, accuracy;
        require_production, require_current_source=true)
    _ar1_preference_timing_write(destination, artifact)
    _ar1_preference_timing_progress("AR(1) preference timing artifact: ", destination)
    for row in artifact.rows
        println(row.method, ": mean ", row.mean_total_seconds,
            " s; median ", row.median_total_seconds, " s across ", row.n_agents, " agents")
    end
    return artifact
end

"""Generated AR(1) artifact paths, figure results, and compact table path."""
struct AR1PaperOutputs{F}
    contest_path::String
    exhibit_path::String
    paths_artifact::String
    figures::F
    table_path::String
    ce_table_path::String
end

"""Separate the optional trajectory-artifact flag from the positional CLI arguments."""
function ar1_path_arguments(args)
    positional = String[]
    paths_artifact = projectdir("data", "ar1-updated-paths.jld2")
    seen_path = false
    index = 1
    while index <= length(args)
        if args[index] == "--paths-artifact"
            !seen_path && index < length(args) || error(
                "--paths-artifact requires one path and may appear only once.")
            paths_artifact = abspath(args[index + 1])
            seen_path = true
            index += 2
        else
            push!(positional, args[index])
            index += 1
        end
    end
    return positional, paths_artifact
end

function ar1_timing_arguments(args)
    positional = String[]
    timing_path = nothing
    seen = false
    index = 1
    while index <= length(args)
        if args[index] == "--timing-artifact"
            !seen && index < length(args) || error(
                "--timing-artifact requires one path and may appear only once.")
            timing_path = abspath(args[index + 1])
            seen = true
            index += 2
        else
            push!(positional, args[index])
            index += 1
        end
    end
    return positional, timing_path
end

"""Separate timing-run options from the other AR(1) arguments."""
function ar1_timing_run_arguments(args)
    positional = String[]
    timing_output = nothing
    resume, smoke = false, false
    index = 1
    while index <= length(args)
        flag = args[index]
        if flag == "--timing-output"
            isnothing(timing_output) && index < length(args) &&
                !startswith(args[index + 1], "--") || error(
                "--timing-output requires one path and may appear only once.")
            timing_output = abspath(args[index + 1])
            index += 2
        elseif flag == "--resume"
            !resume || error("--resume may appear only once")
            resume = true
            index += 1
        elseif flag == "--smoke"
            !smoke || error("--smoke may appear only once")
            smoke = true
            index += 1
        else
            push!(positional, flag)
            index += 1
        end
    end
    return positional, timing_output, resume, smoke
end

"""Run or render the AR(1) paper exercise according to `args`."""
function main(args=ARGS)
    if !isempty(args) && first(args) == "--fixed-timing"
        return Ptd.FixedCalibrationTiming.main(args[2:end]; panels=(:ar1,))
    end
    args, timing_output, resume, smoke = ar1_timing_run_arguments(args)
    args, timing_path = ar1_timing_arguments(args)
    if !isempty(args) && first(args) == "--timing-only"
        length(args) == 2 && !startswith(args[2], "--") && isnothing(timing_path) || error(
            "usage: 4_ar1.jl --timing-only ACCURACY_ARTIFACT [--timing-output PATH] [--resume] [--smoke]")
        BLAS.set_num_threads(1)
        output_path = isnothing(timing_output) ? projectdir("data", "ar1-preference-timings.jld2") : timing_output
        return run_ar1_preference_timings(args[2]; output_path, resume, smoke)
    end
    isnothing(timing_output) && !resume && !smoke || error(
        "--timing-output, --resume, and --smoke require --timing-only")
    positional, paths_artifact = ar1_path_arguments(args)
    mode = isempty(positional) ? :production :
        first(positional) == "--render-only" ? :render :
        first(positional) == "--update-paths" ? :update_paths : :invalid
    mode !== :invalid || error(
        "usage: 4_ar1.jl [--render-only|--update-paths CONTEST_ARTIFACT EXHIBIT_ARTIFACT [GRAPH_DIRECTORY] [TABLE_PATH]] [--paths-artifact PATH] [--timing-artifact PATH]")
    BLAS.set_num_threads(1)
    BLAS.get_num_threads() == 1 || error(
        "The AR(1) paper exercise requires one BLAS thread.")
    if mode !== :production
        3 <= length(positional) <= 5 || error(
            "usage: 4_ar1.jl --render-only|--update-paths CONTEST_ARTIFACT EXHIBIT_ARTIFACT [GRAPH_DIRECTORY] [TABLE_PATH] [--paths-artifact PATH] [--timing-artifact PATH]")
        contest_path = abspath(positional[2])
        exhibit_path = abspath(positional[3])
    else
        contest = Ptd.AR1.run_production_contest()
        contest_path = contest.publication.path
        exhibit_config = Ptd.AR1.production_ar1_contest_exhibit_config()
        exhibit = Ptd.AR1.run_ar1_contest_exhibits(exhibit_config;
            output_path=Ptd.AR1.default_ar1_contest_exhibit_artifact_path(
                exhibit_config),
            warmup=true,
            require_publication=true,
            compress=true)
        exhibit_path = exhibit.path
    end
    if !isnothing(timing_path)
        Ptd.AR1._ar1_paper_timing_summary(timing_path,
            Ptd.AR1.load_ar1_contest_welfare(contest_path); accuracy_path=contest_path)
    end
    graph_directory = mode !== :production && length(positional) >= 4 ?
        abspath(positional[4]) : Ptd.graphs_dir()
    table_path = mode !== :production && length(positional) == 5 ?
        abspath(positional[5]) : Ptd.tables_dir("ar1_contest_summary.tex")
    paths_artifact in (contest_path, exhibit_path) && error(
        "The updated path artifact must be separate from the contest and surface artifacts.")
    if mode !== :render
        Ptd.AR1.recompute_ar1_contest_paths(contest_path;
            output_path=paths_artifact,
            workers=min(Threads.nthreads(), 5),
            overwrite=true,
            solver_update_reason="Recompute the original heterogeneous Figure 9 configurations after the Julia 1.12.7 and EGM solver corrections.")
    end
    isfile(paths_artifact) || error(
        "The heterogeneous path artifact is missing: $paths_artifact. Run --update-paths first.")
    figures = Ptd.AR1.render_ar1_paper_figures(
        contest_path, exhibit_path; graph_directory,
        updated_paths_path=paths_artifact,
        require_current_path_source=mode !== :render)
    table = Ptd.AR1.render_ar1_paper_table(
        contest_path, exhibit_path; output_path=table_path,
        include_sample_note=false, timing_artifact_path=timing_path)
    ce_table = Ptd.AR1.render_ar1_ce_gap_table(contest_path;
        output_path=joinpath(dirname(table_path), "ar1_contest_ce_gaps.tex"))
    println("Single-thread timing artifact: ", isnothing(timing_path) ? "not reported" : timing_path)
    println("AR(1) contest artifact: ", contest_path)
    println("AR(1) exhibit artifact: ", exhibit_path)
    println("AR(1) policy surface: ", figures.surface_path)
    println("AR(1) path figure: ", figures.paths_path)
    println("AR(1) updated heterogeneous path artifact: ", paths_artifact)
    println("AR(1) path fingerprint: ", figures.replay_fingerprint)
    for configuration in figures.selected_configurations
        println("  contest ", configuration.contest_id,
            ": beta=", configuration.beta, ", gamma=", configuration.gamma,
            ", x1=", configuration.initial_cash)
    end
    println("AR(1) paper table: ", table)
    println("AR(1) CEG table: ", ce_table)
    return AR1PaperOutputs(contest_path, exhibit_path, paths_artifact, figures, table, ce_table)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
