"""Generate the IID accuracy and single-path timing tables in Section 4.4.1."""

using DrWatson
@quickactivate "Ptd"

using Dates
using JLD2
using LinearAlgebra
using Ptd
using SHA
using Statistics

function _iid_random_timing_sources()
    return Ptd.IID._computational_input_tree(pkgdir(Ptd)).files
end

function _iid_random_timing_session()
    return (; started_utc=string(now(UTC)), julia_version=string(VERSION),
        julia_threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(),
        cpu_name=Sys.CPU_NAME, machine=Sys.MACHINE, kernel=string(Sys.KERNEL))
end

function _iid_random_timing_write(path, artifact; replace::Bool=false)
    destination = Ptd._validate_data_output(path)
    !replace && (ispath(destination) || islink(destination)) &&
        throw(ArgumentError("refusing to overwrite existing IID timing artifact: $destination"))
    mkpath(dirname(destination))
    temporary = Ptd._temporary_jld_path(destination)
    try
        JLD2.jldsave(temporary; artifact)
        !replace && (ispath(destination) || islink(destination)) &&
            throw(ArgumentError("IID timing destination appeared during measurement"))
        Base.rename(temporary, destination)
    finally
        isfile(temporary) && rm(temporary)
    end
    return destination
end

function _iid_random_timing_authenticate(payload)
    return merge(payload, (; authentication_fingerprint=Ptd.stable_fingerprint(payload)))
end

function _iid_random_timing_checkpoint(path, configuration, accuracy_identity, sources, sessions, measurements)
    payload = (; schema=Ptd.IID.IID_RANDOM_TIMING_CHECKPOINT_SCHEMA, configuration,
        accuracy=accuracy_identity, source_files=sources, sessions, measurements)
    return _iid_random_timing_write(path, _iid_random_timing_authenticate(payload); replace=true)
end

"""Time fresh construction and one path, excluding setup and alignment validation."""
function _measure_iid_random_path(method, inputs, assets, options, shocks)
    started = time_ns()
    policy = Ptd.build_iid_contest_policy(method, inputs.model_config, inputs.process;
        base_rule=inputs.base_rule, asset_grid=method === :dcegm ? assets : nothing,
        ptd_osa_options=options)
    construction_seconds = (time_ns() - started) / 1e9
    Ptd.validate_iid_contest_policy_alignment(policy, method, inputs.model_config, inputs.process)
    started = time_ns()
    evaluation = Ptd.evaluate_policy_paths(policy, method, inputs.model_config,
        inputs.process, inputs.initial, shocks)
    evaluation_seconds = (time_ns() - started) / 1e9
    Ptd.IID._iid_timing_require(evaluation.method === method && length(evaluation.utility) == 1 &&
        all(isfinite, evaluation.utility) && all(isfinite, evaluation.total_transfers) &&
        all(>=(0), evaluation.floor_hits), "invalid IID single-path evaluation")
    return (; construction_seconds, evaluation_seconds,
        total_seconds=construction_seconds + evaluation_seconds, utility=only(evaluation.utility))
end

"""
Measure every saved accuracy agent once in each IID variance panel. The first
path of each original hidden shock block is reused across methods; its utility
must reproduce the saved accuracy result. Warmup and checkpoint I/O are untimed;
automatic GC during construction/evaluation remains inside the measured clock.
An authenticated checkpoint allows interrupted measurements to resume without
reusing any policy. Existing accuracy and completed timing files are never overwritten.
"""
function run_iid_random_timings(accuracy_path::AbstractString;
    output_path::AbstractString, resume::Bool=false, require_production::Bool=true)
    require = Ptd.IID._iid_timing_require
    require(Threads.nthreads() == 1 && BLAS.get_num_threads() == 1,
        "IID random timings require one Julia thread and one BLAS thread")
    source_path, destination = abspath(accuracy_path), Ptd._validate_data_output(output_path)
    require(source_path != destination && !ispath(destination) && !islink(destination),
        "IID timing output must be a new file distinct from its accuracy source")
    checkpoint_path = destination * ".partial.jld2"
    require(source_path != checkpoint_path, "IID checkpoint cannot overwrite the accuracy source")
    accuracy_sha = bytes2hex(sha256(read(source_path)))
    accuracy = Ptd.IID.load_iid_contest(source_path)
    config = Ptd.IID._iid_timing_run_config(accuracy; require_production)
    configuration = Ptd.IID._iid_timing_configuration(accuracy, config)
    accuracy_identity = (; source_path=Ptd.portable_path(source_path),
        file_sha256=accuracy_sha,
        authentication_fingerprint=accuracy["metadata"].authentication_fingerprint,
        result_fingerprint=accuracy["metadata"].result_fingerprint)
    sources = _iid_random_timing_sources()
    measurements, sessions = NamedTuple[], NamedTuple[]
    if resume
        require(isfile(checkpoint_path), "IID timing checkpoint does not exist")
        checkpoint = JLD2.load(checkpoint_path, "artifact")
        payload = (; (name => getproperty(checkpoint, name) for name in keys(checkpoint)
            if name !== :authentication_fingerprint)...)
        require(checkpoint.schema == Ptd.IID.IID_RANDOM_TIMING_CHECKPOINT_SCHEMA &&
            Ptd.stable_fingerprint(payload) == checkpoint.authentication_fingerprint,
            "IID timing checkpoint authentication differs")
        require(checkpoint.configuration == configuration && checkpoint.accuracy == accuracy_identity &&
            checkpoint.source_files == sources, "IID timing checkpoint inputs changed")
        append!(measurements, checkpoint.measurements)
        append!(sessions, checkpoint.sessions)
        Ptd.IID._validate_iid_timing_samples(measurements, accuracy, configuration; complete=false)
    else
        require(!ispath(checkpoint_path) && !islink(checkpoint_path),
            "IID timing checkpoint exists; use --resume or another output path")
    end
    session = _iid_random_timing_session()
    if !isempty(sessions)
        previous = last(sessions)
        require(all(name -> getproperty(previous, name) == getproperty(session, name),
            (:julia_version, :julia_threads, :blas_threads, :cpu_name, :machine, :kernel)),
            "IID timing checkpoint execution environment changed")
    end
    push!(sessions, session)
    session_index = length(sessions)
    _iid_random_timing_checkpoint(checkpoint_path, configuration, accuracy_identity, sources, sessions, measurements)
    assets = Ptd.IID._asset_grid(config)
    order = Ptd.IID._iid_timing_order(configuration)
    for panel in configuration.panels
        pending = filter(index -> order[index].panel === panel, length(measurements)+1:length(order))
        isempty(pending) && continue
        first_index = order[first(pending)].accuracy_index
        plain = configuration.specs[first_index]
        spec = Ptd.ContestSpec(plain.id, plain.beta, plain.gamma, plain.initial_cash)
        inputs = Ptd.IID._model_inputs(config, spec, getproperty(config.panels, panel))
        shocks = only(filter(row -> row.panel === panel && row.accuracy_index == first_index,
            configuration.shock_records)).shocks
        for method in configuration.methods
            println("IID random timing warmup: ", panel, " / ", method)
            flush(stdout)
            sample = _measure_iid_random_path(method, inputs, assets, config.ptd_osa_options, shocks)
            panel_index = findfirst(==(panel), configuration.panels)
            method_index = findfirst(==(method), configuration.methods)
            reference = accuracy["pathwise_utilities"][1, method_index, first_index, panel_index]
            require(abs(sample.utility - reference) <= configuration.utility_rtol * max(1.0, abs(reference)),
                "IID warmup differs from saved accuracy utility")
        end
        GC.gc()
        for index in pending
            identity = order[index]
            plain = configuration.specs[identity.accuracy_index]
            spec = Ptd.ContestSpec(plain.id, plain.beta, plain.gamma, plain.initial_cash)
            inputs = Ptd.IID._model_inputs(config, spec, getproperty(config.panels, panel))
            shocks = only(filter(row -> row.panel === panel && row.accuracy_index == identity.accuracy_index,
                configuration.shock_records)).shocks
            println("IID random timing: ", panel, " agent ", identity.accuracy_index, "/", config.n_contests,
                " / ", identity.method)
            flush(stdout)
            sample = _measure_iid_random_path(identity.method, inputs, assets, config.ptd_osa_options, shocks)
            push!(measurements, merge(identity, (; spec_id=spec.id, path_index=1, session_index), sample))
            Ptd.IID._validate_iid_timing_samples(measurements, accuracy, configuration; complete=false)
            _iid_random_timing_checkpoint(checkpoint_path, configuration, accuracy_identity, sources, sessions, measurements)
            println("Completed in ", sample.total_seconds, " seconds")
            flush(stdout)
        end
    end
    require(sources == _iid_random_timing_sources(), "IID timing source files changed during measurement")
    require(accuracy_sha == bytes2hex(sha256(read(source_path))), "IID accuracy artifact changed during timing")
    payload = (; schema=Ptd.IID.IID_RANDOM_TIMING_SCHEMA, configuration, accuracy=accuracy_identity,
        source_files=sources, source_fingerprint=Ptd.stable_fingerprint(sources), sessions,
        measurements, rows=Ptd.IID._iid_timing_rows(measurements, configuration))
    artifact = _iid_random_timing_authenticate(payload)
    Ptd.IID._validate_iid_random_timing_artifact(artifact, accuracy; require_production)
    _iid_random_timing_write(destination, artifact)
    println("IID random timing artifact: ", destination)
    for row in artifact.rows
        println(row.panel, " / ", row.method, ": mean ", row.mean_seconds,
            " s; median ", row.median_seconds, " s (", row.n_agents, " agents)")
    end
    return artifact
end

"""Locations of the IID simulation artifact and rendered paper table."""
struct IIDPaperOutputs
    artifact_path::String
    table_path::String
    ce_table_path::String
end

function iid_timing_arguments(args)
    positional = String[]
    timing_path = nothing
    timing_output = nothing
    resume = false
    seen = false
    index = 1
    while index <= length(args)
        if args[index] == "--timing-artifact"
            !seen && index < length(args) || error(
                "--timing-artifact requires one path and may appear only once.")
            timing_path = abspath(args[index + 1])
            seen = true
            index += 2
        elseif args[index] == "--timing-output"
            isnothing(timing_output) && index < length(args) || error(
                "--timing-output requires one path and may appear only once.")
            timing_output = abspath(args[index + 1])
            index += 2
        elseif args[index] == "--resume"
            !resume || error("--resume may appear only once")
            resume = true
            index += 1
        else
            push!(positional, args[index])
            index += 1
        end
    end
    return positional, timing_path, timing_output, resume
end

"""Run or render the IID paper exercise according to `args`."""
function main(args=ARGS)
    if !isempty(args) && first(args) == "--fixed-timing"
        return Ptd.FixedCalibrationTiming.main(args[2:end]; panels=(:iid_baseline, :iid_high_variance))
    end
    args, timing_path, timing_output, resume = iid_timing_arguments(args)
    timing_only = !isempty(args) && first(args) == "--timing-only"
    if timing_only
        length(args) == 2 && isnothing(timing_path) || error(
            "usage: 3_iid.jl --timing-only ACCURACY_ARTIFACT [--timing-output PATH] [--resume]")
        BLAS.set_num_threads(1)
        output_path = isnothing(timing_output) ? projectdir("data", "iid-random-timings.jld2") : timing_output
        return run_iid_random_timings(args[2]; output_path, resume)
    end
    isnothing(timing_output) && !resume || error("--timing-output and --resume require --timing-only")
    render_only = !isempty(args) && first(args) == "--render-only"
    if render_only
        2 <= length(args) <= 3 || error(
            "usage: 3_iid.jl --render-only ARTIFACT [TABLE_PATH] [--timing-artifact PATH]")
        artifact_path = abspath(args[2])
    else
        isempty(args) || error("usage: 3_iid.jl [--render-only ARTIFACT [TABLE_PATH]] [--timing-artifact PATH]")
        BLAS.set_num_threads(1)
        result = Ptd.IID.run_production_contest()
        artifact_path = result.publication.path
    end
    table_path = render_only && length(args) == 3 ? abspath(args[3]) :
        Ptd.tables_dir("iid_contest_summary.tex")
    rendered = Ptd.IID.render_iid_paper_table(
        artifact_path; output_path=table_path, include_sample_note=false,
        timing_artifact_path=timing_path)
    ce_table = Ptd.IID.render_iid_ce_gap_table(artifact_path;
        output_path=joinpath(dirname(table_path), "iid_contest_ce_gaps.tex"))
    println("Single-thread timing artifact: ", isnothing(timing_path) ? "not reported" : timing_path)
    println("IID artifact: ", artifact_path)
    println("IID paper table: ", rendered)
    println("IID CEG table: ", ce_table)
    return IIDPaperOutputs(artifact_path, rendered, ce_table)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
