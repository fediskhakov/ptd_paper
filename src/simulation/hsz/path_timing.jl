# Single-thread HSZ timings with one policy retained across common shock paths.
import ..Ptd: evaluate_policy_paths, plan_cache_statistics, portable_path

export run_hsz_path_timings, load_hsz_path_timing_summary

const HSZ_PATH_TIMING_SCHEMA = "ptd-hsz-path-timing-v1"
const HSZ_PATH_TIMING_CHECKPOINT_SCHEMA = "ptd-hsz-path-timing-checkpoint-v1"

_hsz_timing_require(condition, message) = condition || throw(ArgumentError(message))
_hsz_timing_sha(path) = bytes2hex(SHA.sha256(read(path)))

function _hsz_path_timing_configuration(accuracy; require_production=true,
    implementation::Symbol=:canonical)
    _hsz_timing_require(accuracy.config.gamma == 1.0, "HSZ timings require log utility")
    return _hsz_path_timing_configuration_from_inputs(
        _scientific_configuration(accuracy.config), accuracy.innovations;
        require_production, implementation)
end

function _hsz_path_timing_configuration_from_inputs(scientific, innovations;
    require_production=true, implementation::Symbol=:canonical)
    n_paths = size(innovations, 1)
    _hsz_timing_require(scientific.beta == 0.98 && scientific.utility === :log &&
        16.0 in scientific.paths.initial_cash_support,
        "HSZ timings require log utility, beta=.98, and initial cash 16")
    _hsz_timing_require(n_paths >= 1 && ndims(innovations) == 3 &&
        size(innovations) == (n_paths, scientific.horizon - 1, 2) && all(isfinite, innovations),
        "HSZ timing shock dimensions or values are invalid")
    require_production && _hsz_timing_require(
        scientific == _scientific_configuration(production_hsz_example_config()) &&
        n_paths == 100 && implementation === :canonical,
        "HSZ paper timings require the production calibration and 100 saved paths")
    _hsz_timing_require(implementation in (:canonical, :injected),
        "unknown HSZ timing implementation")
    return (; panel=:hsz, methods=HSZ_EXAMPLE_METHODS, production=require_production,
        implementation, scientific_configuration=scientific,
        beta=scientific.beta, gamma=1.0, initial_cash=16.0,
        horizon=scientific.horizon, n_paths, innovations=copy(innovations),
        shock_fingerprint=stable_fingerprint(innovations),
        julia_threads=1, blas_threads=1,
        warmup=:discarded_full_construction_and_one_path_per_method,
        warmups_per_method_per_session=1,
        reuse=:same_policy_and_cache_across_all_paths,
        path_order=:saved_accuracy_order,
        construction=:one_fresh_policy_after_warmup,
        total=:construction_plus_sum_of_path_times,
        summary=:mean_and_median_of_path_evaluation_times,
        inside_timers=(:construction, :path_evaluation, :automatic_gc),
        outside_timers=(:model_and_grid_setup, :warmup, :explicit_gc,
            :alignment_checks, :result_validation, :summaries, :file_io))
end

function _hsz_path_timing_sources()
    paths = String["Project.toml", "Manifest.toml", "src/Ptd.jl",
        "scripts/5_hsz.jl", "src/simulation/hsz/path_timing.jl",
        "src/simulation/hsz/example.jl", "src/simulation/hsz/utility_comparison.jl",
        "src/simulation/policies/hsz.jl", "src/simulation/common/contest_design.jl",
        "src/simulation/common/contest_evaluator.jl",
        "src/simulation/common/contest_statistics.jl",
        "src/simulation/common/reproducibility.jl"]
    for directory in ("src/ptd", "src/g2egm")
        for (root, _, names) in walkdir(joinpath(PROJECT_ROOT, directory))
            append!(paths, [relpath(joinpath(root, name), PROJECT_ROOT)
                for name in names if endswith(name, ".jl")])
        end
    end
    return [(; path, sha256=_hsz_timing_sha(joinpath(PROJECT_ROOT, path)))
        for path in sort!(unique(paths))]
end

function _hsz_path_timing_session()
    return (; started_utc=string(now(UTC)), julia_version=string(VERSION),
        julia_threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(),
        blas_vendor=string(BLAS.vendor()), cpu_name=Sys.CPU_NAME,
        machine=Sys.MACHINE, kernel=string(Sys.KERNEL))
end

function _validate_hsz_path_timing_sessions(sessions, measurements)
    fields = (:julia_version, :julia_threads, :blas_threads, :blas_vendor,
        :cpu_name, :machine, :kernel)
    _hsz_timing_require(!isempty(sessions) && all(session ->
        session.julia_threads == session.blas_threads == 1 &&
        all(name -> !isempty(getproperty(session, name)),
            (:started_utc, :julia_version, :blas_vendor, :cpu_name, :machine, :kernel)), sessions),
        "HSZ path timings require one Julia thread and one BLAS thread")
    _hsz_timing_require(all(session -> all(name -> getproperty(session, name) ==
        getproperty(first(sessions), name), fields), sessions),
        "HSZ timing execution environment changed across sessions")
    _hsz_timing_require(all(sample -> sample.session_index isa Integer &&
        1 <= sample.session_index <= length(sessions), measurements),
        "HSZ timing sample session is missing")
    return nothing
end

function _validate_hsz_path_timing_measurements(measurements, configuration, accuracy=nothing;
    complete=true)
    n = configuration.n_paths
    methods = configuration.methods
    _hsz_timing_require(complete ? length(measurements) == length(methods) :
        length(measurements) <= length(methods), "HSZ timing method coverage differs")
    columns = isnothing(accuracy) ? nothing : findall(==(configuration.initial_cash), accuracy.initial_cash)
    isnothing(accuracy) || _hsz_timing_require(length(columns) == n &&
        accuracy.shock_row[columns] == collect(1:n), "HSZ reference paths do not match the timing shock order")
    for (sample, method) in zip(measurements, methods)
        _hsz_timing_require(sample.method === method && sample.n_paths == n &&
            length(sample.evaluation_seconds) == length(sample.utility) ==
            length(sample.floor_hits) == length(sample.total_transfers) == n,
            "HSZ timing sample method or path count differs")
        _hsz_timing_require(isfinite(sample.construction_seconds) && sample.construction_seconds >= 0 &&
            all(t -> isfinite(t) && t >= 0, sample.evaluation_seconds) &&
            sample.total_seconds == sample.construction_seconds + sum(sample.evaluation_seconds),
            "HSZ timing durations are invalid")
        _hsz_timing_require(all(isfinite, sample.utility) && all(isfinite, sample.total_transfers) &&
            all(>=(0), sample.floor_hits) && all(>=(0), sample.total_transfers),
            "HSZ timed replay contains invalid outcomes")
        if !isnothing(accuracy)
            reference = accuracy.paths[method]
            _hsz_timing_require(
                all(isapprox.(sample.utility, reference.utility[columns]; atol=1e-9, rtol=1e-11)) &&
                sample.floor_hits == reference.floor_hits[columns] &&
                all(isapprox.(sample.total_transfers, reference.total_transfers[columns]; atol=1e-9, rtol=1e-11)),
                "HSZ timed replay differs from the saved accuracy paths")
        end
        stats = sample.cache_statistics
        if method !== :g2egm && configuration.implementation === :canonical
            _hsz_timing_require(!isnothing(stats) &&
                all(name -> getproperty(stats, name) isa Integer && getproperty(stats, name) >= 0,
                    (:entries, :hits, :misses, :builds)) && stats.entries == stats.misses == stats.builds,
                "HSZ retained-policy cache statistics are invalid")
        end
    end
    return nothing
end

function _hsz_path_timing_rows(measurements, configuration)
    return [(; panel=:hsz, method=sample.method, n_paths=configuration.n_paths,
        construction_seconds=sample.construction_seconds,
        mean_evaluation_seconds=mean(sample.evaluation_seconds),
        median_evaluation_seconds=median(sample.evaluation_seconds),
        evaluation_seconds=sum(sample.evaluation_seconds), total_seconds=sample.total_seconds)
        for sample in measurements]
end

_hsz_timing_payload(artifact) = (; (name => getproperty(artifact, name)
    for name in keys(artifact) if name !== :authentication_fingerprint)...)
_authenticate_hsz_path_timing(payload) = merge(payload,
    (; authentication_fingerprint=stable_fingerprint(payload)))

function _validate_hsz_path_timing_artifact(artifact, accuracy=nothing;
    require_production=true, require_current_source=false)
    _hsz_timing_require(artifact.schema == HSZ_PATH_TIMING_SCHEMA &&
        stable_fingerprint(_hsz_timing_payload(artifact)) == artifact.authentication_fingerprint,
        "HSZ path timing schema or authentication differs")
    require_production && _hsz_timing_require(artifact.configuration.production &&
        artifact.configuration.implementation === :canonical,
        "nonproduction timings cannot populate the HSZ paper table")
    expected = _hsz_path_timing_configuration_from_inputs(
        artifact.configuration.scientific_configuration, artifact.configuration.innovations;
        require_production=artifact.configuration.production,
        implementation=artifact.configuration.implementation)
    _hsz_timing_require(artifact.configuration == expected &&
        occursin(r"^[0-9a-f]{64}$", artifact.accuracy.file_sha256) &&
        occursin(r"^[0-9a-f]{64}$", artifact.accuracy.scientific_fingerprint),
        "HSZ timing calibration, shocks, protocol, or accuracy identity differs")
    isnothing(accuracy) || _hsz_timing_require(artifact.configuration ==
        _hsz_path_timing_configuration(accuracy; require_production=artifact.configuration.production,
            implementation=artifact.configuration.implementation) &&
        artifact.accuracy.scientific_fingerprint == accuracy.scientific_fingerprint,
        "HSZ timing calibration, shocks, protocol, or accuracy identity differs")
    sources = artifact.source_files
    paths = getproperty.(sources, :path)
    _hsz_timing_require(!isempty(paths) && length(unique(paths)) == length(paths) &&
        all(path -> !isabspath(path) && !(".." in splitpath(path)), paths) &&
        all(file -> occursin(r"^[0-9a-f]{64}$", file.sha256), sources) &&
        all(path -> path in paths, ("src/Ptd.jl",
            "src/simulation/hsz/path_timing.jl", "Project.toml", "Manifest.toml")) &&
        # Preserve recorded provenance from before script numbering and consolidation.
        count(path -> path in paths, ("scripts/5_hsz.jl", "scripts/12_hsz_path_timing.jl",
            "scripts/hsz_path_timing.jl")) == 1 &&
        artifact.source_fingerprint == stable_fingerprint(sources),
        "HSZ timing source inventory differs")
    require_current_source && _hsz_timing_require(sources == _hsz_path_timing_sources(),
        "HSZ timing computational sources changed")
    _validate_hsz_path_timing_sessions(artifact.sessions, artifact.measurements)
    _validate_hsz_path_timing_measurements(artifact.measurements, artifact.configuration, accuracy)
    _hsz_timing_require(artifact.rows == _hsz_path_timing_rows(artifact.measurements, artifact.configuration),
        "HSZ timing summaries differ from the raw measurements")
    return nothing
end

"""Authenticate retained-policy timings against their saved accuracy paths."""
function load_hsz_path_timing_summary(path::AbstractString, accuracy_path::Union{Nothing,AbstractString}=nothing;
    require_production=true, require_current_source=false)
    source_path = abspath(path)
    timing_sha = _hsz_timing_sha(source_path)
    accuracy_sha = isnothing(accuracy_path) ? nothing : _hsz_timing_sha(accuracy_path)
    accuracy = isnothing(accuracy_path) ? nothing : load_hsz_utility_comparison(accuracy_path)
    artifact = JLD2.load(source_path, "artifact")
    _validate_hsz_path_timing_artifact(artifact, accuracy; require_production, require_current_source)
    _hsz_timing_require(_hsz_timing_sha(source_path) == timing_sha &&
        (isnothing(accuracy_path) || (artifact.accuracy.file_sha256 == accuracy_sha &&
            _hsz_timing_sha(accuracy_path) == accuracy_sha)),
        "HSZ timing or parent accuracy file changed")
    return (; artifact, configuration=artifact.configuration,
        rows=Dict((row.panel, row.method) => row for row in artifact.rows),
        artifact_sha256=timing_sha, artifact_fingerprint=artifact.authentication_fingerprint,
        source_fingerprint=artifact.source_fingerprint, source_files=artifact.source_files,
        provenance_fingerprint=stable_fingerprint((; artifact.accuracy, artifact.sessions)), source_path)
end

function _write_hsz_path_timing(path, artifact; replace=false)
    destination = _validate_data_output(path)
    _hsz_timing_require(replace || !(ispath(destination) || islink(destination)),
        "refusing to overwrite HSZ timing artifact")
    mkpath(dirname(destination))
    temporary = _temporary_jld_path(destination)
    try
        JLD2.jldsave(temporary; artifact)
        open(temporary, "r") do io
            ccall(:fsync, Cint, (Cint,), fd(io)) == 0 || error("could not sync HSZ timing file")
        end
        _hsz_timing_require(replace || !(ispath(destination) || islink(destination)),
            "HSZ timing destination appeared during measurement")
        Base.rename(temporary, destination)
    finally
        isfile(temporary) && rm(temporary)
    end
    return destination
end

_hsz_path_cache_statistics(policy) = hasproperty(policy, :cache) ?
    plan_cache_statistics(policy.cache) : nothing

function _hsz_timing_progress(parts...)
    println(string(now(UTC)), " UTC ", parts...)
    flush(stdout)
end

"""
Time one fresh policy construction and consecutive paths sharing that policy.
Every method receives an excluded disposable full construction/one-path warmup.
Completed methods are checkpointed atomically; resumption starts at a method
boundary with unchanged sources, accuracy data and execution environment.
Injected helpers and tiny saved accuracy artifacts are supported only when
`require_production=false`, and are marked in the authenticated configuration.
"""
function run_hsz_path_timings(accuracy_path::AbstractString;
    output_path::AbstractString=joinpath(PROJECT_ROOT, "data", "hsz-path-timings.jld2"),
    resume=false, require_production=true,
    policy_builder=build_hsz_contest_policy, evaluator=evaluate_policy_paths,
    alignment_validator=validate_hsz_contest_policy_alignment,
    cache_reader=_hsz_path_cache_statistics)
    _hsz_timing_require(Threads.nthreads() == BLAS.get_num_threads() == 1,
        "HSZ path timings require one Julia thread and one BLAS thread")
    implementation = policy_builder === build_hsz_contest_policy && evaluator === evaluate_policy_paths &&
        alignment_validator === validate_hsz_contest_policy_alignment &&
        cache_reader === _hsz_path_cache_statistics ? :canonical : :injected
    source_path, destination = abspath(accuracy_path), abspath(output_path)
    checkpoint_path = destination * ".partial.jld2"
    _validate_utility_render_paths(source_path, (destination, checkpoint_path))
    _hsz_timing_require(!(ispath(destination) || islink(destination)),
        "HSZ timing output must be a new file")
    accuracy_sha = _hsz_timing_sha(source_path)
    accuracy = load_hsz_utility_comparison(source_path)
    configuration = _hsz_path_timing_configuration(accuracy; require_production, implementation)
    accuracy_identity = (; source_path=portable_path(source_path),
        file_sha256=accuracy_sha,
        scientific_fingerprint=accuracy.scientific_fingerprint)
    sources = _hsz_path_timing_sources()
    measurements, sessions = NamedTuple[], NamedTuple[]
    if resume
        _hsz_timing_require(isfile(checkpoint_path), "HSZ timing checkpoint does not exist")
        checkpoint = JLD2.load(checkpoint_path, "artifact")
        _hsz_timing_require(checkpoint.schema == HSZ_PATH_TIMING_CHECKPOINT_SCHEMA &&
            stable_fingerprint(_hsz_timing_payload(checkpoint)) == checkpoint.authentication_fingerprint &&
            checkpoint.configuration == configuration && checkpoint.accuracy == accuracy_identity &&
            checkpoint.source_files == sources, "HSZ timing checkpoint authentication or inputs changed")
        append!(measurements, checkpoint.measurements)
        append!(sessions, checkpoint.sessions)
        _validate_hsz_path_timing_measurements(measurements, configuration, accuracy; complete=false)
    else
        _hsz_timing_require(!(ispath(checkpoint_path) || islink(checkpoint_path)),
            "HSZ timing checkpoint exists; use --resume or another output path")
    end
    push!(sessions, _hsz_path_timing_session())
    _validate_hsz_path_timing_sessions(sessions, measurements)
    session_index = length(sessions)
    function checkpoint!()
        _hsz_timing_require(sources == _hsz_path_timing_sources() && accuracy_sha == _hsz_timing_sha(source_path),
            "HSZ timing source inputs changed during measurement")
        payload = (; schema=HSZ_PATH_TIMING_CHECKPOINT_SCHEMA, configuration,
            accuracy=accuracy_identity, source_files=sources, sessions, measurements)
        _write_hsz_path_timing(checkpoint_path, _authenticate_hsz_path_timing(payload); replace=true)
    end
    checkpoint!()
    model, process = _model(accuracy.config)
    rule, grid = gauss_hermite_rule(process, 5), _g2_grid(accuracy.config, process)
    initial = ObservedState(1, 16.0,
        HSZState(process.unconditional_log_earnings, process.unconditional_log_medical))
    shocks = [view(configuration.innovations, path:path, :, :) for path in 1:configuration.n_paths]
    for method in configuration.methods[length(measurements)+1:end]
        _hsz_timing_progress("Warmup / ", method)
        warm_policy = policy_builder(method, model, process; base_rule=rule, g2egm_grid=grid)
        alignment_validator(warm_policy, method, model, process)
        evaluator(warm_policy, method, model, process, initial, first(shocks))
        warm_policy = nothing
        GC.gc()
        _hsz_timing_progress("Construction / ", method)
        started = time_ns()
        policy = policy_builder(method, model, process; base_rule=rule, g2egm_grid=grid)
        construction_seconds = (time_ns() - started) / 1e9
        alignment_validator(policy, method, model, process)
        evaluation_seconds, utility, total_transfers = Float64[], Float64[], Float64[]
        floor_hits = Int[]
        for path in eachindex(shocks)
            started = time_ns()
            result = evaluator(policy, method, model, process, initial, shocks[path])
            elapsed = (time_ns() - started) / 1e9
            _hsz_timing_require(result.method === method && length(result.utility) == 1 &&
                length(result.floor_hits) == length(result.total_transfers) == 1,
                "HSZ evaluator must return exactly one path")
            push!(evaluation_seconds, elapsed)
            push!(utility, only(result.utility))
            push!(floor_hits, only(result.floor_hits))
            push!(total_transfers, only(result.total_transfers))
            path % 10 == 0 && _hsz_timing_progress(method, " completed ", path, "/", configuration.n_paths, " paths")
        end
        push!(measurements, (; method, n_paths=configuration.n_paths, session_index,
            construction_seconds, evaluation_seconds, utility, floor_hits, total_transfers,
            total_seconds=construction_seconds + sum(evaluation_seconds),
            cache_statistics=cache_reader(policy)))
        policy = nothing
        _validate_hsz_path_timing_measurements(measurements, configuration, accuracy; complete=false)
        checkpoint!()
        _hsz_timing_progress("Completed / ", method, "; total ", last(measurements).total_seconds, " s")
    end
    payload = (; schema=HSZ_PATH_TIMING_SCHEMA, configuration, accuracy=accuracy_identity,
        source_files=sources, source_fingerprint=stable_fingerprint(sources), sessions, measurements,
        rows=_hsz_path_timing_rows(measurements, configuration))
    artifact = _authenticate_hsz_path_timing(payload)
    _validate_hsz_path_timing_artifact(artifact, accuracy; require_production, require_current_source=true)
    _write_hsz_path_timing(destination, artifact)
    _hsz_timing_progress("HSZ path timing artifact: ", destination)
    return artifact
end
