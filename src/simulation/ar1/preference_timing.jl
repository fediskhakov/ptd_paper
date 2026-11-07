# Authenticated one-path timings for five selected AR(1) preference settings.
using JLD2
using Statistics: mean, median
import ..Ptd: _single_log_closed_grid, _shifted_log_closed_grid

export load_ar1_preference_timing_summary

const AR1_PREFERENCE_TIMING_SCHEMA = "ptd-ar1-preference-timing-v1"
const AR1_PREFERENCE_TIMING_CHECKPOINT_SCHEMA = "ptd-ar1-preference-timing-checkpoint-v1"
const AR1_PREFERENCE_TIMING_GAMMAS = (1.0, 1.5, 2.0, 2.5, 3.0)
const AR1_PREFERENCE_TIMING_SEED = UInt64(0x5054445f54494d49)

_ar1_preference_timing_require(condition, message) =
    condition || throw(ArgumentError(message))

function _ar1_preference_timing_grid(; smoke::Bool=false)
    smoke || return production_run_config().g2egm_grids
    return AR1G2ContestGridConfiguration(:preference_timing_smoke,
        _single_log_closed_grid(30_000.0, 101),
        collect(range(-3.0, 3.0; length=9)),
        _shifted_log_closed_grid(3.0, 33_000.0 + exp(3.0), 101),
        collect(range(-3.0, 3.0; length=9)))
end

function _ar1_preference_timing_configuration(accuracy;
    smoke::Bool=false, require_production::Bool=true,
    warmup_gammas::Tuple=(1.0,))
    # Preserve the first production artifact's recorded two-warmup protocol.
    _ar1_preference_timing_require(warmup_gammas in ((1.0,), (1.0, 1.5)),
        "unsupported AR(1) preference timing warmup settings")
    production = production_run_config()
    scientific = accuracy["scientific_configuration"]
    require_production && _ar1_preference_timing_require(
        scientific == scientific_configuration(production),
        "AR(1) paper timings require the production accuracy calibration")
    _ar1_preference_timing_require(Tuple(accuracy["methods"]) == REQUIRED_METHODS,
        "AR(1) accuracy methods differ from timing methods")
    horizon = smoke ? 3 : production.horizon
    # This is the same seeded path used by the historical fixed-log timing case.
    shocks = randn(Xoshiro(AR1_PREFERENCE_TIMING_SEED), 1, horizon - 1)
    return (; smoke, accuracy_scientific_configuration=scientific,
        panel=:ar1, methods=REQUIRED_METHODS, beta=0.98,
        gammas=AR1_PREFERENCE_TIMING_GAMMAS, initial_cash=16.0,
        initial_log_income=0.0, n_agents=length(AR1_PREFERENCE_TIMING_GAMMAS),
        horizon, return_rate=production.return_rate, cash_floor=production.cash_floor,
        rho=production.rho, sigma=production.sigma,
        unconditional_log_mean=production.unconditional_log_mean,
        quadrature_nodes=production.quadrature_nodes_per_dimension,
        grid=_plain_ar1_g2_grid_configuration(_ar1_preference_timing_grid(; smoke)),
        solver_options=_plain_g2egm_options(production.g2egm_options),
        ptd_osa_options=scientific_configuration(production).ptd_osa_options,
        seed=AR1_PREFERENCE_TIMING_SEED, innovations=shocks,
        shock_fingerprint=stable_fingerprint(shocks),
        julia_threads=1, blas_threads=1, paths_per_agent=1,
        measured_runs_per_agent=1, selection=:five_selected_preference_settings,
        path_selection=:one_common_seeded_path_across_agents_and_methods,
        method_order=:rotating_by_agent,
        warmup=:discarded_full_construction_and_path_per_method_at_each_warmup_gamma,
        warmup_gammas,
        warmups_per_method_per_measured_session=length(warmup_gammas),
        cache=:fresh_policy_each_agent_method,
        total=:sum_of_separately_measured_construction_and_path,
        summary=:mean_and_median_across_five_preference_settings,
        inside_timers=(:fresh_policy_construction, :full_backward_solve,
            :one_path_evaluation, :automatic_gc),
        outside_timers=(:model_and_grid_setup, :shock_generation, :warmup,
            :explicit_gc, :policy_alignment, :output_validation,
            :summary_statistics, :progress_logging, :file_io))
end

function _ar1_preference_timing_order(configuration)
    methods = configuration.methods
    return [(; panel=:ar1, agent_index=index, gamma,
        method=methods[mod1(index + offset, length(methods))])
        for (index, gamma) in enumerate(configuration.gammas)
        for offset in 0:length(methods)-1]
end

function _validate_ar1_preference_timing_samples(samples, configuration;
    complete::Bool=true)
    expected = _ar1_preference_timing_order(configuration)
    _ar1_preference_timing_require(complete ? length(samples) == length(expected) :
        length(samples) <= length(expected),
        "AR(1) preference timings do not cover the required five agents and three methods")
    for (sample, identity) in zip(samples, expected)
        _ar1_preference_timing_require(sample.panel === identity.panel &&
            sample.method === identity.method && sample.agent_index == identity.agent_index &&
            sample.gamma == identity.gamma && sample.beta == configuration.beta &&
            sample.initial_cash == configuration.initial_cash &&
            sample.shock_fingerprint == configuration.shock_fingerprint,
            "AR(1) preference timing sample identity or common path differs")
        _ar1_preference_timing_require(isfinite(sample.utility) &&
            all(value -> isfinite(value) && value >= 0,
                (sample.construction_seconds, sample.evaluation_seconds, sample.total_seconds)) &&
            sample.total_seconds == sample.construction_seconds + sample.evaluation_seconds,
            "AR(1) preference timing duration or utility is invalid")
    end
    return nothing
end

function _ar1_preference_timing_rows(samples, configuration)
    return [(; panel=:ar1, method, n_agents=configuration.n_agents,
        mean_total_seconds=mean(sample.total_seconds for sample in samples if sample.method === method),
        median_total_seconds=median([sample.total_seconds for sample in samples if sample.method === method]))
        for method in configuration.methods]
end

function _ar1_preference_timing_sources()
    root = PROJECT_ROOT
    paths = String[]
    for (directory, _, files) in walkdir(joinpath(root, "src"))
        append!(paths, [joinpath(directory, file) for file in files if endswith(file, ".jl")])
    end
    append!(paths, [joinpath(root, "scripts", "4_ar1.jl"),
        joinpath(root, "Project.toml"), joinpath(root, "Manifest.toml")])
    return [(path=relpath(path, root), sha256=bytes2hex(SHA.sha256(read(path))))
        for path in sort(paths)]
end

function _ar1_preference_timing_session()
    return (; started_utc=string(now(UTC)), julia_version=string(VERSION),
        julia_threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(),
        blas_vendor=string(BLAS.vendor()), cpu_name=Sys.CPU_NAME,
        machine=Sys.MACHINE, kernel=string(Sys.KERNEL))
end

function _validate_ar1_preference_timing_sessions(sessions, measurements)
    require = _ar1_preference_timing_require
    require(!isempty(sessions) && all(session ->
        session.julia_threads == session.blas_threads == 1 &&
        all(name -> !isempty(getproperty(session, name)),
            (:started_utc, :julia_version, :blas_vendor, :cpu_name, :machine, :kernel)), sessions),
        "AR(1) preference timings require one Julia thread and one BLAS thread")
    fields = (:julia_version, :julia_threads, :blas_threads, :blas_vendor,
        :cpu_name, :machine, :kernel)
    require(all(session -> all(name -> getproperty(session, name) ==
        getproperty(first(sessions), name), fields), sessions),
        "AR(1) preference timing execution environment changed across sessions")
    require(all(sample -> sample.session_index isa Integer &&
        1 <= sample.session_index <= length(sessions), measurements),
        "AR(1) preference timing sample session is missing")
    return nothing
end

function _validate_ar1_preference_timing_artifact(artifact, accuracy;
    require_production::Bool=true, require_current_source::Bool=false)
    require = _ar1_preference_timing_require
    require(artifact.schema == AR1_PREFERENCE_TIMING_SCHEMA,
        "unsupported AR(1) preference-timing schema")
    payload = (; (name => getproperty(artifact, name) for name in keys(artifact)
        if name !== :authentication_fingerprint)...)
    require(stable_fingerprint(payload) == artifact.authentication_fingerprint,
        "AR(1) preference timing authentication fingerprint differs")
    require_production && require(!artifact.configuration.smoke,
        "smoke timings cannot populate the AR(1) paper table")
    require(artifact.accuracy.authentication_fingerprint == accuracy["metadata"].authentication_fingerprint &&
        artifact.accuracy.result_fingerprint == accuracy["metadata"].result_fingerprint,
        "AR(1) preference timings belong to a different accuracy artifact")
    require(artifact.configuration == _ar1_preference_timing_configuration(accuracy;
        smoke=artifact.configuration.smoke, require_production,
        warmup_gammas=artifact.configuration.warmup_gammas),
        "AR(1) timing calibration, preference settings, common shocks, or protocol differ")
    sources = artifact.source_files
    paths = getproperty.(sources, :path)
    require(!isempty(sources) && length(unique(paths)) == length(paths) &&
        all(path -> !isabspath(path) && !(".." in splitpath(path)), paths) &&
        all(file -> occursin(r"^[0-9a-f]{64}$", file.sha256), sources) &&
        all(path -> path in paths, ("src/Ptd.jl", "src/simulation/ar1/preference_timing.jl",
            "Project.toml", "Manifest.toml")) &&
        # Preserve authenticated inventories recorded before script consolidation.
        count(path -> path in paths, ("scripts/4_ar1.jl", "scripts/11_ar1_preference_timing.jl",
            "scripts/ar1_preference_timing.jl")) == 1 &&
        stable_fingerprint(sources) == artifact.source_fingerprint,
        "AR(1) preference timing source inventory differs")
    require_current_source && require(sources == _ar1_preference_timing_sources(),
        "AR(1) preference timing source inputs differ from the current source")
    _validate_ar1_preference_timing_sessions(artifact.sessions, artifact.measurements)
    _validate_ar1_preference_timing_samples(artifact.measurements, artifact.configuration)
    require(artifact.rows == _ar1_preference_timing_rows(artifact.measurements, artifact.configuration),
        "AR(1) preference timing means or medians differ from raw measurements")
    return nothing
end

"""
Load single-path timing means and medians across the five selected AR(1) agents.
Authenticate the recorded warmup count: one for current runs or two for the
first production artifact. Neither protocol reuses a warmup policy in a timed solve.
"""
function load_ar1_preference_timing_summary(path::AbstractString, accuracy_path::AbstractString;
    require_production::Bool=true, require_current_source::Bool=false)
    source_path = abspath(path)
    accuracy_sha = bytes2hex(SHA.sha256(read(accuracy_path)))
    timing_sha = bytes2hex(SHA.sha256(read(source_path)))
    accuracy = load_ar1_contest_welfare(accuracy_path)
    artifact = JLD2.load(source_path, "artifact")
    _validate_ar1_preference_timing_artifact(artifact, accuracy;
        require_production, require_current_source)
    _ar1_preference_timing_require(artifact.accuracy.file_sha256 == accuracy_sha &&
        bytes2hex(SHA.sha256(read(accuracy_path))) == accuracy_sha &&
        bytes2hex(SHA.sha256(read(source_path))) == timing_sha,
        "AR(1) preference timing or parent accuracy file differs")
    return (; artifact, configuration=artifact.configuration,
        rows=Dict((row.panel, row.method) => row for row in artifact.rows),
        artifact_sha256=timing_sha, artifact_fingerprint=artifact.authentication_fingerprint,
        source_fingerprint=artifact.source_fingerprint, source_files=artifact.source_files,
        provenance_fingerprint=stable_fingerprint((; artifact.accuracy, artifact.sessions)),
        source_path)
end
