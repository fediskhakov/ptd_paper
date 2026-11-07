# Authenticated timings for every agent in the saved IID accuracy experiment.
using Statistics: mean, median

export load_iid_random_timing_summary

const IID_RANDOM_TIMING_SCHEMA = "ptd-iid-random-timing-v1"
const IID_RANDOM_TIMING_CHECKPOINT_SCHEMA = "ptd-iid-random-timing-checkpoint-v1"

_iid_timing_require(condition, message) = condition || throw(ArgumentError(message))

function _iid_timing_run_config(accuracy; require_production::Bool=true)
    scientific = accuracy["scientific_configuration"]
    model, outer = scientific.model, scientific.outer_design
    hidden, grid = scientific.hidden_evaluation, scientific.dcegm
    panels = Dict(panel.id => panel.sigma for panel in model.panels)
    config = IIDContestRunConfig(n_contests=outer.n_contests, n_paths=hidden.n_paths,
        horizon=model.horizon, return_rate=model.return_rate, cash_floor=model.cash_floor,
        bounds=ParameterBounds(outer.beta_bounds, outer.gamma_bounds, outer.initial_cash_bounds),
        panels=IIDVariancePanels(; baseline=panels[:baseline], high_variance=panels[:high_variance]),
        parameter_seed=outer.parameter_seed, replacement_parameter_seed=outer.replacement_parameter_seed,
        evaluation_seed=hidden.evaluation_seed, methods=Tuple(accuracy["methods"]),
        quadrature_nodes_per_dimension=scientific.quadrature.nodes_per_dimension,
        dcegm_asset_minimum=grid.asset_minimum, dcegm_asset_maximum=grid.asset_maximum,
        dcegm_asset_points=grid.asset_points, ptd_osa_options=PTDOSAOptions(; scientific.ptd_osa_options...))
    _iid_timing_require(scientific_configuration(config) == scientific,
        "IID timing cannot reconstruct the complete accuracy calibration")
    require_production && _iid_timing_require(_is_production_config(config),
        "paper IID timing requires the production accuracy design")
    completion = accuracy["run_audit"].completion
    _iid_timing_require(completion.completed_contests == config.n_contests &&
        completion.replacement_count == 0 && length(accuracy["outer_design"]) == config.n_contests &&
        size(accuracy["pathwise_utilities"]) ==
            (config.n_paths, length(config.methods), config.n_contests, length(keys(config.panels))),
        "IID timings require the entire original accuracy sample without replacements")
    return config
end

function _iid_timing_shock_records(accuracy, config)
    design = contest_design_config(config)
    records = NamedTuple[]
    for panel in keys(config.panels), (index, plain) in enumerate(accuracy["outer_design"])
        spec = ContestSpec(plain.id, plain.beta, plain.gamma, plain.initial_cash)
        block = predraw_innovations(spec, design, panel)
        audit = only(filter(row -> row.contest_id == spec.id, accuracy["run_audit"].contests))
        panel_audit = only(filter(row -> row.panel === panel, audit.panels))
        _iid_timing_require(shock_fingerprint(block) == panel_audit.block_fingerprint,
            "IID timing shocks differ from the saved accuracy experiment")
        shocks = copy(block[1:1, :])
        push!(records, (; panel, accuracy_index=index, spec_id=spec.id, path_index=1,
            accuracy_block_fingerprint=panel_audit.block_fingerprint,
            shocks, shock_fingerprint=shock_fingerprint(shocks)))
    end
    return records
end

function _iid_timing_configuration(accuracy, config)
    return (; scientific_configuration=accuracy["scientific_configuration"],
        specs=accuracy["outer_design"], methods=config.methods,
        panels=Tuple(keys(config.panels)), n_agents=config.n_contests,
        shock_records=_iid_timing_shock_records(accuracy, config),
        julia_threads=1, blas_threads=1, paths_per_agent=1, measured_runs_per_agent=1,
        selection=:all_original_accuracy_agents, path_selection=:first_saved_accuracy_path,
        method_order=:rotating_by_agent, warmup=:discarded_full_construction_and_path_per_panel_method,
        total=:sum_of_separately_measured_construction_and_path, utility_rtol=256eps(Float64),
        inside_timers=(:fresh_policy_construction, :full_backward_solve, :one_path_evaluation, :automatic_gc),
        outside_timers=(:model_and_grid_setup, :shock_generation, :warmup, :postwarmup_gc,
            :policy_alignment, :accuracy_comparison, :summary_statistics, :progress_logging, :file_io))
end

function _iid_timing_order(configuration)
    methods = configuration.methods
    return [(; panel, accuracy_index=index,
        method=methods[mod1(index + offset, length(methods))])
        for panel in configuration.panels for index in 1:configuration.n_agents
        for offset in 0:length(methods)-1]
end

function _validate_iid_timing_samples(samples, accuracy, configuration; complete::Bool=true)
    expected = _iid_timing_order(configuration)
    _iid_timing_require(complete ? length(samples) == length(expected) : length(samples) <= length(expected),
        "IID timing measurements do not cover the required sample")
    for (sample, identity) in zip(samples, expected)
        _iid_timing_require(sample.panel === identity.panel && sample.method === identity.method &&
            sample.accuracy_index == identity.accuracy_index &&
            sample.spec_id == configuration.specs[identity.accuracy_index].id && sample.path_index == 1,
            "IID timing sample identities or method order differ")
        panel_index = findfirst(==(sample.panel), configuration.panels)
        method_index = findfirst(==(sample.method), configuration.methods)
        reference = accuracy["pathwise_utilities"][1, method_index, sample.accuracy_index, panel_index]
        _iid_timing_require(isfinite(sample.utility) &&
            abs(sample.utility - reference) <= configuration.utility_rtol * max(1.0, abs(reference)),
            "IID timed utility differs from the saved first accuracy path")
        _iid_timing_require(all(value -> isfinite(value) && value >= 0,
            (sample.construction_seconds, sample.evaluation_seconds, sample.total_seconds)) &&
            sample.total_seconds == sample.construction_seconds + sample.evaluation_seconds,
            "IID timing durations are invalid")
    end
    return nothing
end

function _iid_timing_rows(samples, configuration)
    return [(; panel, method, n_agents=configuration.n_agents,
        mean_seconds=mean(sample.total_seconds for sample in samples
            if sample.panel === panel && sample.method === method),
        median_seconds=median([sample.total_seconds for sample in samples
            if sample.panel === panel && sample.method === method]))
        for panel in configuration.panels for method in configuration.methods]
end

function _validate_iid_random_timing_artifact(artifact, accuracy; require_production::Bool=true)
    _iid_timing_require(artifact.schema == IID_RANDOM_TIMING_SCHEMA, "unsupported IID random-timing schema")
    payload = (; (name => getproperty(artifact, name) for name in keys(artifact)
        if name !== :authentication_fingerprint)...)
    _iid_timing_require(stable_fingerprint(payload) == artifact.authentication_fingerprint,
        "IID random-timing authentication fingerprint differs")
    config = _iid_timing_run_config(accuracy; require_production)
    _iid_timing_require(artifact.accuracy.authentication_fingerprint ==
        accuracy["metadata"].authentication_fingerprint &&
        artifact.accuracy.result_fingerprint == accuracy["metadata"].result_fingerprint,
        "IID timings belong to a different accuracy artifact")
    _iid_timing_require(artifact.configuration == _iid_timing_configuration(accuracy, config),
        "IID timing calibration, agents, shocks, or measurement protocol differ")
    _iid_timing_require(!isempty(artifact.source_files) &&
        stable_fingerprint(artifact.source_files) == artifact.source_fingerprint,
        "IID timing source fingerprints differ")
    _iid_timing_require(!isempty(artifact.sessions) && all(session ->
        session.julia_threads == session.blas_threads == 1 && !isempty(session.julia_version), artifact.sessions),
        "IID timing runtime requires one Julia thread and one BLAS thread")
    _iid_timing_require(all(sample -> 1 <= sample.session_index <= length(artifact.sessions), artifact.measurements),
        "IID timing sample session is missing")
    _validate_iid_timing_samples(artifact.measurements, accuracy, artifact.configuration)
    _iid_timing_require(artifact.rows == _iid_timing_rows(artifact.measurements, artifact.configuration),
        "IID timing means or medians differ from the raw measurements")
    return nothing
end

"""Load authenticated single-path timings for exactly the saved IID accuracy agents."""
function load_iid_random_timing_summary(path::AbstractString, accuracy_path::AbstractString;
    require_production::Bool=true)
    accuracy = load_iid_contest(accuracy_path)
    saved = JLD2.load(path)
    _iid_timing_require(haskey(saved, "artifact"), "IID timing artifact payload is missing")
    artifact = saved["artifact"]
    _validate_iid_random_timing_artifact(artifact, accuracy; require_production)
    _iid_timing_require(artifact.accuracy.file_sha256 == bytes2hex(SHA.sha256(read(accuracy_path))),
        "IID timing parent accuracy file differs")
    return (; artifact, configuration=artifact.configuration,
        rows=Dict((row.panel, row.method) => row for row in artifact.rows),
        artifact_sha256=bytes2hex(SHA.sha256(read(path))),
        artifact_fingerprint=artifact.authentication_fingerprint,
        source_fingerprint=artifact.source_fingerprint,
        provenance_fingerprint=stable_fingerprint((; artifact.accuracy, artifact.sessions)))
end
