# Current-solver evidence on the original heterogeneous Figure 9 observations.
using JLD2
using ..Ptd: _validate_data_output, _temporary_jld_path

export AR1UpdatedPathOptions, AR1UpdatedPathResult, AR1UpdatedPathRenderResult,
       recompute_ar1_contest_paths, save_ar1_updated_paths,
       load_ar1_updated_paths, render_ar1_updated_paths

const AR1_UPDATED_PATH_SCHEMA = "ptd-ar1-updated-heterogeneous-paths-v1"

"""Original cash targets and number of original shock rows shown per configuration."""
struct AR1UpdatedPathOptions{N}
    targets::NTuple{N,Float64}
    paths_per_configuration::Int
    function AR1UpdatedPathOptions(; targets=AR1_CONTEST_PATH_TARGETS,
        paths_per_configuration::Integer=AR1_CONTEST_PATHS_PER_CONFIGURATION)
        cash_targets = Tuple(Float64.(collect(targets)))
        !isempty(cash_targets) && all(isfinite, cash_targets) ||
            throw(ArgumentError("path targets must be nonempty and finite"))
        paths_per_configuration > 0 || throw(ArgumentError(
            "paths_per_configuration must be positive"))
        return new{length(cash_targets)}(cash_targets, Int(paths_per_configuration))
    end
end

"""Current-solver matrices for one method on the selected original shock rows."""
struct AR1UpdatedMethodTrace
    income::Matrix{Float64}
    cash::Matrix{Float64}
    consumption::Matrix{Float64}
end

"""In-memory recomputation of one unchanged archived contest configuration."""
struct AR1UpdatedConfigurationResult{S}
    selection::S
    block_fingerprint::String
    innovation_rows::Matrix{Float64}
    income::Matrix{Float64}
    series::Dict{Symbol,AR1UpdatedMethodTrace}
    replayed_utility::Dict{Symbol,Vector{Float64}}
    historical_utility::Dict{Symbol,Vector{Float64}}
    diagnostics::Vector{NamedTuple}
end

"""Authenticated new trace evidence, its reusable artifact path, and figure payload."""
struct AR1UpdatedPathResult{A,T}
    path::String
    artifact::A
    traces::T
end

"""Figure paths, selected calibrations, and authenticated comparison provenance."""
struct AR1UpdatedPathRenderResult{C,D}
    path::String
    paths_path::String
    selected_configurations::C
    comparisons::D
    source_authentication_fingerprint::String
    payload_fingerprint::String
end

_updated_field(record, field::Symbol) =
    _exhibit_render_field(record, field, "updated AR(1) path artifact")

"""Economic inputs held fixed when only solver and sampling-schema versions change."""
function _updated_economic_inputs(scientific)
    outer = _updated_field(scientific, :outer_design)
    return (model=_updated_field(scientific, :model),
        outer_design=(n_contests=_updated_field(outer, :n_contests),
            beta_bounds=_updated_field(outer, :beta_bounds),
            gamma_bounds=_updated_field(outer, :gamma_bounds),
            initial_cash_bounds=_updated_field(outer, :initial_cash_bounds),
            parameter_seed=_updated_field(outer, :parameter_seed),
            replacement_parameter_seed=_updated_field(outer, :replacement_parameter_seed)),
        hidden_evaluation=_updated_field(scientific, :hidden_evaluation),
        methods=_updated_field(scientific, :methods),
        quadrature=_updated_field(scientific, :quadrature),
        g2egm=_updated_field(scientific, :g2egm),
        ptd_osa_options=_updated_field(scientific, :ptd_osa_options))
end

function _updated_compatible_configuration(artifact, config::AR1ContestRunConfig)
    archived = _updated_field(artifact, :scientific_configuration)
    current = scientific_configuration(config)
    stable_fingerprint(_updated_economic_inputs(archived)) ==
        stable_fingerprint(_updated_economic_inputs(current)) || throw(ArgumentError(
            "updated AR(1) paths must preserve the archived economic inputs, " *
            "design bounds, seeds, quadrature, grids, and numerical options"))
    Symbol.(collect(_updated_field(artifact, :methods))) == collect(config.methods) ||
        throw(ArgumentError("archived method ordering differs from the current configuration"))
    Symbol.(collect(_updated_field(artifact, :panels))) == [AR1_PANEL] ||
        throw(ArgumentError("the archived artifact must contain the AR(1) panel"))
    return archived, current
end

"""Recompute one unchanged specification after authenticating its full shock block."""
function _updated_configuration(artifact, config::AR1ContestRunConfig,
    selection, path_count::Int, hooks::AR1RunnerHooks)
    spec = selection.spec
    innovations = _validate_predraw(
        hooks.predraw(spec, contest_design_config(config), AR1_PANEL), config)
    domain = _evaluator_domain_audit(config, spec, innovations)
    block_fingerprint = shock_fingerprint(innovations)
    block_fingerprint == _stored_ar1_block_fingerprint(artifact,
        selection.contest_index; expected_contest_id=spec.id) ||
        throw(ArgumentError("regenerated shocks differ from the archived block " *
            "for contest $(spec.id)"))
    inputs = _model_inputs(config, spec)
    source_utilities = _updated_field(artifact, :pathwise_utilities)
    series = Dict{Symbol,AR1UpdatedMethodTrace}()
    current = Dict{Symbol,Vector{Float64}}()
    historical = Dict{Symbol,Vector{Float64}}()
    audits = NamedTuple[]
    for (method_index, method) in pairs(config.methods)
        policy = hooks.build_policy(method, inputs.model_config, inputs.process,
            inputs.base_rule, config.g2egm_grids, _effective_g2egm_options(config),
            config.ptd_osa_options)
        hooks.validate_policy(policy, method, inputs.model_config, inputs.process)
        quadrature = _compact_quadrature_audit(method,
            hooks.quadrature_audit(policy), inputs.base_rule)
        hooks.reset_diagnostics!(policy)
        rows = Any[]
        # The shared evaluator checks feasibility, terminal consume-all, and
        # repeated-state action consistency. It reveals shocks after the action.
        evaluation = evaluate_policy_paths(policy, method, inputs.model_config,
            inputs.process, inputs.initial, view(innovations, 1:path_count, :);
            action_audit=PolicyActionAudit(), trajectory_rows=rows)
        _validate_evaluation(evaluation, method, path_count)
        matrices = _ar1_trajectory_matrices(rows, config.horizon, path_count)
        series[method] = AR1UpdatedMethodTrace(
            matrices.income, matrices.cash, matrices.consumption)
        current[method] = copy(evaluation.utility)
        historical[method] = Float64.(collect(view(source_utilities,
            1:path_count, method_index, selection.contest_index, 1)))
        push!(audits, (contest_id=spec.id, method, quadrature,
            policy_domain=domain, solver=hooks.diagnostics(policy),
            feasible_actions_checked=true, terminal_actions_checked=true,
            evaluation_fingerprint=_evaluation_fingerprint(evaluation)))
    end
    reference = series[first(config.methods)].income
    all(series[method].income == reference for method in config.methods) ||
        throw(ErrorException("updated methods do not share identical income paths"))
    return AR1UpdatedConfigurationResult(selection, block_fingerprint,
        Matrix{Float64}(view(innovations, 1:path_count, :)), reference,
        series, current, historical, audits)
end

"""Plain historical/current utility differences; these do not modify the old table."""
function _updated_comparisons(results, methods)
    return [(contest_id=result.selection.spec.id, method,
        historical_utility=copy(result.historical_utility[method]),
        current_utility=copy(result.replayed_utility[method]),
        utility_change=result.replayed_utility[method] - result.historical_utility[method],
        mean_change=sum(result.replayed_utility[method] - result.historical_utility[method]) /
            length(result.replayed_utility[method]),
        maximum_absolute_change=maximum(abs,
            result.replayed_utility[method] - result.historical_utility[method]))
        for result in results for method in methods]
end

function _updated_payload(artifact)
    return (traces=_updated_field(artifact, :traces),
        comparisons=_updated_field(artifact, :comparisons),
        diagnostics=_updated_field(artifact, :diagnostics),
        source_scientific_configuration=_updated_field(artifact, :source_scientific_configuration),
        scientific_configuration=_updated_field(artifact, :scientific_configuration),
        provenance=_updated_field(artifact, :provenance),
        options=_updated_field(artifact, :options),
        solver_update_reason=_updated_field(artifact, :solver_update_reason))
end

_updated_authentication(metadata) = stable_fingerprint((
    artifact_schema=_updated_field(metadata, :artifact_schema),
    payload_fingerprint=_updated_field(metadata, :payload_fingerprint),
    source_authentication_fingerprint=_updated_field(metadata, :source_authentication_fingerprint),
    scientific_fingerprint=_updated_field(metadata, :scientific_fingerprint),
    source_scientific_fingerprint=_updated_field(metadata, :source_scientific_fingerprint),
    created_utc=_updated_field(metadata, :created_utc)))

"""Authenticate every stored trace, comparison, option and provenance field."""
function _validate_updated_paths(artifact; require_current_source::Bool=true)
    metadata = _updated_field(artifact, :metadata)
    _updated_field(metadata, :artifact_schema) == AR1_UPDATED_PATH_SCHEMA ||
        throw(ArgumentError("unsupported updated AR(1) path schema"))
    stable_fingerprint(_updated_payload(artifact)) ==
        _updated_field(metadata, :payload_fingerprint) ||
        throw(ArgumentError("updated AR(1) path payload fingerprint mismatch"))
    _updated_authentication(metadata) ==
        _updated_field(metadata, :authentication_fingerprint) ||
        throw(ArgumentError("updated AR(1) path authentication mismatch"))
    traces = _updated_field(artifact, :traces)
    normalized = _validate_normalized_traces(traces)
    _validate_ar1_contest_trace_groups(traces, normalized)
    _updated_field(traces, :source_authentication_fingerprint) ==
        _updated_field(metadata, :source_authentication_fingerprint) ||
        throw(ArgumentError("updated paths disagree with the archived artifact identity"))
    stable_fingerprint(_updated_field(artifact, :scientific_configuration)) ==
        _updated_field(metadata, :scientific_fingerprint) ||
        throw(ArgumentError("updated scientific fingerprint mismatch"))
    stable_fingerprint(_updated_field(artifact, :source_scientific_configuration)) ==
        _updated_field(metadata, :source_scientific_fingerprint) ||
        throw(ArgumentError("archived scientific fingerprint mismatch"))
    for row in _updated_field(artifact, :comparisons)
        old, fresh, change = row.historical_utility, row.current_utility, row.utility_change
        all(isfinite, old) && all(isfinite, fresh) && all(isfinite, change) ||
            throw(ArgumentError("updated comparison utilities must be finite"))
        fresh - old == change || throw(ArgumentError("updated utility differences disagree"))
    end
    if require_current_source
        provenance = _updated_field(artifact, :provenance)
        stable_fingerprint(_computational_input_tree(PROJECT_ROOT)) ==
            _updated_field(provenance, :input_tree_fingerprint) ||
            throw(ArgumentError("updated paths were computed from different current sources; " *
                "recompute them or explicitly inspect with require_current_source=false"))
    end
    return artifact
end

"""
    _recorded_welfare_source(recorded)

Return the absolute path of the welfare artifact that saved traces name.

# Arguments
- `recorded::AbstractString`: the saved `traces.source_artifact`. Current
  artifacts record a path relative to `PROJECT_ROOT`, such as
  `"data/ar1-contest-welfare.jld2"`, or only the file name when the source
  lay outside the package. Older artifacts recorded an absolute path.

# Returns
- `String`: `recorded` resolved against `PROJECT_ROOT`; an absolute
  `recorded` is only normalised, because `joinpath` keeps it unchanged.
"""
function _recorded_welfare_source(recorded::AbstractString)

    path = abspath(joinpath(PROJECT_ROOT, recorded))

    return path
end

"""
    save_ar1_updated_paths(path, artifact; overwrite=false, compress=true)

Atomically save the new, plain JLD2 evidence after authenticating its complete
payload. The archived welfare artifact is never written by this function.
"""
function save_ar1_updated_paths(path::AbstractString, artifact;
    overwrite::Bool=false, compress::Bool=true)
    _validate_updated_paths(artifact)
    destination = _validate_data_output(path)
    ispath(destination) && !overwrite && throw(ArgumentError(
        "updated path artifact already exists: $destination"))
    recorded_source = _updated_field(
        _updated_field(artifact, :traces), :source_artifact)
    destination == _recorded_welfare_source(recorded_source) &&
        throw(ArgumentError(
            "updated paths cannot overwrite their archived welfare source"))
    mkpath(dirname(destination))
    temporary = _temporary_jld_path(destination)
    try
        JLD2.jldopen(temporary, "w"; compress) do file
            file["artifact"] = artifact
        end
        load_ar1_updated_paths(temporary)
        ispath(destination) && !overwrite && throw(ArgumentError(
            "another process published the updated path artifact"))
        mv(temporary, destination; force=overwrite)
    finally
        rm(temporary; force=true)
    end
    return destination
end

"""Load and validate stored current-solver traces without invoking any solver."""
function load_ar1_updated_paths(path::AbstractString; require_current_source::Bool=true)
    artifact = JLD2.jldopen(path, "r") do file
        haskey(file, "artifact") || throw(ArgumentError("updated path artifact payload is missing"))
        file["artifact"]
    end
    return _validate_updated_paths(artifact; require_current_source)
end

"""
    recompute_ar1_contest_paths(source_path; solver_update_reason, ...)

Re-solve the five original heterogeneous Figure 9 configurations with current
policies and evaluate their first 20 original shock rows. The full archived
shock-block hashes, preferences, cash, economics, grids and quadrature must
match exactly. A nonempty `solver_update_reason` explicitly declares the code
revision; fresh utilities are saved alongside historical utilities and their
differences, rather than required to reproduce them. This creates separate
current-solver evidence and leaves the archived welfare table unchanged.
"""
function recompute_ar1_contest_paths(source_path::AbstractString;
    output_path::AbstractString=datadir("ar1-updated-paths.jld2"),
    config::AR1ContestRunConfig=production_run_config(),
    options::AR1UpdatedPathOptions=AR1UpdatedPathOptions(),
    workers::Integer=min(Threads.nthreads(), length(options.targets)),
    solver_update_reason::AbstractString,
    hooks::AR1RunnerHooks=AR1RunnerHooks(),
    overwrite::Bool=false, compress::Bool=true)
    output_path = _validate_data_output(output_path)
    isempty(strip(solver_update_reason)) && throw(ArgumentError(
        "a nonempty solver_update_reason is required for current-solver evidence"))
    validate_run_config(config)
    options.paths_per_configuration <= config.n_paths || throw(ArgumentError(
        "displayed path count exceeds the archived evaluation block"))
    workers > 0 || throw(ArgumentError("workers must be positive"))
    _is_production_config(config) && !_is_default_hooks(hooks) && throw(ArgumentError(
        "production updated paths require the default validated runner hooks"))
    ispath(output_path) && !overwrite && throw(ArgumentError(
        "updated path artifact already exists: $(abspath(output_path))"))
    abspath(source_path) == abspath(output_path) && throw(ArgumentError(
        "updated paths cannot overwrite the archived welfare source"))
    source = load_ar1_contest_welfare(source_path)
    archived, current = _updated_compatible_configuration(source, config)
    specs = _ar1_contest_specs(source)
    length(specs) == config.n_contests || throw(DimensionMismatch(
        "archived contest design size differs from the configuration"))
    audits = _updated_field(_updated_field(source, :run_audit), :contests)
    successful = [specs[Int(_updated_field(row, :contest_id))] for row in audits]
    selections = _select_ar1_contest_path_configurations(successful, options.targets)
    context = _capture_run_context(config, PROJECT_ROOT)
    LinearAlgebra.BLAS.set_num_threads(1)
    results = Vector{AR1UpdatedConfigurationResult}(undef, length(selections))
    work = index -> begin
        results[index] = _updated_configuration(source, config, selections[index],
            options.paths_per_configuration, hooks)
        @info "Updated heterogeneous AR(1) configuration completed" contest_id=selections[index].spec.id
        nothing
    end
    count = min(Int(workers), length(selections))
    if count == 1
        serial_scheduler(work, eachindex(selections), 1)
    else
        threaded_scheduler(work, eachindex(selections), count)
    end
    _assert_context!(context, "updated heterogeneous paths completed")
    traces = _combine_ar1_contest_path_replays(source_path, source, config,
        selections, results, options.paths_per_configuration)
    traces = merge(traces, (is_exact_historical_replay=false,
        recomputed_with_updated_solver=true, updates_archived_welfare=false,
        current_utility=traces.replayed_utility))
    source_metadata = _updated_field(source, :metadata)
    payload = (traces,
        comparisons=_updated_comparisons(results, config.methods),
        diagnostics=reduce(vcat, (result.diagnostics for result in results)),
        source_scientific_configuration=archived,
        scientific_configuration=current,
        provenance=(input_tree=context.input_tree,
            input_tree_fingerprint=context.input_tree_fingerprint,
            source=context.source, execution_environment=context.execution_environment,
            simulation_metadata=context.simulation_metadata),
        options=(targets=options.targets,
            paths_per_configuration=options.paths_per_configuration,
            workers=count, original_path_ids=collect(1:options.paths_per_configuration)),
        solver_update_reason=String(solver_update_reason))
    metadata = (artifact_schema=AR1_UPDATED_PATH_SCHEMA,
        payload_fingerprint=stable_fingerprint(payload),
        source_authentication_fingerprint=String(_updated_field(source_metadata, :authentication_fingerprint)),
        scientific_fingerprint=scientific_fingerprint(config),
        source_scientific_fingerprint=String(_updated_field(source_metadata, :scientific_fingerprint)),
        created_utc=Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z")
    metadata = merge(metadata,
        (authentication_fingerprint=_updated_authentication(metadata),))
    artifact = merge((; metadata), payload)
    path = save_ar1_updated_paths(output_path, artifact; overwrite, compress)
    return AR1UpdatedPathResult(path, artifact, traces)
end

"""
Render heterogeneous Figure 9 from authenticated stored data and its recorded source.
Set `require_current_source=true` to additionally require a matching live source tree.
"""
function render_ar1_updated_paths(path::AbstractString;
    graph_directory::AbstractString=graphs_dir(),
    size::Tuple{Int,Int}=(1_650, 1_500), dpi::Int=300,
    require_current_source::Bool=false)
    artifact = load_ar1_updated_paths(path; require_current_source)
    destination = _validate_data_output(joinpath(abspath(graph_directory),
        AR1_CONTEST_EXHIBIT_PATHS_FILENAME))
    _atomic_savefig(_ar1_contest_trace_figure(artifact.traces; size, dpi), destination)
    return AR1UpdatedPathRenderResult(destination, destination,
        artifact.traces.configurations, artifact.comparisons,
        artifact.metadata.source_authentication_fingerprint,
        artifact.metadata.payload_fingerprint)
end
