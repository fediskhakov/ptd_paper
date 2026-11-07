using Dates
using LinearAlgebra
using SHA
using DrWatson: datadir

import ..Ptd: IID_CONTEST_METHODS,
              _runtime_lock_path,
              IIDLognormalProcess,
              IID_DCEGM_ASSET_MAXIMUM,
              IID_DCEGM_GRID_SCHEMA,
              IID_DCEGM_REPORTED_POINTS,
              IIDState,
              PTDOSAOptions,
              ObservedState,
              PolicyActionAudit,
              PolicyPathEvaluation,
              StochasticPTDConfig,
              build_iid_contest_policy,
              evaluate_policy_paths,
              gauss_hermite_rule,
              iid_contest_asset_grid,
              iid_dcegm_asset_grid,
              iid_contest_diagnostics,
              iid_contest_quadrature_audit,
              reset_iid_contest_diagnostics!,
              validate_iid_contest_policy_alignment
using ..Ptd: ContestDesignConfig,
                       ContestFailure,
                       ContestSpec,
                       ParameterBounds,
                       build_contest_design,
                       complete_contest_design,
                       predraw_innovations
using ..Ptd: git_metadata,
                         PROJECT_ROOT,
                         simulation_metadata,
                         stable_fingerprint

export IIDContestRunConfig,
       IIDVariancePanels,
       RunnerHooks,
       RUNNER_SCHEMA_VERSION,
       TIMING_SCHEMA_VERSION,
       production_run_config,
       validate_run_config,
       validate_production_environment,
       contest_design_config,
       scientific_configuration,
       scientific_fingerprint,
       shock_fingerprint,
       serial_scheduler,
       threaded_scheduler,
       run_welfare_contest,
       run_timing_pass,
       run_production_contest

const RUNNER_SCHEMA_VERSION = "ptd-iid-contest-runner-v3"
const TIMING_SCHEMA_VERSION = "ptd-iid-contest-timing-v2"
const SHOCK_SCHEMA_VERSION = "ptd-iid-standard-normal-block-v1"
const COMPLETE_DESIGN_SCHEMA_VERSION = "ptd-iid-fixed-design-v2"
const RUN_CONTEXT_SCHEMA_VERSION = "ptd-iid-run-context-v1"
const INPUT_TREE_SCHEMA_VERSION = "ptd-iid-computational-inputs-v1"
const EXECUTION_ENVIRONMENT_SCHEMA_VERSION = "ptd-iid-execution-environment-v1"
const REQUIRED_METHODS = (:ptd_sceq, :ptd_osa, :dcegm)
const DEFAULT_PARAMETER_SEED = UInt64(0x5054445f50415241)
const DEFAULT_REPLACEMENT_PARAMETER_SEED = UInt64(0x5054445f49494452)
const DEFAULT_EVALUATION_SEED = UInt64(0x5054445f4556414c)
const SOLVER_SCHEMAS = (
    ptd_sceq="ptd-sceq-conditional-mean-v1",
    ptd_osa="ptd-osa-certified-five-node-v1",
    dcegm="iid-dcegm-five-node-single-log-v2",
)

"""Innovation standard deviations for the two IID variance panels."""
Base.@kwdef struct IIDVariancePanels
    baseline::Float64 = 0.1
    high_variance::Float64 = 0.5
end

Base.keys(::IIDVariancePanels) = (:baseline, :high_variance)
Base.values(panels::IIDVariancePanels) = (panels.baseline, panels.high_variance)
Base.pairs(panels::IIDVariancePanels) =
    (:baseline => panels.baseline, :high_variance => panels.high_variance)
Base.convert(::Type{IIDVariancePanels}, panels::NamedTuple) =
    IIDVariancePanels(; panels...)

"""Complete scientific configuration for an IID contest run.

The defaults are the locked production design. Smaller values are accepted for
tests and pilots, but `validate_production_environment` rejects every departure
from the production dimensions, calibration, quadrature, or DCEGM grid.
"""
Base.@kwdef struct IIDContestRunConfig
    n_contests::Int = 100
    n_paths::Int = 100
    horizon::Int = 50
    return_rate::Float64 = 0.10
    cash_floor::Float64 = 3.0
    bounds::ParameterBounds = ParameterBounds(
        (0.90, 0.995), (0.5, 5.0), (3.0, 50.0))
    panels::IIDVariancePanels = IIDVariancePanels()
    parameter_seed::UInt64 = DEFAULT_PARAMETER_SEED
    replacement_parameter_seed::UInt64 = DEFAULT_REPLACEMENT_PARAMETER_SEED
    evaluation_seed::UInt64 = DEFAULT_EVALUATION_SEED
    methods::NTuple{3,Symbol} = REQUIRED_METHODS
    quadrature_nodes_per_dimension::Int = 5
    dcegm_asset_minimum::Float64 = 0.0
    dcegm_asset_maximum::Float64 = IID_DCEGM_ASSET_MAXIMUM
    dcegm_asset_points::Int = IID_DCEGM_REPORTED_POINTS
    ptd_osa_options::PTDOSAOptions = PTDOSAOptions()
end

"""Return the locked configuration for the paper's IID contest."""
production_run_config() = IIDContestRunConfig()

"""Validate an IID contest configuration and return it unchanged."""
function validate_run_config(config::IIDContestRunConfig)
    config.n_contests > 0 || throw(ArgumentError("n_contests must be positive"))
    config.n_paths > 0 || throw(ArgumentError("n_paths must be positive"))
    config.horizon >= 2 || throw(ArgumentError("horizon must be at least two"))
    isfinite(config.return_rate) && config.return_rate > -1.0 ||
        throw(ArgumentError("return_rate must be finite and exceed -1"))
    isfinite(config.cash_floor) && config.cash_floor >= 0.0 ||
        throw(ArgumentError("cash_floor must be finite and nonnegative"))
    config.bounds.initial_cash[1] >= config.cash_floor || throw(ArgumentError(
        "initial-cash support cannot begin below the cash floor"))
    config.parameter_seed != config.evaluation_seed || throw(ArgumentError(
        "parameter and evaluator shock streams require distinct root seeds"))
    config.replacement_parameter_seed != config.parameter_seed ||
        throw(ArgumentError(
            "initial and replacement parameter streams require distinct root seeds"))
    config.replacement_parameter_seed != config.evaluation_seed ||
        throw(ArgumentError(
            "replacement parameters and evaluator shocks require distinct root seeds"))
    config.methods == REQUIRED_METHODS || throw(ArgumentError(
        "the IID contest requires exactly $(REQUIRED_METHODS) in that order"))
    config.methods == IID_CONTEST_METHODS || throw(ErrorException(
        "runner and evaluator method matrices disagree"))
    config.quadrature_nodes_per_dimension == 5 || throw(ArgumentError(
        "PTD-OSA and DCEGM require exactly five quadrature nodes"))
    all(sigma -> isfinite(sigma) && sigma > 0.0, values(config.panels)) ||
        throw(ArgumentError("IID panel standard deviations must be positive"))
    config.dcegm_asset_minimum == 0.0 || throw(ArgumentError(
        "the DCEGM asset grid must begin at zero"))
    isfinite(config.dcegm_asset_maximum) &&
        config.dcegm_asset_maximum > config.dcegm_asset_minimum ||
        throw(ArgumentError("the DCEGM asset maximum must be finite and positive"))
    config.dcegm_asset_points >= 2 || throw(ArgumentError(
        "the DCEGM asset grid needs at least two points"))
    return config
end

function _is_production_config(config::IIDContestRunConfig)
    return scientific_fingerprint(config) ==
           scientific_fingerprint(production_run_config())
end

"""Validate the production design and clean source tree, using one BLAS thread.

The Julia thread count controls parallel execution without changing the design.
"""
function validate_production_environment(config::IIDContestRunConfig;
    project_root::AbstractString=PROJECT_ROOT, context=nothing)
    validate_run_config(config)
    _is_production_config(config) || throw(ArgumentError(
        "the production entry point only accepts the locked 100-by-100 design"))
    LinearAlgebra.BLAS.set_num_threads(1)
    LinearAlgebra.BLAS.get_num_threads() == 1 || throw(ErrorException(
        "production requires one BLAS thread"))
    frozen = isnothing(context) ?
        _capture_run_context(config, project_root) : context
    abspath(project_root) == frozen.project_root || throw(ArgumentError(
        "the frozen run context belongs to a different project root"))
    frozen.scientific_fingerprint == scientific_fingerprint(config) ||
        throw(ArgumentError(
            "the frozen run context belongs to a different scientific configuration"))
    _assert_run_context_unchanged!(frozen, "production validation")
    source = frozen.source
    !source.dirty || throw(ArgumentError(
        "production refuses a dirty scoped Ptd tree; commit the runner and inputs first"))
    return (config=config, source=source, context=frozen)
end

"""Convert an IID run configuration to its shared contest-design contract."""
function contest_design_config(config::IIDContestRunConfig)
    validate_run_config(config)
    return ContestDesignConfig(
        n_contests=config.n_contests,
        n_paths=config.n_paths,
        horizon=config.horizon,
        floor=config.cash_floor,
        bounds=config.bounds,
        parameter_seed=config.parameter_seed,
        evaluation_seed=config.evaluation_seed,
    )
end

function _base_rule_configuration()
    rule = gauss_hermite_rule(5)
    return (
        family=:positive_gauss_hermite,
        nodes_per_dimension=rule.nodes_per_dimension,
        standard_normal_nodes=Tuple(only(node) for node in rule.nodes),
        normalized_weights=rule.weights,
    )
end

"""Plain, canonical-token-safe configuration used as the artifact identity."""
function scientific_configuration(config::IIDContestRunConfig)
    validate_run_config(config)
    options = config.ptd_osa_options
    return (
        runner_schema=RUNNER_SCHEMA_VERSION,
        model=(
            horizon=config.horizon,
            return_rate=config.return_rate,
            cash_floor=config.cash_floor,
            income_mean=1.0,
            panels=(
                (id=:baseline, sigma=config.panels.baseline),
                (id=:high_variance, sigma=config.panels.high_variance),
            ),
        ),
        outer_design=(
            n_contests=config.n_contests,
            beta_bounds=config.bounds.beta,
            gamma_bounds=config.bounds.gamma,
            initial_cash_bounds=config.bounds.initial_cash,
            parameter_seed=config.parameter_seed,
            sampler_schema=COMPLETE_DESIGN_SCHEMA_VERSION,
            replacement_parameter_seed=config.replacement_parameter_seed,
            completion_rule=:retain_original_configurations_and_failures,
        ),
        hidden_evaluation=(
            shock_schema=SHOCK_SCHEMA_VERSION,
            distribution=:standard_normal,
            layout=:paths_by_nonterminal_dates,
            n_paths=config.n_paths,
            evaluation_seed=config.evaluation_seed,
            stream_derivation=:splitmix64_contest_panel,
        ),
        methods=config.methods,
        solver_schemas=SOLVER_SCHEMAS,
        quadrature=_base_rule_configuration(),
        dcegm=(
            grid_schema=IID_DCEGM_GRID_SCHEMA,
            asset_minimum=config.dcegm_asset_minimum,
            asset_maximum=config.dcegm_asset_maximum,
            asset_points=config.dcegm_asset_points,
        ),
        ptd_osa_options=(
            tie_atol=options.tie_atol,
            tie_rtol=options.tie_rtol,
            certification_atol=options.certification_atol,
            direct_iterations=options.direct_iterations,
        ),
    )
end

"""Return the stable scientific fingerprint of an IID run configuration."""
scientific_fingerprint(config::IIDContestRunConfig) =
    stable_fingerprint(scientific_configuration(config))

function _computational_input_tree(project_root::AbstractString)
    root = abspath(project_root)
    candidates = String[]
    for relative in ("Project.toml", "Manifest.toml",
                     joinpath("scripts", "3_iid.jl"))
        path = joinpath(root, relative)
        isfile(path) && push!(candidates, path)
    end
    source_root = joinpath(root, "src")
    isdir(source_root) || throw(ArgumentError(
        "project source directory does not exist: $source_root"))
    for (directory, _, files) in walkdir(source_root)
        for file in files
            endswith(file, ".jl") || continue
            push!(candidates, joinpath(directory, file))
        end
    end
    sort!(candidates; by=path -> relpath(path, root))
    isempty(candidates) && throw(ArgumentError(
        "no computational inputs were found below $root"))
    files = [(;
        path=replace(relpath(path, root), '\\' => '/'),
        bytes=filesize(path),
        sha256=bytes2hex(SHA.sha256(read(path))),
    ) for path in candidates]
    return (schema=INPUT_TREE_SCHEMA_VERSION, files)
end

function _execution_environment_configuration(runtime, packages)
    return (
        schema=EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
        julia_version=String(_field(runtime, :julia_version)),
        word_size=Int(_field(runtime, :word_size)),
        kernel=String(_field(runtime, :kernel)),
        machine=String(_field(runtime, :machine)),
        cpu_name=String(_field(runtime, :cpu_name)),
        packages=copy(packages),
    )
end

function _capture_run_context(config::IIDContestRunConfig,
                              gitpath::AbstractString)
    configuration = scientific_configuration(config)
    root = abspath(gitpath)
    captured = simulation_metadata(configuration, root)
    input_tree = _computational_input_tree(root)
    input_tree_fingerprint = stable_fingerprint(input_tree)
    packages = copy(captured.packages)
    environment = _execution_environment_configuration(
        captured.runtime, packages)
    environment_fingerprint = stable_fingerprint(environment)
    metadata = (
        configuration=captured.configuration,
        source=merge(captured.source,
            (status_entries=copy(captured.source.status_entries),)),
        runtime=captured.runtime,
        packages,
        computational_inputs=input_tree,
        input_tree_fingerprint,
        execution_environment=environment,
        environment_fingerprint,
    )
    context = (
        schema=RUN_CONTEXT_SCHEMA_VERSION,
        project_root=root,
        scientific_fingerprint=stable_fingerprint(configuration),
        source=metadata.source,
        input_tree,
        input_tree_fingerprint,
        execution_environment=environment,
        environment_fingerprint,
        simulation_metadata=metadata,
    )
    _assert_run_context_unchanged!(context, "public-run capture")
    return context
end

function _assert_run_context_unchanged!(context, boundary::AbstractString)
    live_source = git_metadata(context.project_root)
    live_source.commit == context.source.commit || throw(ErrorException(
        "source commit changed after run start at $boundary"))
    live_inputs = _computational_input_tree(context.project_root)
    stable_fingerprint(live_inputs) == context.input_tree_fingerprint ||
        throw(ErrorException(
            "computational inputs changed after run start at $boundary"))
    # The loaded Julia/package environment cannot change without an explicit
    # in-process package operation. Its captured fingerprint keys the run;
    # Project.toml and Manifest.toml changes are already caught above.
    return context
end

function _artifact_run_fingerprint(scientific::String,
                                   source_commit::String,
                                   input_tree_fingerprint::String,
                                   environment_fingerprint::String)
    return stable_fingerprint((
        artifact_schema=RUNNER_SCHEMA_VERSION,
        scientific_fingerprint=scientific,
        source_commit,
        input_tree_fingerprint,
        environment_fingerprint,
    ))
end

_artifact_run_fingerprint(scientific::String, context) =
    _artifact_run_fingerprint(scientific, context.source.commit,
        context.input_tree_fingerprint, context.environment_fingerprint)

function _asset_grid(config::IIDContestRunConfig)
    config.dcegm_asset_minimum == 0.0 || throw(ArgumentError(
        "the single-log IID DCEGM grid must begin at zero"))
    return iid_dcegm_asset_grid(config.dcegm_asset_points;
        asset_maximum=config.dcegm_asset_maximum)
end

"""Return DCEGM's grid only for the method that actually consumes it."""
_method_asset_grid(method::Symbol, config::IIDContestRunConfig) =
    method === :dcegm ? _asset_grid(config) : nothing

function _default_build_policy(method, model_config, process, base_rule,
                               asset_grid, ptd_osa_options)
    return build_iid_contest_policy(method, model_config, process;
        base_rule, asset_grid, ptd_osa_options)
end

_default_evaluate(policy, method, model_config, process, initial, innovations,
                  action_audit) =
    evaluate_policy_paths(
        policy, method, model_config, process, initial, innovations;
        action_audit)
_default_diagnostics(policy) = iid_contest_diagnostics(policy)
_default_reset!(policy) = reset_iid_contest_diagnostics!(policy)
_default_audit(policy) = iid_contest_quadrature_audit(policy)
_default_validate_policy(policy, method, model_config, process) =
    validate_iid_contest_policy_alignment(
        policy, method, model_config, process)

"""Callbacks for policy construction, path evaluation, and diagnostics.

The builder receives a model configuration, shock law, quadrature rule,
asset grid, and PTD-OSA options.
"""
struct RunnerHooks{P,B,E,D,R,A,V}
    predraw::P
    build_policy::B
    evaluate::E
    diagnostics::D
    reset_diagnostics!::R
    quadrature_audit::A
    validate_policy::V
end

RunnerHooks(; predraw=predraw_innovations,
              build_policy=_default_build_policy,
              evaluate=_default_evaluate,
              diagnostics=_default_diagnostics,
              reset_diagnostics! = _default_reset!,
              quadrature_audit=_default_audit,
              validate_policy=_default_validate_policy) =
    RunnerHooks(predraw, build_policy, evaluate, diagnostics,
        reset_diagnostics!, quadrature_audit, validate_policy)

function _is_default_hooks(hooks::RunnerHooks)
    return hooks.predraw === predraw_innovations &&
           hooks.build_policy === _default_build_policy &&
           hooks.evaluate === _default_evaluate &&
           hooks.diagnostics === _default_diagnostics &&
           hooks.reset_diagnostics! === _default_reset! &&
           hooks.quadrature_audit === _default_audit &&
           hooks.validate_policy === _default_validate_policy
end

function _validate_runner_evaluation(evaluation,
                                     method::Symbol,
                                     expected_paths::Integer)
    evaluation isa PolicyPathEvaluation || throw(ArgumentError(
        "runner evaluation hook must return PolicyPathEvaluation"))
    expected = Int(expected_paths)
    expected > 0 || throw(ArgumentError(
        "expected path count must be positive"))
    evaluation.method == method || throw(ArgumentError(
        "evaluation method $(evaluation.method) does not match $method"))
    length(evaluation.utility) == expected || throw(DimensionMismatch(
        "utility length does not match the runner path count"))
    length(evaluation.floor_hits) == expected || throw(DimensionMismatch(
        "floor-hit length does not match the runner path count"))
    length(evaluation.total_transfers) == expected || throw(DimensionMismatch(
        "transfer length does not match the runner path count"))
    all(isfinite, evaluation.utility) || throw(ArgumentError(
        "contest runner utilities must all be finite"))
    all(hits -> hits >= 0, evaluation.floor_hits) || throw(ArgumentError(
        "floor-hit diagnostics must be nonnegative"))
    all(transfer -> isfinite(transfer) && transfer >= 0.0,
        evaluation.total_transfers) || throw(ArgumentError(
        "total-transfer diagnostics must be finite and nonnegative"))
    return evaluation
end

"""Apply `f` to `items` serially using the scheduler interface."""
function serial_scheduler(f::F, items, workers::Integer=1) where {F}
    workers > 0 || throw(ArgumentError("workers must be positive"))
    for item in items
        f(item)
    end
    return nothing
end

"""Bounded outer-task scheduler."""
function threaded_scheduler(f::F, items, workers::Integer) where {F}
    worker_count = Int(workers)
    worker_count > 0 || throw(ArgumentError("workers must be positive"))
    jobs = collect(items)
    isempty(jobs) && return nothing
    cursor = Threads.Atomic{Int}(1)
    @sync for _ in 1:min(worker_count, length(jobs))
        Threads.@spawn begin
            while true
                index = Threads.atomic_add!(cursor, 1)
                index > length(jobs) && break
                f(jobs[index])
            end
        end
    end
    return nothing
end

"""Stable SHA-256 identity of a real paths-by-dates shock block.

Float64 bit patterns are encoded in network byte order, avoiding the very large
canonical-token allocation that hashing a production matrix element by element
would otherwise require.
"""
function shock_fingerprint(innovations::AbstractMatrix{<:Real})
    all(isfinite, innovations) || throw(ArgumentError(
        "every evaluator innovation must be finite"))
    header = string(SHOCK_SCHEMA_VERSION, '|', size(innovations, 1), '|',
        size(innovations, 2), "|Float64|")
    words = Vector{UInt64}(undef, length(innovations))
    for (index, value) in enumerate(innovations)
        words[index] = hton(reinterpret(UInt64, Float64(value)))
    end
    context = SHA.SHA2_256_CTX()
    SHA.update!(context, collect(codeunits(header)))
    SHA.update!(context, reinterpret(UInt8, words))
    payload = bytes2hex(SHA.digest!(context))
    return stable_fingerprint((
        shock_schema=SHOCK_SCHEMA_VERSION,
        dimensions=Tuple(size(innovations)),
        payload_sha256=payload,
    ))
end

function _model_inputs(config::IIDContestRunConfig,
                       spec::ContestSpec,
                       sigma::Float64)
    model_config = StochasticPTDConfig(
        beta=spec.beta,
        gamma=spec.gamma,
        return_rate=config.return_rate,
        cash_floor=config.cash_floor,
        terminal_date=config.horizon,
    )
    process = IIDLognormalProcess(sigma=sigma)
    initial = ObservedState(1, spec.initial_cash, IIDState())
    base_rule = gauss_hermite_rule(
        process, config.quadrature_nodes_per_dimension)
    return (; model_config, process, initial, base_rule)
end

function _validate_predraw(innovations, config::IIDContestRunConfig)
    innovations isa AbstractMatrix{<:Real} || throw(ArgumentError(
        "the evaluator predraw must be a real paths-by-dates matrix"))
    size(innovations) == (config.n_paths, config.horizon - 1) ||
        throw(DimensionMismatch(
            "the evaluator predraw must have size " *
            "($(config.n_paths), $(config.horizon - 1))"))
    all(isfinite, innovations) || throw(ArgumentError(
        "the evaluator predraw contains a nonfinite innovation"))
    return innovations
end

function _field(container, name::Symbol)
    if container isa NamedTuple
        hasproperty(container, name) || throw(ArgumentError(
            "missing field $name"))
        return getproperty(container, name)
    elseif container isa AbstractDict
        haskey(container, name) && return container[name]
        haskey(container, String(name)) && return container[String(name)]
    end
    throw(ArgumentError("missing field $name"))
end

function _quadrature_fingerprint(rule)
    return stable_fingerprint((
        schema="canonical-standard-normal-gh-v1",
        nodes_per_dimension=rule.nodes_per_dimension,
        nodes=Tuple(only(node) for node in rule.nodes),
        weights=rule.weights,
    ))
end

function _runner_quadrature_bits(values)
    return Tuple(begin
        scalar = if value isa Tuple
            length(value) == 1 || throw(ArgumentError(
                "runner quadrature nodes must have one coordinate"))
            only(value)
        else
            value
        end
        scalar isa Real || throw(ArgumentError(
            "runner quadrature values must be real"))
        reinterpret(UInt64, Float64(scalar))
    end for value in values)
end

function _declared_quadrature_bits(audit, name::Symbol)
    values = _field(audit, name)
    return Tuple(begin
        value isa Integer || throw(ArgumentError(
            "quadrature bit audits must contain integers"))
        UInt64(value)
    end for value in values)
end

function _compact_quadrature_audit(method::Symbol, audit, expected_rule)
    Symbol(_field(audit, :method)) == method || throw(ErrorException(
        "quadrature audit method does not match $method"))
    uses = _field(audit, :uses_quadrature)
    uses isa Bool || throw(ArgumentError(
        "policy quadrature audit must expose a Bool uses_quadrature"))
    if method === :ptd_sceq
        uses == false || throw(ErrorException(
            "PTD-SCEQ must not report a numerical quadrature"))
        return (uses_quadrature=false, nodes_per_dimension=0,
            quadrature_fingerprint="none")
    end
    uses == true || throw(ErrorException(
        "$method must use the shared five-node quadrature"))
    nodes = Int(_field(audit, :nodes_per_dimension))
    nodes == 5 || throw(ErrorException(
        "$method did not use five quadrature nodes"))
    actual_node_bits = _runner_quadrature_bits(
        _field(audit, :base_nodes))
    actual_weight_bits = _runner_quadrature_bits(
        _field(audit, :base_weights))
    actual_node_bits == _declared_quadrature_bits(
        audit, :base_node_bits) || throw(ErrorException(
            "$method quadrature node audit is internally inconsistent"))
    actual_weight_bits == _declared_quadrature_bits(
        audit, :base_weight_bits) || throw(ErrorException(
            "$method quadrature weight audit is internally inconsistent"))
    actual_node_bits == _runner_quadrature_bits(expected_rule.nodes) ||
        throw(ErrorException(
            "$method did not use the shared quadrature nodes"))
    actual_weight_bits == _runner_quadrature_bits(expected_rule.weights) ||
        throw(ErrorException(
            "$method did not use the shared quadrature weights"))
    return (uses_quadrature=true, nodes_per_dimension=nodes,
        quadrature_fingerprint=_quadrature_fingerprint(expected_rule))
end

function _update_float_vector!(context, values)
    words = Vector{UInt64}(undef, length(values))
    for (index, value) in enumerate(values)
        words[index] = hton(reinterpret(UInt64, Float64(value)))
    end
    SHA.update!(context, reinterpret(UInt8, words))
    return context
end

function _update_integer_vector!(context, values)
    words = Vector{UInt64}(undef, length(values))
    for (index, value) in enumerate(values)
        words[index] = hton(reinterpret(UInt64, Int64(value)))
    end
    SHA.update!(context, reinterpret(UInt8, words))
    return context
end

"""Compact bit-level authentication digest for a path evaluation."""
function _evaluation_fingerprint(evaluation::PolicyPathEvaluation)
    context = SHA.SHA2_256_CTX()
    header = string("ptd-path-evaluation-v1|", evaluation.method, '|',
        length(evaluation.utility), '|')
    SHA.update!(context, collect(codeunits(header)))
    _update_float_vector!(context, evaluation.utility)
    _update_integer_vector!(context, evaluation.floor_hits)
    _update_float_vector!(context, evaluation.total_transfers)
    return bytes2hex(SHA.digest!(context))
end

function _run_panel_welfare(config::IIDContestRunConfig,
                            design_config::ContestDesignConfig,
                            spec::ContestSpec,
                            panel::Symbol,
                            sigma::Float64,
                            hooks::RunnerHooks)
    innovations = _validate_predraw(
        hooks.predraw(spec, design_config, panel), config)
    block_fingerprint = shock_fingerprint(innovations)
    inputs = _model_inputs(config, spec, sigma)
    method_results = Vector{Any}(undef, length(config.methods))
    for (method_index, method) in enumerate(config.methods)
        policy = hooks.build_policy(method, inputs.model_config,
            inputs.process, inputs.base_rule, _method_asset_grid(method, config),
            config.ptd_osa_options)
        hooks.validate_policy(policy, method, inputs.model_config, inputs.process)
        quadrature = _compact_quadrature_audit(method,
            hooks.quadrature_audit(policy), inputs.base_rule)
        hooks.reset_diagnostics!(policy)
        action_audit = PolicyActionAudit()
        evaluation = hooks.evaluate(policy, method, inputs.model_config,
            inputs.process, inputs.initial, innovations, action_audit)
        _validate_runner_evaluation(evaluation, method, config.n_paths)
        method_results[method_index] = (;
            method,
            evaluation,
            evaluation_fingerprint=_evaluation_fingerprint(evaluation),
            quadrature,
            solver_diagnostics=hooks.diagnostics(policy),
        )
    end
    return (; panel, block_fingerprint, methods=method_results)
end

function _run_contest_configuration(config::IIDContestRunConfig,
                                    design_config::ContestDesignConfig,
                                    spec::ContestSpec,
                                    hooks::RunnerHooks)
    panels = Dict{Symbol,Any}()
    for (panel, sigma) in pairs(config.panels)
        panels[panel] = _run_panel_welfare(
            config, design_config, spec, panel, sigma, hooks)
    end
    fingerprints = getfield.(collect(values(panels)), :block_fingerprint)
    length(unique(fingerprints)) == length(fingerprints) ||
        throw(ErrorException("IID panels unexpectedly share a shock block"))
    return (; contest_id=spec.id, panels)
end

function _run_welfare_unlocked(config::IIDContestRunConfig,
                               hooks::RunnerHooks;
                               workers::Int,
                               scheduler,
                               context)
    validate_run_config(config)
    scientific = scientific_fingerprint(config)
    scientific == context.scientific_fingerprint || throw(ArgumentError(
        "the frozen run context does not match the welfare configuration"))
    _assert_run_context_unchanged!(context, "welfare start")
    design_config = contest_design_config(config)
    run_fingerprint = _artifact_run_fingerprint(scientific, context)
    attempt = spec -> _run_contest_configuration(
        config, design_config, spec, hooks)
    selection = complete_contest_design(design_config;
        replacement_seed=config.replacement_parameter_seed,
        attempt, scheduler, workers,
        audit_schema=COMPLETE_DESIGN_SCHEMA_VERSION)
    _assert_run_context_unchanged!(context, "welfare completion")
    return (scientific_fingerprint=scientific,
        configuration_fingerprint=run_fingerprint,
        source_commit=context.source.commit,
        input_tree_fingerprint=context.input_tree_fingerprint,
        environment_fingerprint=context.environment_fingerprint,
        outer_design=selection.plain_design,
        sampler_audit=selection.audit,
        design=selection.design,
        results=selection.results)
end

function _validate_public_production_call(config::IIDContestRunConfig,
                                          workers::Int,
                                          scheduler,
                                          hooks::RunnerHooks;
                                          gitpath::AbstractString,
                                          context)
    _is_production_config(config) || return nothing
    workers == Threads.nthreads() || throw(ArgumentError(
        "production outer workers must match the Julia thread count"))
    scheduler === threaded_scheduler || throw(ArgumentError(
        "production requires the built-in threaded scheduler"))
    _is_default_hooks(hooks) || throw(ArgumentError(
        "injected runner hooks are restricted to nonproduction pilots and tests"))
    return validate_production_environment(
        config; project_root=gitpath, context)
end

"""Run every IID contest in memory under one process-level lock."""
function run_welfare_contest(config::IIDContestRunConfig=production_run_config();
    root::AbstractString=datadir(),
    workers::Integer=Threads.nthreads(),
    scheduler=threaded_scheduler,
    hooks::RunnerHooks=RunnerHooks(),
    gitpath::AbstractString=PROJECT_ROOT,
    lock_run::Bool=true)
    worker_count = Int(workers)
    worker_count > 0 || throw(ArgumentError("workers must be positive"))
    context = _capture_run_context(config, gitpath)
    _validate_public_production_call(
        config, worker_count, scheduler, hooks; gitpath, context)
    _is_production_config(config) && !lock_run && throw(ArgumentError(
        "production welfare execution requires the process-level run lock"))
    action = () -> _run_welfare_unlocked(config, hooks;
        workers=worker_count, scheduler, context)
    run_fingerprint = _artifact_run_fingerprint(
        scientific_fingerprint(config), context)
    return lock_run ? _with_run_lock(action, root, run_fingerprint) : action()
end

function _lock_path(root::AbstractString, run_fingerprint::String)
    isempty(run_fingerprint) && throw(ArgumentError(
        "the IID contest fingerprint cannot be empty"))
    return _runtime_lock_path(root, :iid_contest)
end

function _with_run_lock(action::F, root::AbstractString,
                        run_fingerprint::String) where {F}
    lock_path = _lock_path(root, run_fingerprint)
    mkpath(dirname(lock_path))
    try
        mkdir(lock_path)
    catch error
        isdir(lock_path) && throw(ArgumentError(
            "contest run is already locked at $lock_path; inspect its owner " *
            "before removing a stale lock"))
        rethrow(error)
    end
    try
        open(joinpath(lock_path, "owner.txt"), "w") do io
            println(io, "pid=", getpid())
            println(io, "hostname=", Base.gethostname())
            println(io, "destination=", abspath(root))
            println(io, "run_fingerprint=", run_fingerprint)
            println(io, "julia_threads=", Threads.nthreads())
            println(io, "started_utc=", Dates.format(now(UTC),
                dateformat"yyyy-mm-ddTHH:MM:SS.sss"), "Z")
        end
        return action()
    finally
        isdir(lock_path) && rm(lock_path; recursive=true)
    end
end

function _warm_method!(config::IIDContestRunConfig,
                       spec::ContestSpec,
                       sigma::Float64,
                       method::Symbol,
                       innovations,
                       hooks::RunnerHooks)
    inputs = _model_inputs(config, spec, sigma)
    policy = hooks.build_policy(method, inputs.model_config, inputs.process,
        inputs.base_rule, _method_asset_grid(method, config), config.ptd_osa_options)
    hooks.validate_policy(
        policy, method, inputs.model_config, inputs.process)
    audit = hooks.quadrature_audit(policy)
    _compact_quadrature_audit(
        method, audit, inputs.base_rule)
    action_audit = PolicyActionAudit()
    evaluation = hooks.evaluate(policy, method, inputs.model_config,
        inputs.process, inputs.initial, view(innovations, 1:1, :),
        action_audit)
    _validate_runner_evaluation(evaluation, method, 1)
    return nothing
end

function _run_timing_unlocked(config::IIDContestRunConfig,
                              hooks::RunnerHooks;
                              design,
                              plain_design=nothing,
                              sampler_audit=nothing,
                              workers::Int,
                              scheduler,
                              context)
    validate_run_config(config)
    scientific = scientific_fingerprint(config)
    scientific == context.scientific_fingerprint || throw(ArgumentError(
        "the frozen run context does not match the timing configuration"))
    _assert_run_context_unchanged!(context, "timing start")
    source_commit = context.source.commit
    run_fingerprint = _artifact_run_fingerprint(scientific, context)
    design_config = contest_design_config(config)
    length(design) == config.n_contests || throw(DimensionMismatch(
        "timing design does not contain every completed contest"))
    resolved_plain_design = isnothing(plain_design) ?
        [(id=spec.id, beta=spec.beta, gamma=spec.gamma,
          initial_cash=spec.initial_cash) for spec in design] :
        collect(plain_design)
    resolved_sampler_audit = isnothing(sampler_audit) ?
        (schema=:internal_unscreened_design,
         final_design_fingerprint=stable_fingerprint(resolved_plain_design)) :
        sampler_audit
    failed_slots = hasproperty(resolved_sampler_audit, :failed_slots) ?
        Set(resolved_sampler_audit.failed_slots) : Set{Int}()
    # Time the common successful subset while preserving the original design
    # fingerprint and reporting how many solver failures were excluded.
    design = filter(spec -> !(spec.id in failed_slots), design)
    indexed = collect(enumerate(design))
    totals = NamedTuple[]
    seen_block_fingerprints = Set{String}()

    for (panel, sigma) in pairs(config.panels)
        # The full common-shock panel is materialized before any warm-up or
        # timer. These exact matrices are read by all three method passes.
        blocks = [_validate_predraw(
            hooks.predraw(spec, design_config, panel), config)
                  for spec in design]
        block_fingerprints = shock_fingerprint.(blocks)
        length(unique(block_fingerprints)) == length(block_fingerprints) ||
            throw(ErrorException(
                "distinct outer contests unexpectedly share a shock block"))
        all(fingerprint -> !(fingerprint in seen_block_fingerprints),
            block_fingerprints) || throw(ErrorException(
                "distinct timing panels unexpectedly share a shock block"))
        union!(seen_block_fingerprints, block_fingerprints)

        for method in config.methods
            if isempty(design)
                push!(totals, (panel=panel, method=method, total_seconds=0.0,
                    status=:no_successful_contests))
                continue
            end
            _warm_method!(config, first(design), sigma, method,
                first(blocks), hooks)
            outputs = Vector{Any}(undef, length(design))
            started = time_ns()
            scheduler(indexed, workers) do indexed_spec
                index, spec = indexed_spec
                inputs = _model_inputs(config, spec, sigma)
                # Policy construction and complete hidden-path evaluation are
                # both inside this one panel-method wall-clock interval.
                policy = hooks.build_policy(method, inputs.model_config,
                    inputs.process, inputs.base_rule,
                    _method_asset_grid(method, config),
                    config.ptd_osa_options)
                hooks.validate_policy(
                    policy, method, inputs.model_config, inputs.process)
                action_audit = PolicyActionAudit()
                outputs[index] = hooks.evaluate(policy, method,
                    inputs.model_config, inputs.process, inputs.initial,
                    blocks[index], action_audit)
            end
            total_seconds = Float64(time_ns() - started) / 1.0e9

            # Validation of returned path evaluations is outside the interval.
            # Policy-alignment validation above is included in construction.
            for output in outputs
                _validate_runner_evaluation(
                    output, method, config.n_paths)
            end
            push!(totals, (panel=panel, method=method, total_seconds))
            _assert_run_context_unchanged!(
                context, "timing panel $panel method $method")
        end
    end

    return (
        artifact_schema=TIMING_SCHEMA_VERSION,
        scientific_fingerprint=scientific,
        run_fingerprint,
        source_commit,
        input_tree_fingerprint=context.input_tree_fingerprint,
        environment_fingerprint=context.environment_fingerprint,
        outer_design_fingerprint=stable_fingerprint(resolved_plain_design),
        sampler_audit_fingerprint=stable_fingerprint(resolved_sampler_audit),
        simulation_metadata=context.simulation_metadata,
        thread_count=workers,
        timed_contests=length(design),
        failed_contests=length(failed_slots),
        totals,
    )
end

"""Run an uninterrupted, no-JLD2 timing pass.

Each returned panel-method record contains one `total_seconds` value. Shock
generation, compilation warm-up, output validation, and all file I/O are
outside the timer. Policy-alignment validation and per-action feasibility checks
are included in the workload. If interrupted, this function returns nothing and no
sidecar is written.
"""
function run_timing_pass(config::IIDContestRunConfig=production_run_config();
    design=build_contest_design(contest_design_config(config)),
    workers::Integer=Threads.nthreads(),
    scheduler=threaded_scheduler,
    hooks::RunnerHooks=RunnerHooks(),
    gitpath::AbstractString=PROJECT_ROOT,
    lock_root::Union{Nothing,AbstractString}=nothing)
    worker_count = Int(workers)
    worker_count > 0 || throw(ArgumentError("workers must be positive"))
    context = _capture_run_context(config, gitpath)
    _validate_public_production_call(
        config, worker_count, scheduler, hooks; gitpath, context)
    _is_production_config(config) && isnothing(lock_root) &&
        throw(ArgumentError(
            "production timing execution requires the process-level run lock"))
    action = () -> _run_timing_unlocked(config, hooks;
        design, workers=worker_count, scheduler, context)
    if isnothing(lock_root)
        return action()
    end
    run_fingerprint = _artifact_run_fingerprint(
        scientific_fingerprint(config), context)
    return _with_run_lock(action, lock_root, run_fingerprint)
end

"""Execute the locked production workflow without launching it implicitly."""
function run_production_contest(config::IIDContestRunConfig=production_run_config();
    root::AbstractString=datadir(),
    gitpath::AbstractString=PROJECT_ROOT,
    compress::Bool=true)
    context = _capture_run_context(config, gitpath)
    validate_production_environment(config;
        project_root=gitpath, context)
    hooks = RunnerHooks()
    scientific = scientific_fingerprint(config)
    run_fingerprint = _artifact_run_fingerprint(
        scientific, context)
    action = function ()
        welfare = _run_welfare_unlocked(config, hooks;
            workers=Threads.nthreads(),
            scheduler=threaded_scheduler,
            context)
        timing = _run_timing_unlocked(config, hooks;
            design=welfare.design,
            plain_design=welfare.outer_design,
            sampler_audit=welfare.sampler_audit,
            workers=Threads.nthreads(),
            scheduler=threaded_scheduler,
            context)
        _assert_run_context_unchanged!(context, "single-artifact publication")
        publication = publish_iid_contest(config, welfare, timing;
            output_path=joinpath(abspath(root), "iid-contest.jld2"),
            context, compress)
        return (; welfare, timing, publication)
    end
    return _with_run_lock(action, root, run_fingerprint)
end
