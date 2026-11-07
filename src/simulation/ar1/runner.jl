using Dates
using LinearAlgebra
using Random
using SHA
using DrWatson: datadir

import ..Ptd: AR1G2ContestGridConfiguration,
              _runtime_lock_path,
              LogAR1Process,
              LogAR1State,
              PTDOSAOptions,
              ObservedState,
              PolicyActionAudit,
              PolicyPathEvaluation,
              StochasticPTDConfig,
              _plain_ar1_g2_grid_configuration,
              ar1_contest_diagnostics,
              ar1_contest_quadrature_audit,
              build_ar1_contest_policy,
              candidate_ar1_g2_grid_configuration,
              evaluate_policy_paths,
              gauss_hermite_rule,
              reset_ar1_contest_diagnostics!,
              validate_ar1_contest_policy_alignment
using ..Ptd: ContestDesignConfig,
                       ContestSpec,
                       ParameterBounds,
                       build_contest_design,
                       complete_contest_design, ContestFailure
using ..G2EGM: AR1G2EGMOptions,
               G2EGM_REFERENCE_COMMIT,
               G2EGM_SOLVER_SCHEMA
using ..Ptd: git_metadata,
                         PROJECT_ROOT,
                         simulation_metadata,
                         stable_fingerprint

export AR1ContestRunConfig,
       AR1RunnerHooks,
       RUNNER_SCHEMA_VERSION,
       TIMING_SCHEMA_VERSION,
       production_run_config,
       validate_run_config,
       contest_design_config,
       scientific_configuration,
       scientific_fingerprint,
       sample_ar1_contest_design,
       shock_fingerprint,
       predraw_ar1_innovations,
       serial_scheduler,
       threaded_scheduler,
       run_welfare_contest,
       run_timing_pass,
       run_production_contest,
       validate_production_environment

const RUNNER_SCHEMA_VERSION = "ptd-ar1-contest-runner-v4"
const TIMING_SCHEMA_VERSION = "ptd-ar1-contest-timing-v3"
const SHOCK_SCHEMA_VERSION = "ptd-ar1-standard-normal-block-v1"
const COMPLETE_DESIGN_SCHEMA_VERSION = "ptd-ar1-fixed-design-v3"
const RUN_CONTEXT_SCHEMA_VERSION = "ptd-ar1-run-context-v1"
const INPUT_TREE_SCHEMA_VERSION = "ptd-ar1-computational-inputs-v1"
const EXECUTION_ENVIRONMENT_SCHEMA_VERSION =
    "ptd-ar1-execution-environment-v1"
const REQUIRED_METHODS = (:ptd_sceq, :ptd_osa, :g2egm)
const AR1_PANEL = :ar1
const DEFAULT_PARAMETER_SEED = UInt64(0x5054445f50415241)
const DEFAULT_REPLACEMENT_PARAMETER_SEED = UInt64(0x5054445f5245504c)
const DEFAULT_EVALUATION_SEED = UInt64(0x5054445f41523145)
const UPSTREAM_G2EGM_COMMIT = G2EGM_REFERENCE_COMMIT
const SOLVER_SCHEMAS = (
    ptd_sceq="ptd-sceq-ar1-conditional-mean-v1",
    ptd_osa="ptd-osa-ar1-certified-q5-v1",
    g2egm=G2EGM_SOLVER_SCHEMA,
)
const DEFAULT_G2EGM_OPTIONS = AR1G2EGMOptions()

"""Plain solver settings at the scientific-artifact serialization boundary."""
function _plain_g2egm_options(options::AR1G2EGMOptions)
    return (bounding_box_padding=options.bounding_box_padding,
     minimum_barycentric_weight=options.minimum_barycentric_weight,
     hole_search_radius=options.hole_search_radius,
     value_atol=options.value_atol,
     value_rtol=options.value_rtol,
     determinant_tolerance=options.determinant_tolerance,
     reject_mixed_floor_regimes=options.reject_mixed_floor_regimes,
     enforce_monotone_saving=options.enforce_monotone_saving,
     monotonicity_atol=options.monotonicity_atol,
     monotonicity_rtol=options.monotonicity_rtol,
     direct_iterations=options.direct_iterations)
end

"""Scientific and numerical configuration for the one-panel AR(1) contest."""
Base.@kwdef struct AR1ContestRunConfig
    n_contests::Int = 100
    n_paths::Int = 100
    horizon::Int = 50
    return_rate::Float64 = 0.10
    cash_floor::Float64 = 3.0
    bounds::ParameterBounds = ParameterBounds(
        (0.90, 0.995), (0.5, 3.5), (3.0, 50.0))
    rho::Float64 = 0.9
    sigma::Float64 = 0.1
    unconditional_log_mean::Float64 = 0.0
    initial_income::Float64 = 1.0
    parameter_seed::UInt64 = DEFAULT_PARAMETER_SEED
    replacement_parameter_seed::UInt64 = DEFAULT_REPLACEMENT_PARAMETER_SEED
    evaluation_seed::UInt64 = DEFAULT_EVALUATION_SEED
    methods::NTuple{3,Symbol} = REQUIRED_METHODS
    quadrature_nodes_per_dimension::Int = 5
    g2egm_grids::AR1G2ContestGridConfiguration =
        candidate_ar1_g2_grid_configuration()
    g2egm_options::AR1G2EGMOptions = DEFAULT_G2EGM_OPTIONS
    ptd_osa_options::PTDOSAOptions = PTDOSAOptions()
end

"""Return the locked configuration for the paper's AR(1) contest."""
production_run_config() = AR1ContestRunConfig()

function _is_production_config(config::AR1ContestRunConfig)
    return scientific_fingerprint(config) ==
           scientific_fingerprint(production_run_config())
end

function _effective_g2egm_options(config::AR1ContestRunConfig)
    effective = config.g2egm_options
    effective.bounding_box_padding isa Integer &&
        effective.bounding_box_padding >= 0 || throw(ArgumentError(
            "G2EGM bounding_box_padding must be a nonnegative integer"))
    effective.hole_search_radius isa Integer &&
        effective.hole_search_radius >= 0 || throw(ArgumentError(
            "G2EGM hole_search_radius must be a nonnegative integer"))
    effective.minimum_barycentric_weight isa Real &&
        isfinite(effective.minimum_barycentric_weight) &&
        -1.0 <= effective.minimum_barycentric_weight <= 0.0 ||
        throw(ArgumentError(
            "G2EGM minimum_barycentric_weight must lie in [-1,0]"))
    all(value -> value isa Real && isfinite(value) && value >= 0.0,
        (effective.value_atol, effective.value_rtol,
         effective.determinant_tolerance, effective.monotonicity_atol,
         effective.monotonicity_rtol)) || throw(ArgumentError(
            "G2EGM tolerances must be finite and nonnegative"))
    effective.reject_mixed_floor_regimes isa Bool || throw(ArgumentError(
        "G2EGM reject_mixed_floor_regimes must be Bool"))
    effective.enforce_monotone_saving isa Bool || throw(ArgumentError(
        "G2EGM enforce_monotone_saving must be Bool"))
    effective.direct_iterations isa Integer &&
        effective.direct_iterations >= 16 || throw(ArgumentError(
            "G2EGM direct_iterations must be an integer of at least 16"))
    Float64(effective.minimum_barycentric_weight) ==
        config.g2egm_grids.barycentric_weight_floor || throw(ArgumentError(
            "the G2EGM option and grid barycentric-weight floors disagree"))
    return effective
end

function _policy_cash_upper(grids::AR1G2ContestGridConfiguration)
    asset_maximum = last(grids.asset_grid)
    policy_last = searchsortedlast(grids.common_cash_grid, asset_maximum)
    policy_last >= 1 || throw(ArgumentError(
        "the common G2EGM cash grid has no policy node at or below a_max"))
    return grids.common_cash_grid[policy_last]
end

"""Validate an AR(1) contest configuration and return it unchanged."""
function validate_run_config(config::AR1ContestRunConfig)
    config.n_contests > 0 || throw(ArgumentError("n_contests must be positive"))
    config.n_paths > 0 || throw(ArgumentError("n_paths must be positive"))
    config.horizon >= 2 || throw(ArgumentError("horizon must be at least two"))
    isfinite(config.return_rate) && config.return_rate > -1.0 ||
        throw(ArgumentError("return_rate must be finite and exceed -1"))
    isfinite(config.cash_floor) && config.cash_floor >= 0.0 ||
        throw(ArgumentError("cash_floor must be finite and nonnegative"))
    config.bounds.initial_cash[1] >= config.cash_floor || throw(ArgumentError(
        "initial-cash support cannot begin below the cash floor"))
    isfinite(config.rho) && abs(config.rho) < 1.0 || throw(ArgumentError(
        "rho must be finite and have absolute value below one"))
    isfinite(config.sigma) && config.sigma > 0.0 || throw(ArgumentError(
        "sigma must be finite and positive"))
    config.rho == 0.9 || throw(ArgumentError(
        "the locked AR(1) exercise requires rho=0.9"))
    config.sigma == 0.1 || throw(ArgumentError(
        "the locked AR(1) exercise requires sigma=0.1"))
    config.unconditional_log_mean == 0.0 || throw(ArgumentError(
        "the locked AR(1) exercise requires unconditional log mean zero"))
    config.initial_income == 1.0 || throw(ArgumentError(
        "the locked AR(1) exercise requires initial income one"))
    config.parameter_seed != config.evaluation_seed || throw(ArgumentError(
        "parameter and evaluator shock streams require distinct root seeds"))
    config.replacement_parameter_seed != config.parameter_seed ||
        throw(ArgumentError(
            "initial and replacement parameter streams require distinct root seeds"))
    config.replacement_parameter_seed != config.evaluation_seed ||
        throw(ArgumentError(
            "replacement parameters and evaluator shocks require distinct root seeds"))
    config.methods == REQUIRED_METHODS || throw(ArgumentError(
        "the AR(1) contest requires exactly $(REQUIRED_METHODS) in that order"))
    config.quadrature_nodes_per_dimension == 5 || throw(ArgumentError(
        "PTD-OSA and G2EGM require exactly five quadrature nodes"))
    first(config.g2egm_grids.common_cash_grid) == config.cash_floor ||
        throw(ArgumentError(
            "the common G2EGM cash grid must begin at the contest floor"))
    config.bounds.initial_cash[2] <= _policy_cash_upper(config.g2egm_grids) ||
        throw(ArgumentError(
            "the initial-cash support exceeds the nonterminal G2EGM policy domain"))
    _effective_g2egm_options(config)
    return config
end

"""Convert an AR(1) run configuration to its shared contest-design contract."""
function contest_design_config(config::AR1ContestRunConfig)
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

"""Return the complete scientific declaration for an AR(1) contest."""
function scientific_configuration(config::AR1ContestRunConfig)
    validate_run_config(config)
    options = config.ptd_osa_options
    return (
        runner_schema=RUNNER_SCHEMA_VERSION,
        model=(
            family=:scalar_log_ar1,
            transition=:conditional_gaussian_log_ar1,
            horizon=config.horizon,
            return_rate=config.return_rate,
            cash_floor=config.cash_floor,
            rho=config.rho,
            innovation_sigma=config.sigma,
            unconditional_log_mean=config.unconditional_log_mean,
            initial_income=config.initial_income,
            initial_log_income=log(config.initial_income),
            panels=((id=AR1_PANEL, rho=config.rho, sigma=config.sigma,
                initial_income=config.initial_income),),
        ),
        outer_design=(
            n_contests=config.n_contests,
            beta_bounds=config.bounds.beta,
            gamma_bounds=config.bounds.gamma,
            initial_cash_bounds=config.bounds.initial_cash,
            parameter_seed=config.parameter_seed,
            sampler_schema=COMPLETE_DESIGN_SCHEMA_VERSION,
            replacement_parameter_seed=config.replacement_parameter_seed,
            completion_rule=
                :retain_original_configurations_and_failures,
        ),
        hidden_evaluation=(
            shock_schema=SHOCK_SCHEMA_VERSION,
            distribution=:standard_normal,
            layout=:paths_by_nonterminal_dates,
            n_paths=config.n_paths,
            evaluation_seed=config.evaluation_seed,
            stream_derivation=:splitmix64_single_ar1_panel,
        ),
        methods=config.methods,
        solver_schemas=SOLVER_SCHEMAS,
        quadrature=_base_rule_configuration(),
        g2egm=(
            implementation=:pure_julia_problem_specific_adaptation,
            candidate_screen=:local_triangles_with_postdecision_interpolant,
            final_value_operator=:fresh_q5_on_child_common_grid,
            registered_method=:g2egm,
            display_name="G2EGM",
            upstream_repository="https://github.com/JeppeDruedahl/G2EGM",
            upstream_commit=UPSTREAM_G2EGM_COMMIT,
            upstream_license=:MIT,
            reference_file_sha256=(
                solve_m="78a7b8ac62ce998e2fe6a5b4da534bdba1fc94163f4b93e1a58ac23d7507ac34",
                fun_m="4fb0db10e36c29eb245050759b215d082bd3d4abbe41e61ceaf687e410090f77",
                egmfun_m="5990faa61b522f8149678f1fcd3d2bf58f1274a55eb27157aafa4a8ad8081d15",
                upper_envelope_cpp="38b22e6d2a01a8699ff1dc81050316a02624ddd3a06e53bac28b1f2ace1e902d",
            ),
            grids=_plain_ar1_g2_grid_configuration(config.g2egm_grids),
            options=_plain_g2egm_options(_effective_g2egm_options(config)),
        ),
        ptd_osa_options=(
            tie_atol=options.tie_atol,
            tie_rtol=options.tie_rtol,
            certification_atol=options.certification_atol,
            direct_iterations=options.direct_iterations,
        ),
    )
end

"""Return the stable fingerprint of an AR(1) scientific declaration."""
scientific_fingerprint(config::AR1ContestRunConfig) =
    stable_fingerprint(scientific_configuration(config))

function _splitmix64(value::UInt64)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

function _evaluation_seed(spec::ContestSpec, design::ContestDesignConfig)
    1 <= spec.id <= design.n_contests || throw(ArgumentError(
        "contest id is outside the configured design"))
    return _splitmix64(design.evaluation_seed + UInt64(spec.id - 1))
end

"""One evaluator-owned paths-by-dates normal block for the sole AR(1) panel."""
function predraw_ar1_innovations(spec::ContestSpec,
                                 design::ContestDesignConfig,
                                 panel::Symbol=AR1_PANEL)
    panel === AR1_PANEL || throw(ArgumentError("the AR(1) panel id is :ar1"))
    rng = Xoshiro(_evaluation_seed(spec, design))
    return randn(rng, design.n_paths, design.horizon - 1)
end

"""Fingerprint an evaluator-owned AR(1) innovation block."""
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
    return stable_fingerprint((
        shock_schema=SHOCK_SCHEMA_VERSION,
        dimensions=Tuple(size(innovations)),
        payload_sha256=bytes2hex(SHA.digest!(context)),
    ))
end

function _validate_predraw(innovations, config::AR1ContestRunConfig)
    innovations isa AbstractMatrix{<:Real} || throw(ArgumentError(
        "the evaluator predraw must be a real paths-by-dates matrix"))
    size(innovations) == (config.n_paths, config.horizon - 1) ||
        throw(DimensionMismatch("the AR(1) evaluator predraw has the wrong size"))
    all(isfinite, innovations) || throw(ArgumentError(
        "the evaluator predraw contains a nonfinite innovation"))
    return innovations
end

"""Prove that every hidden path remains inside the nonterminal G2EGM domain.

The recursion uses zero consumption, hence gives a policy-independent upper
bound on cash for PTD-SCEQ, PTD-OSA, and G2EGM alike. Nonterminal cash must not
exceed the largest common-cash node at or below `a_max`; this prevents public
policy interpolation from using the first unaudited padding node above the
asset endpoint. The analytic terminal query may use the full declared
common-grid padding but remains fail-closed outside that larger domain.
"""
function _evaluator_domain_audit(config::AR1ContestRunConfig,
                                 spec::ContestSpec,
                                 innovations)
    _validate_predraw(innovations, config)
    policy_cash_upper = _policy_cash_upper(config.g2egm_grids)
    log_lower = first(config.g2egm_grids.common_log_income_grid)
    log_upper = last(config.g2egm_grids.common_log_income_grid)
    upper_cash = fill(Float64(spec.initial_cash), config.n_paths)
    log_income = fill(log(config.initial_income), config.n_paths)
    minimum_cash_margin = Inf
    minimum_log_income_margin = Inf
    minimum_terminal_cash_margin = Inf
    maximum_nonterminal_cash = -Inf
    minimum_nonterminal_log_income = Inf
    maximum_nonterminal_log_income = -Inf
    gross = 1.0 + config.return_rate

    for step in 1:config.horizon-1
        all(isfinite, upper_cash) && all(isfinite, log_income) ||
            throw(ErrorException(
                "the AR(1) zero-consumption domain bound became nonfinite"))
        cash_maximum = maximum(upper_cash)
        ell_minimum, ell_maximum = extrema(log_income)
        cash_maximum <= policy_cash_upper || throw(DomainError(cash_maximum,
            "a hidden path can exceed the nonterminal G2EGM cash domain"))
        ell_minimum >= log_lower && ell_maximum <= log_upper ||
            throw(DomainError((ell_minimum, ell_maximum),
                "a hidden path leaves the G2EGM log-income domain"))
        maximum_nonterminal_cash = max(maximum_nonterminal_cash, cash_maximum)
        minimum_nonterminal_log_income = min(
            minimum_nonterminal_log_income, ell_minimum)
        maximum_nonterminal_log_income = max(
            maximum_nonterminal_log_income, ell_maximum)
        minimum_cash_margin = min(
            minimum_cash_margin, policy_cash_upper - cash_maximum)
        minimum_log_income_margin = min(minimum_log_income_margin,
            ell_minimum - log_lower, log_upper - ell_maximum)

        next_log_income = config.unconditional_log_mean .+
            config.rho .* (log_income .- config.unconditional_log_mean) .+
            config.sigma .* view(innovations, :, step)
        upper_cash = max.(gross .* upper_cash .+ exp.(next_log_income),
            config.cash_floor)
        log_income = next_log_income
    end
    terminal_cash_upper = last(config.g2egm_grids.common_cash_grid)
    all(isfinite, upper_cash) && all(isfinite, log_income) ||
        throw(ErrorException(
            "the AR(1) terminal zero-consumption bound became nonfinite"))
    terminal_cash_maximum = maximum(upper_cash)
    terminal_log_minimum, terminal_log_maximum = extrema(log_income)
    terminal_cash_maximum <= terminal_cash_upper ||
        throw(DomainError(terminal_cash_maximum,
            "a hidden path can exceed the terminal G2EGM cash domain"))
    terminal_log_minimum >= log_lower && terminal_log_maximum <= log_upper ||
        throw(DomainError((terminal_log_minimum, terminal_log_maximum),
            "a hidden terminal state leaves the G2EGM log-income domain"))
    minimum_terminal_cash_margin =
        terminal_cash_upper - terminal_cash_maximum
    minimum_log_income_margin = min(minimum_log_income_margin,
        terminal_log_minimum - log_lower,
        log_upper - terminal_log_maximum)
    return (gate_passed=true,
        policy_cash_upper,
        terminal_cash_upper,
        policy_log_income_lower=log_lower,
        policy_log_income_upper=log_upper,
        maximum_nonterminal_cash,
        maximum_terminal_cash=terminal_cash_maximum,
        minimum_nonterminal_log_income,
        maximum_nonterminal_log_income,
        minimum_cash_margin,
        minimum_terminal_cash_margin,
        minimum_log_income_margin)
end

function _model_inputs(config::AR1ContestRunConfig, spec::ContestSpec)
    model_config = StochasticPTDConfig(
        beta=spec.beta,
        gamma=spec.gamma,
        return_rate=config.return_rate,
        cash_floor=config.cash_floor,
        terminal_date=config.horizon,
    )
    process = LogAR1Process(rho=config.rho, sigma=config.sigma,
        unconditional_log_mean=config.unconditional_log_mean)
    initial = ObservedState(1, spec.initial_cash,
        LogAR1State(log(config.initial_income)))
    base_rule = gauss_hermite_rule(
        process, config.quadrature_nodes_per_dimension)
    return (; model_config, process, initial, base_rule)
end

function _default_build_policy(method, model_config, process, base_rule,
                               grids, g2egm_options, ptd_osa_options)
    return build_ar1_contest_policy(method, model_config, process;
        base_rule, ptd_osa_options, g2egm_grids=grids,
        g2egm_options)
end

_default_evaluate(policy, method, model_config, process, initial,
                  innovations, action_audit) =
    evaluate_policy_paths(policy, method, model_config, process, initial,
        innovations; action_audit)
_default_diagnostics(policy) = ar1_contest_diagnostics(policy)
_default_reset!(policy) = reset_ar1_contest_diagnostics!(policy)
_default_audit(policy) = ar1_contest_quadrature_audit(policy)
_default_validate(policy, method, model_config, process) =
    validate_ar1_contest_policy_alignment(
        policy, method, model_config, process)

"""Injectable seam; production ultimately requires every default hook."""
struct AR1RunnerHooks{P,B,E,D,R,A,V}
    predraw::P
    build_policy::B
    evaluate::E
    diagnostics::D
    reset_diagnostics!::R
    quadrature_audit::A
    validate_policy::V
end

AR1RunnerHooks(; predraw=predraw_ar1_innovations,
    build_policy=_default_build_policy,
    evaluate=_default_evaluate,
    diagnostics=_default_diagnostics,
    reset_diagnostics! = _default_reset!,
    quadrature_audit=_default_audit,
    validate_policy=_default_validate) =
    AR1RunnerHooks(predraw, build_policy, evaluate, diagnostics,
        reset_diagnostics!, quadrature_audit, validate_policy)

function _is_default_hooks(hooks::AR1RunnerHooks)
    return hooks.predraw === predraw_ar1_innovations &&
           hooks.build_policy === _default_build_policy &&
           hooks.evaluate === _default_evaluate &&
           hooks.diagnostics === _default_diagnostics &&
           hooks.reset_diagnostics! === _default_reset! &&
           hooks.quadrature_audit === _default_audit &&
           hooks.validate_policy === _default_validate
end

"""Apply `f` to `items` serially using the scheduler interface."""
function serial_scheduler(f::F, items, workers::Integer=1) where {F}
    workers > 0 || throw(ArgumentError("workers must be positive"))
    foreach(f, items)
    return nothing
end

"""Apply `f` with a bounded number of Julia worker tasks."""
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

"""
    sample_ar1_contest_design(config; certify_g2egm, workers, scheduler)

Build the fixed-seed Latin-hypercube sample and record certification failures
without replacing their parameter draws. `failure_policy=:replace` explicitly
requests the historical conditional sample with a finite replacement budget.
The full design, outcomes, and audit retain the original contest identifiers.
"""
function sample_ar1_contest_design(config::AR1ContestRunConfig;
    certify_g2egm,
    workers::Integer=Threads.nthreads(), scheduler=threaded_scheduler,
    failure_policy::Symbol=:retain, max_replacement_rounds::Integer=100)

    validate_run_config(config)
    worker_count = Int(workers)
    worker_count > 0 || throw(ArgumentError("workers must be positive"))
    attempt = spec -> begin
        passed = certify_g2egm(spec)
        passed === false && return false
        return true
    end
    completed = complete_contest_design(contest_design_config(config);
        replacement_seed=config.replacement_parameter_seed,
        attempt,
        scheduler,
        workers=worker_count,
        audit_schema=COMPLETE_DESIGN_SCHEMA_VERSION,
        failure_policy, max_replacement_rounds)
    return (;
        design=completed.design,
        provenance=completed.provenance,
        audit=completed.audit,
        results=completed.results,
        plain_design=completed.plain_design,
    )
end

function _certify_g2egm_configuration(config::AR1ContestRunConfig,
                                      hooks::AR1RunnerHooks,
                                      spec::ContestSpec)
    inputs = _model_inputs(config, spec)
    hooks.build_policy(:g2egm, inputs.model_config, inputs.process,
        inputs.base_rule, config.g2egm_grids,
        _effective_g2egm_options(config), config.ptd_osa_options)
    return true
end

function _sample_ar1_contest_design(config::AR1ContestRunConfig,
                                    hooks::AR1RunnerHooks;
                                    workers::Integer,
                                    scheduler)
    certify = spec -> _certify_g2egm_configuration(config, hooks, spec)
    return sample_ar1_contest_design(config;
        certify_g2egm=certify, workers, scheduler)
end

function _field(container, name::Symbol)
    if container isa NamedTuple
        hasproperty(container, name) && return getproperty(container, name)
    elseif container isa AbstractDict
        haskey(container, name) && return container[name]
        haskey(container, String(name)) && return container[String(name)]
    end
    throw(ArgumentError("missing field $name"))
end

function _quadrature_bits(values)
    return Tuple(reinterpret(UInt64, Float64(value isa Tuple ? only(value) : value))
                 for value in values)
end

function _compact_quadrature_audit(method::Symbol, audit, expected_rule)
    Symbol(_field(audit, :method)) == method || throw(ErrorException(
        "quadrature audit method does not match $method"))
    uses = _field(audit, :uses_quadrature)
    uses isa Bool || throw(ArgumentError(
        "policy quadrature audit must expose Bool uses_quadrature"))
    if method === :ptd_sceq
        uses == false || throw(ErrorException(
            "PTD-SCEQ must not report numerical quadrature"))
        return (uses_quadrature=false, nodes_per_dimension=0,
            quadrature_fingerprint="none")
    end
    uses == true || throw(ErrorException("$method must report quadrature"))
    Int(_field(audit, :nodes_per_dimension)) == 5 || throw(ErrorException(
        "$method did not use five quadrature nodes"))
    _quadrature_bits(_field(audit, :base_nodes)) ==
        _quadrature_bits(expected_rule.nodes) || throw(ErrorException(
            "$method did not use the shared Q5 nodes"))
    _quadrature_bits(_field(audit, :base_weights)) ==
        _quadrature_bits(expected_rule.weights) || throw(ErrorException(
            "$method did not use the shared Q5 weights"))
    return (uses_quadrature=true, nodes_per_dimension=5,
        quadrature_fingerprint=stable_fingerprint((
            nodes=_quadrature_bits(expected_rule.nodes),
            weights=_quadrature_bits(expected_rule.weights))))
end

function _validate_evaluation(evaluation, method::Symbol, paths::Int)
    evaluation isa PolicyPathEvaluation || throw(ArgumentError(
        "runner evaluation hook must return PolicyPathEvaluation"))
    evaluation.method == method || throw(ArgumentError(
        "evaluation method does not match $method"))
    length(evaluation.utility) == paths &&
        length(evaluation.floor_hits) == paths &&
        length(evaluation.total_transfers) == paths ||
        throw(DimensionMismatch("runner evaluation vectors have the wrong length"))
    all(isfinite, evaluation.utility) || throw(ArgumentError(
        "AR(1) contest utilities must be finite"))
    all(>=(0), evaluation.floor_hits) || throw(ArgumentError(
        "floor-hit diagnostics must be nonnegative"))
    all(value -> isfinite(value) && value >= 0.0,
        evaluation.total_transfers) || throw(ArgumentError(
            "transfer diagnostics must be finite and nonnegative"))
    return evaluation
end

function _computational_input_tree(project_root::AbstractString)
    root = abspath(project_root)
    candidates = String[]
    for relative in ("Project.toml", "Manifest.toml",
                     joinpath("scripts", "4_ar1.jl"))
        path = joinpath(root, relative)
        isfile(path) && push!(candidates, path)
    end
    source_root = joinpath(root, "src")
    isdir(source_root) || throw(ArgumentError(
        "project source directory does not exist: $source_root"))
    for (directory, _, files) in walkdir(source_root), file in files
        endswith(file, ".jl") && push!(candidates, joinpath(directory, file))
    end
    sort!(candidates; by=path -> relpath(path, root))
    files = [(path=replace(relpath(path, root), '\\' => '/'),
              bytes=filesize(path),
              sha256=bytes2hex(SHA.sha256(read(path)))) for path in candidates]
    return (schema=INPUT_TREE_SCHEMA_VERSION, files)
end

function _capture_run_context(config::AR1ContestRunConfig,
                              project_root::AbstractString)
    root = abspath(project_root)
    configuration = scientific_configuration(config)
    simulation = simulation_metadata(configuration, root)
    input_tree = _computational_input_tree(root)
    input_tree_fingerprint = stable_fingerprint(input_tree)
    environment = (
        schema=EXECUTION_ENVIRONMENT_SCHEMA_VERSION,
        julia_version=simulation.runtime.julia_version,
        word_size=simulation.runtime.word_size,
        kernel=simulation.runtime.kernel,
        machine=simulation.runtime.machine,
        cpu_name=simulation.runtime.cpu_name,
        packages=copy(simulation.packages),
    )
    environment_fingerprint = stable_fingerprint(environment)
    metadata = merge(simulation, (
        computational_inputs=input_tree,
        input_tree_fingerprint,
        execution_environment=environment,
        environment_fingerprint,
    ))
    return (
        schema=RUN_CONTEXT_SCHEMA_VERSION,
        project_root=root,
        scientific_fingerprint=stable_fingerprint(configuration),
        source=merge(simulation.source,
            (status_entries=copy(simulation.source.status_entries),)),
        input_tree,
        input_tree_fingerprint,
        execution_environment=environment,
        environment_fingerprint,
        simulation_metadata=metadata,
    )
end

function _assert_context!(context, boundary::AbstractString)
    live_source = git_metadata(context.project_root)
    live_source.commit == context.source.commit || throw(ErrorException(
        "source commit changed after AR(1) run start at $boundary"))
    stable_fingerprint(_computational_input_tree(context.project_root)) ==
        context.input_tree_fingerprint || throw(ErrorException(
            "computational inputs changed after AR(1) run start at $boundary"))
    return context
end

function _artifact_run_fingerprint(scientific::String, context)
    return stable_fingerprint((
        artifact_schema=RUNNER_SCHEMA_VERSION,
        scientific_fingerprint=scientific,
        source_commit=context.source.commit,
        input_tree_fingerprint=context.input_tree_fingerprint,
        environment_fingerprint=context.environment_fingerprint,
    ))
end

function _evaluation_fingerprint(evaluation::PolicyPathEvaluation)
    return stable_fingerprint((
        schema="ptd-ar1-path-evaluation-v1",
        method=evaluation.method,
        utility_bits=Tuple(reinterpret.(UInt64, evaluation.utility)),
        floor_hits=Tuple(evaluation.floor_hits),
        transfer_bits=Tuple(reinterpret.(UInt64, evaluation.total_transfers)),
    ))
end

function _plain_design(design, provenance=nothing)
    specifications = [(id=spec.id, beta=spec.beta, gamma=spec.gamma,
        initial_cash=spec.initial_cash) for spec in design]
    isnothing(provenance) && return specifications
    length(provenance) == length(specifications) || throw(DimensionMismatch(
        "AR(1) design provenance has the wrong length"))
    return [merge(specifications[index], provenance[index])
        for index in eachindex(specifications)]
end

function _lock_path(root::AbstractString, run_fingerprint::String)
    isempty(run_fingerprint) && throw(ArgumentError(
        "the AR(1) contest fingerprint cannot be empty"))
    return _runtime_lock_path(root, :ar1_contest)
end

function _with_run_lock(action::F, root::AbstractString,
                        run_fingerprint::String) where {F}
    lock_path = _lock_path(root, run_fingerprint)
    mkpath(dirname(lock_path))
    try
        mkdir(lock_path)
    catch error
        isdir(lock_path) && throw(ArgumentError(
            "AR(1) contest run is already locked at $lock_path; inspect " *
            "its owner before removing a stale lock"))
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

function _run_outer(config, design_config, spec, hooks)
    innovations = _validate_predraw(
        hooks.predraw(spec, design_config, AR1_PANEL), config)
    policy_domain = _evaluator_domain_audit(config, spec, innovations)
    block_fingerprint = shock_fingerprint(innovations)
    inputs = _model_inputs(config, spec)
    effective_g2egm_options = _effective_g2egm_options(config)
    method_results = Vector{Any}(undef, length(config.methods))

    for (method_index, method) in enumerate(config.methods)
        policy = hooks.build_policy(method, inputs.model_config,
            inputs.process, inputs.base_rule, config.g2egm_grids,
            effective_g2egm_options, config.ptd_osa_options)
        hooks.validate_policy(policy, method, inputs.model_config,
            inputs.process)
        quadrature = _compact_quadrature_audit(method,
            hooks.quadrature_audit(policy), inputs.base_rule)
        hooks.reset_diagnostics!(policy)
        action_audit = PolicyActionAudit()
        evaluation = hooks.evaluate(policy, method, inputs.model_config,
            inputs.process, inputs.initial, innovations, action_audit)
        _validate_evaluation(evaluation, method, config.n_paths)
        method_results[method_index] = (;
            method,
            evaluation,
            evaluation_fingerprint=_evaluation_fingerprint(evaluation),
            quadrature,
            solver_diagnostics=hooks.diagnostics(policy),
        )
    end
    panel_result = (;
        panel=AR1_PANEL,
        block_fingerprint,
        policy_domain,
        methods=method_results,
    )
    return (;
        contest_id=spec.id,
        panels=Dict(AR1_PANEL => panel_result),
    )
end

function _run_welfare_unlocked(config, hooks;
                               workers, scheduler, context)
    scientific = scientific_fingerprint(config)
    scientific == context.scientific_fingerprint || throw(ArgumentError(
        "the frozen run context does not match the AR(1) configuration"))
    _assert_context!(context, "welfare start")
    design_config = contest_design_config(config)
    run_fingerprint = _artifact_run_fingerprint(scientific, context)
    attempt = spec -> _run_outer(config, design_config, spec, hooks)
    completed = complete_contest_design(design_config;
        replacement_seed=config.replacement_parameter_seed,
        attempt,
        scheduler,
        workers,
        audit_schema=COMPLETE_DESIGN_SCHEMA_VERSION)
    blocks = [only(values(result.panels)).block_fingerprint
        for result in completed.results if !(result isa ContestFailure)]
    length(unique(blocks)) == length(blocks) || throw(ErrorException(
        "distinct AR(1) contests unexpectedly share a shock block"))
    _assert_context!(context, "welfare completion")
    return (scientific_fingerprint=scientific,
        configuration_fingerprint=run_fingerprint,
        source_commit=context.source.commit,
        input_tree_fingerprint=context.input_tree_fingerprint,
        environment_fingerprint=context.environment_fingerprint,
        outer_design=completed.plain_design,
        sampler_audit=completed.audit,
        design=completed.design,
        results=completed.results)
end

"""Run the fixed AR(1) design, retaining failed configurations in the audit."""
function run_welfare_contest(config::AR1ContestRunConfig;
    root::AbstractString=datadir(),
    workers::Integer=Threads.nthreads(), scheduler=threaded_scheduler,
    hooks::AR1RunnerHooks=AR1RunnerHooks(),
    gitpath::AbstractString=PROJECT_ROOT, lock_run::Bool=true)
    validate_run_config(config)
    worker_count = Int(workers)
    worker_count > 0 || throw(ArgumentError("workers must be positive"))
    context = _capture_run_context(config, gitpath)
    production = _is_production_config(config)
    if production
        worker_count == Threads.nthreads() || throw(ArgumentError(
            "production workers must match the active Julia thread count"))
        scheduler === threaded_scheduler || throw(ArgumentError(
            "production requires the built-in threaded scheduler"))
        _is_default_hooks(hooks) || throw(ArgumentError(
            "injected hooks are restricted to nonproduction pilots and tests"))
        lock_run || throw(ArgumentError(
            "production welfare execution requires the process-level lock"))
        validate_production_environment(config; project_root=gitpath,
            context)
    end
    action = () -> _run_welfare_unlocked(config, hooks;
        workers=worker_count, scheduler, context)
    run_fingerprint = _artifact_run_fingerprint(
        scientific_fingerprint(config), context)
    return lock_run ? _with_run_lock(action, root, run_fingerprint) : action()
end

function _warm_method!(config, spec, method, innovations, hooks)
    inputs = _model_inputs(config, spec)
    policy = hooks.build_policy(method, inputs.model_config, inputs.process,
        inputs.base_rule, config.g2egm_grids,
        _effective_g2egm_options(config),
        config.ptd_osa_options)
    hooks.validate_policy(policy, method, inputs.model_config, inputs.process)
    _compact_quadrature_audit(
        method, hooks.quadrature_audit(policy), inputs.base_rule)
    evaluation = hooks.evaluate(policy, method, inputs.model_config,
        inputs.process, inputs.initial, view(innovations, 1:1, :),
        PolicyActionAudit())
    _validate_evaluation(evaluation, method, 1)
    return nothing
end

function _prepare_timing(config, hooks, selection; context)
    scientific = scientific_fingerprint(config)
    scientific == context.scientific_fingerprint || throw(ArgumentError(
        "the frozen run context does not match the AR(1) timing configuration"))
    _assert_context!(context, "timing start")
    run_fingerprint = _artifact_run_fingerprint(scientific, context)
    design_config = contest_design_config(config)
    design = selection.design
    plain_design = hasproperty(selection, :plain_design) ?
        collect(selection.plain_design) : hasproperty(selection, :provenance) ?
            _plain_design(design, selection.provenance) : _plain_design(design)
    sampler_audit = hasproperty(selection, :audit) ? selection.audit :
        (schema=:internal_unscreened_design,
         final_design_fingerprint=stable_fingerprint(plain_design))
    failed_slots = hasproperty(sampler_audit, :failed_slots) ?
        Set(sampler_audit.failed_slots) : Set{Int}()
    design = filter(spec -> !(spec.id in failed_slots), design)
    blocks = [_validate_predraw(
        hooks.predraw(spec, design_config, AR1_PANEL), config)
              for spec in design]
    for (spec, block) in zip(design, blocks)
        _evaluator_domain_audit(config, spec, block)
    end
    length(unique(shock_fingerprint.(blocks))) == length(blocks) ||
        throw(ErrorException("distinct timing contests share a shock block"))
    return (; scientific, run_fingerprint, design, plain_design, sampler_audit,
        failed_contests=length(failed_slots),
        blocks,
        indexed=collect(enumerate(design)))
end

function _timing_header(prepared, context, workers::Integer)
    return (artifact_schema=TIMING_SCHEMA_VERSION,
        scientific_fingerprint=prepared.scientific,
        run_fingerprint=prepared.run_fingerprint,
        source_commit=context.source.commit,
        input_tree_fingerprint=context.input_tree_fingerprint,
        environment_fingerprint=context.environment_fingerprint,
        outer_design_fingerprint=stable_fingerprint(prepared.plain_design),
        sampler_audit_fingerprint=stable_fingerprint(prepared.sampler_audit),
        simulation_metadata=context.simulation_metadata,
        thread_count=Int(workers),
        timed_contests=length(prepared.design),
        failed_contests=prepared.failed_contests)
end

function _time_method(config, hooks, method::Symbol, prepared;
                      workers::Integer, scheduler, context)
    method in config.methods || throw(ArgumentError(
        "timing method $method is outside the configured solver set"))
    isempty(prepared.design) && return (panel=AR1_PANEL, method,
        total_seconds=0.0, status=:no_successful_contests)
    _warm_method!(config, first(prepared.design), method,
        first(prepared.blocks), hooks)
    outputs = Vector{Any}(undef, length(prepared.design))
    started = time_ns()
    scheduler(prepared.indexed, workers) do indexed_spec
        index, spec = indexed_spec
        inputs = _model_inputs(config, spec)
        policy = hooks.build_policy(method, inputs.model_config,
            inputs.process, inputs.base_rule, config.g2egm_grids,
            _effective_g2egm_options(config), config.ptd_osa_options)
        # Construction is part of the timed workload and the production
        # constructors enforce their own invariants (including the complete
        # G2EGM array fingerprint).  Re-running `validate_policy` here would
        # hash every G2EGM policy/value array a second time inside the clock,
        # although output validation is declared to be outside the interval.
        # The untimed warm-up above independently checks the method contract.
        outputs[index] = hooks.evaluate(policy, method,
            inputs.model_config, inputs.process, inputs.initial,
            prepared.blocks[index], PolicyActionAudit())
    end
    total_seconds = Float64(time_ns() - started) / 1.0e9
    for output in outputs
        _validate_evaluation(output, method, config.n_paths)
    end
    _assert_context!(context, "timing method $method")
    return (panel=AR1_PANEL, method, total_seconds)
end

function _run_timing(config, hooks, selection; workers, scheduler, context)
    prepared = _prepare_timing(config, hooks, selection; context)
    totals = [_time_method(config, hooks, method, prepared;
                  workers, scheduler, context)
              for method in config.methods]
    return merge(_timing_header(prepared, context, workers), (; totals))
end

"""
Return one uninterrupted total for each AR(1) method.

No timing sidecar is written. Production embeds the completed aggregate totals
in the sole final contest artifact.
"""
function run_timing_pass(config::AR1ContestRunConfig;
    workers::Integer=Threads.nthreads(), scheduler=threaded_scheduler,
    hooks::AR1RunnerHooks=AR1RunnerHooks(),
    gitpath::AbstractString=PROJECT_ROOT,
    lock_root::Union{Nothing,AbstractString}=nothing)
    validate_run_config(config)
    worker_count = Int(workers)
    worker_count > 0 || throw(ArgumentError("workers must be positive"))
    context = _capture_run_context(config, gitpath)
    production = _is_production_config(config)
    if production
        worker_count == Threads.nthreads() || throw(ArgumentError(
            "production workers must match the active Julia thread count"))
        scheduler === threaded_scheduler || throw(ArgumentError(
            "production requires the built-in threaded scheduler"))
        _is_default_hooks(hooks) || throw(ArgumentError(
            "injected hooks are restricted to nonproduction pilots and tests"))
        isnothing(lock_root) && throw(ArgumentError(
            "production timing execution requires the process-level lock"))
        validate_production_environment(config; project_root=gitpath,
            context)
    end
    action = function ()
        selection = _sample_ar1_contest_design(config, hooks;
            workers=worker_count, scheduler)
        return _run_timing(config, hooks, selection;
            workers=worker_count, scheduler, context)
    end
    isnothing(lock_root) && return action()
    run_fingerprint = _artifact_run_fingerprint(
        scientific_fingerprint(config), context)
    return _with_run_lock(action, lock_root, run_fingerprint)
end

"""Validate BLAS threading, the source tree, and AR(1) production inputs."""
function validate_production_environment(config::AR1ContestRunConfig;
    project_root::AbstractString=PROJECT_ROOT,
    context=nothing)
    validate_run_config(config)
    scientific_fingerprint(config) ==
        scientific_fingerprint(production_run_config()) || throw(ArgumentError(
            "production accepts only the locked 100-by-100 AR(1) configuration"))
    LinearAlgebra.BLAS.set_num_threads(1)
    LinearAlgebra.BLAS.get_num_threads() == 1 || throw(ErrorException(
        "production requires one BLAS thread"))
    _is_default_hooks(AR1RunnerHooks()) || throw(ErrorException(
        "the default AR(1) runner hook contract is inconsistent"))
    frozen = isnothing(context) ?
        _capture_run_context(config, project_root) : context
    abspath(project_root) == frozen.project_root || throw(ArgumentError(
        "the frozen AR(1) context belongs to a different project root"))
    frozen.scientific_fingerprint == scientific_fingerprint(config) ||
        throw(ArgumentError(
            "the frozen context belongs to a different AR(1) configuration"))
    _assert_context!(frozen, "production validation")
    !frozen.source.dirty || throw(ArgumentError(
        "production refuses a dirty scoped Ptd tree"))
    return (config=config, source=frozen.source, context=frozen)
end


"""Execute the locked paper-declared AR(1) contest and publish one artifact."""
function run_production_contest(config::AR1ContestRunConfig=production_run_config();
    root::AbstractString=datadir(),
    gitpath::AbstractString=PROJECT_ROOT, compress::Bool=true)
    context = _capture_run_context(config, gitpath)
    validate_production_environment(config; project_root=gitpath, context)
    hooks = AR1RunnerHooks()
    run_fingerprint = _artifact_run_fingerprint(
        scientific_fingerprint(config), context)
    action = function ()
        welfare = _run_welfare_unlocked(config, hooks;
            workers=Threads.nthreads(), scheduler=threaded_scheduler,
            context)
        selection = (design=welfare.design,
            audit=welfare.sampler_audit,
            plain_design=welfare.outer_design)
        timing = _run_timing(config, hooks, selection;
            workers=Threads.nthreads(), scheduler=threaded_scheduler,
            context)
        _assert_context!(context, "single-artifact publication")
        publication = publish_ar1_contest(config, welfare, timing;
            output_path=joinpath(abspath(root), "ar1-contest-welfare.jld2"),
            context, compress)
        return (; welfare, timing, publication)
    end
    return _with_run_lock(action, root, run_fingerprint)
end
