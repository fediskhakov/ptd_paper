"""
Pure, nonanticipative policy adapters for the scalar log-AR(1) contest.

The `:g2egm` adapter uses `Ptd.G2EGM`.
"""

import Distributions: Normal

"""Supertype for policies participating in the paper's AR(1) contest."""
abstract type AbstractAR1ContestPolicy end

"""Declared G2EGM grids, independent of evaluator shocks and solver state."""
struct AR1G2ContestGridConfiguration
    schema::Symbol
    asset_grid::Vector{Float64}
    post_log_income_grid::Vector{Float64}
    common_cash_grid::Vector{Float64}
    common_log_income_grid::Vector{Float64}
    barycentric_weight_floor::Float64

    function AR1G2ContestGridConfiguration(schema::Symbol,
        asset_grid, post_log_income_grid, common_cash_grid,
        common_log_income_grid; barycentric_weight_floor::Real=-0.25)
        assets = Float64.(collect(asset_grid))
        post_logs = Float64.(collect(post_log_income_grid))
        cash = Float64.(collect(common_cash_grid))
        common_logs = Float64.(collect(common_log_income_grid))
        weight_floor = Float64(barycentric_weight_floor)
        !isempty(String(schema)) || throw(ArgumentError(
            "The AR(1) G2EGM grid schema cannot be empty."))
        length(assets) >= 2 && first(assets) == 0.0 &&
            all(isfinite, assets) && all(diff(assets) .> 0.0) ||
            throw(ArgumentError(
                "The G2EGM asset grid must start at zero and increase strictly."))
        length(post_logs) >= 3 && isodd(length(post_logs)) &&
            all(isfinite, post_logs) && all(diff(post_logs) .> 0.0) &&
            post_logs[(length(post_logs) + 1) ÷ 2] == 0.0 ||
            throw(ArgumentError(
                "The post-decision log-income grid must be odd, increasing, and contain exact zero."))
        length(cash) >= 2 && first(cash) >= 0.0 &&
            all(isfinite, cash) && all(diff(cash) .> 0.0) ||
            throw(ArgumentError(
                "The common cash grid must be nonnegative and increase strictly."))
        length(common_logs) >= 3 && isodd(length(common_logs)) &&
            all(isfinite, common_logs) && all(diff(common_logs) .> 0.0) &&
            common_logs[(length(common_logs) + 1) ÷ 2] == 0.0 ||
            throw(ArgumentError(
                "The common log-income grid must be odd, increasing, and contain exact zero."))
        first(post_logs) == first(common_logs) &&
            last(post_logs) == last(common_logs) || throw(ArgumentError(
                "Post-decision and common log-income grids must share endpoints."))
        isfinite(weight_floor) && -1.0 < weight_floor <= 0.0 ||
            throw(ArgumentError(
                "The barycentric weight floor must lie in (-1,0]."))
        return new(schema, assets, post_logs, cash, common_logs, weight_floor)
    end
end

function _single_log_closed_grid(maximum::Real, points::Integer)
    upper, n = Float64(maximum), Int(points)
    isfinite(upper) && upper > 0.0 || throw(ArgumentError(
        "The asset-grid maximum must be finite and positive."))
    n >= 2 || throw(ArgumentError("A grid needs at least two points."))
    grid = exp.(range(0.0, log1p(upper); length=n)) .- 1.0
    grid[1], grid[end] = 0.0, upper
    return grid
end

function _shifted_log_closed_grid(minimum::Real, maximum::Real,
                                  points::Integer)
    lower, upper, n = Float64(minimum), Float64(maximum), Int(points)
    isfinite(lower) && isfinite(upper) && 0.0 <= lower < upper ||
        throw(ArgumentError("The shifted-log grid bounds are invalid."))
    n >= 2 || throw(ArgumentError("A grid needs at least two points."))
    grid = lower .+ exp.(range(0.0, log1p(upper - lower); length=n)) .- 1.0
    grid[1], grid[end] = lower, upper
    return grid
end

function _symmetric_sinh_grid(radius::Real, points::Integer, kappa::Real)
    bound, n, curvature = Float64(radius), Int(points), Float64(kappa)
    isfinite(bound) && bound > 0.0 || throw(ArgumentError(
        "The log-income radius must be finite and positive."))
    n >= 3 && isodd(n) || throw(ArgumentError(
        "A symmetric log-income grid requires an odd count of at least three."))
    isfinite(curvature) && curvature > 0.0 || throw(ArgumentError(
        "The sinh-grid curvature must be finite and positive."))
    u = range(-1.0, 1.0; length=n)
    grid = bound .* sinh.(curvature .* u) ./ sinh(curvature)
    grid[1], grid[(n + 1) ÷ 2], grid[end] = -bound, 0.0, bound
    return grid
end

"""Feasible-first pilot grids; never accepted by the production entry point."""
function pilot_ar1_g2_grid_configuration()
    log_radius = 3.0
    asset_maximum = 30_000.0
    cash_maximum = 1.10 * asset_maximum + exp(log_radius)
    return AR1G2ContestGridConfiguration(
        :ar1_g2egm_pilot_v1,
        _single_log_closed_grid(asset_maximum, 401),
        _symmetric_sinh_grid(log_radius, 31, 3.0),
        _shifted_log_closed_grid(3.0, cash_maximum, 301),
        _symmetric_sinh_grid(log_radius, 21, 2.5))
end

"""Return the G2EGM grids declared for the AR(1) exercise in the paper."""
function candidate_ar1_g2_grid_configuration()
    log_radius = 3.0
    asset_maximum = 30_000.0
    cash_maximum = 1.10 * asset_maximum + exp(log_radius)
    return AR1G2ContestGridConfiguration(
        :ar1_g2egm_candidate_reported_v1,
        _single_log_closed_grid(asset_maximum, 1_601),
        _symmetric_sinh_grid(log_radius, 61, 3.0),
        _shifted_log_closed_grid(3.0, cash_maximum, 1_201),
        _symmetric_sinh_grid(log_radius, 41, 2.5))
end

function _plain_ar1_g2_grid_configuration(
    grids::AR1G2ContestGridConfiguration)
    return (;
        schema=grids.schema,
        asset=(minimum=first(grids.asset_grid),
            maximum=last(grids.asset_grid), points=length(grids.asset_grid),
            transform=:single_log1p_closed_range,
            fingerprint=_sha_fingerprint(
                "ar1_g2_asset_grid_v1", join(_float_token.(grids.asset_grid), ','))),
        post_log_income=(minimum=first(grids.post_log_income_grid),
            maximum=last(grids.post_log_income_grid),
            points=length(grids.post_log_income_grid),
            transform=:symmetric_sinh,
            fingerprint=_sha_fingerprint("ar1_g2_post_log_grid_v1",
                join(_float_token.(grids.post_log_income_grid), ','))),
        common_cash=(minimum=first(grids.common_cash_grid),
            maximum=last(grids.common_cash_grid),
            points=length(grids.common_cash_grid),
            transform=:shifted_single_log1p_closed_range,
            fingerprint=_sha_fingerprint("ar1_g2_common_cash_grid_v1",
                join(_float_token.(grids.common_cash_grid), ','))),
        common_log_income=(minimum=first(grids.common_log_income_grid),
            maximum=last(grids.common_log_income_grid),
            points=length(grids.common_log_income_grid),
            transform=:symmetric_sinh,
            fingerprint=_sha_fingerprint("ar1_g2_common_log_grid_v1",
                join(_float_token.(grids.common_log_income_grid), ','))),
        barycentric_weight_floor=grids.barycentric_weight_floor,
    )
end

function _materialize_g2egm_grids(
    grids::AR1G2ContestGridConfiguration)
    backend = _g2egm_module()
    isdefined(backend, :AR1G2EGMGrid) || throw(ArgumentError(
        "Ptd.G2EGM is missing required API entry AR1G2EGMGrid."))
    constructor = getfield(backend, :AR1G2EGMGrid)
    return constructor(copy(grids.asset_grid),
        copy(grids.post_log_income_grid),
        copy(grids.common_cash_grid),
        copy(grids.common_log_income_grid))
end

function _materialize_g2egm_options(configuration::NamedTuple,
                                    weight_floor::Float64)
    backend = _g2egm_module()
    isdefined(backend, :AR1G2EGMOptions) || throw(ArgumentError(
        "Ptd.G2EGM is missing required API entry AR1G2EGMOptions."))
    hasproperty(configuration, :minimum_barycentric_weight) &&
        Float64(configuration.minimum_barycentric_weight) != weight_floor &&
        throw(ArgumentError(
            "G2EGM option minimum_barycentric_weight conflicts with the declared grid configuration."))
    effective = merge((minimum_barycentric_weight=weight_floor,),
        configuration)
    return getfield(backend, :AR1G2EGMOptions)(; pairs(effective)...)
end

"""PTD-SCEQ with a policy-local deterministic-plan cache."""
struct PTDSCEQAR1ContestPolicy <: AbstractAR1ContestPolicy
    config::StochasticPTDConfig
    process::LogAR1Process
    cache::PTDPlanCache
end

"""PTD-OSA with the canonical scalar Q5 rule and diagnostic aggregate."""
struct PTDOSAAR1ContestPolicy <: AbstractAR1ContestPolicy
    config::StochasticPTDConfig
    process::LogAR1Process
    base_rule::GaussHermiteRule{1,5}
    options::PTDOSAOptions
    cache::PTDPlanCache
    diagnostics::_PTDOSAContestDiagnostics
end

"""Solved pure-Julia G2EGM adapter registered under the paper's `:g2egm` id."""
struct G2EGMAR1ContestPolicy{S,G,O} <: AbstractAR1ContestPolicy
    config::StochasticPTDConfig
    process::LogAR1Process
    base_rule::GaussHermiteRule{1,5}
    grids::G
    options::O
    solution::S
    policy_fingerprint::String
end

"""Bundle of the three AR(1) contest policies."""
struct AR1ContestPolicies{D}
    ptd_sceq::PTDSCEQAR1ContestPolicy
    ptd_osa::PTDOSAAR1ContestPolicy
    g2egm::D
end

const AR1_G2EGM_REQUIRED_API = (
    :build_ar1_solution,
    :action,
    :validate_alignment,
    :diagnostics,
    :reset_query_diagnostics!,
    :quadrature_audit,
    :solution_fingerprint,
)

function _g2egm_module()
    isdefined(@__MODULE__, :G2EGM) || throw(ArgumentError(
        "The AR(1) G2EGM contest requires the pure-Julia Ptd.G2EGM module; " *
        "the legacy DcEgm.AgentAR1 solver is not an admissible fallback."))
    backend = getfield(@__MODULE__, :G2EGM)
    backend isa Module || throw(ErrorException(
        "Ptd.G2EGM is not a Julia module"))
    missing = Symbol[name for name in AR1_G2EGM_REQUIRED_API
                     if !isdefined(backend, name)]
    isempty(missing) || throw(ArgumentError(
        "Ptd.G2EGM is missing required API entries: $(join(missing, ", "))"))
    return backend
end

function _validate_ar1_process(process::AbstractIncomeProcess)
    process isa LogAR1Process || throw(ArgumentError(
        "The scalar persistent contest requires LogAR1Process."))
    ar1 = process::LogAR1Process
    ar1.unconditional_log_mean == 0.0 || throw(ArgumentError(
        "The locked AR(1) comparator currently requires unconditional log mean zero."))
    return ar1
end

function _validate_ar1_observed(config::StochasticPTDConfig,
                                observed::ObservedState)
    _validate_observed(config, observed)
    observed.exogenous isa LogAR1State || throw(ArgumentError(
        "An AR(1) contest policy requires ObservedState(..., LogAR1State(...))."))
    return nothing
end

function _validate_ar1_five_node_base_rule(process::LogAR1Process, rule)
    rule isa GaussHermiteRule{1,5} || throw(ArgumentError(
        "The scalar AR(1) contest requires exactly five Gauss-Hermite nodes."))
    expected = gauss_hermite_rule(process, 5)
    _quadrature_bits(rule.nodes) == _quadrature_bits(expected.nodes) &&
        _quadrature_bits(rule.weights) == _quadrature_bits(expected.weights) ||
        throw(ArgumentError(
            "The supplied AR(1) rule is not the canonical scalar Q5 rule."))
    return rule::GaussHermiteRule{1,5}
end

function _resolved_ar1_rule(process::LogAR1Process, base_rule)
    rule = isnothing(base_rule) ? gauss_hermite_rule(process, 5) : base_rule
    return _validate_ar1_five_node_base_rule(process, rule)
end

function _ar1_policy_scope(policy::AbstractAR1ContestPolicy,
                           expected_method::Symbol,
                           method::Symbol,
                           config::StochasticPTDConfig,
                           process::AbstractIncomeProcess)
    method === expected_method || throw(ArgumentError(
        "Policy $(typeof(policy)) must be registered as $expected_method, not $method."))
    ar1 = _validate_ar1_process(process)
    _config_token(policy.config) == _config_token(config) ||
        throw(ArgumentError(
            "The AR(1) policy configuration does not match the requested contest."))
    _process_token(policy.process) == _process_token(ar1) ||
        throw(ArgumentError(
            "The AR(1) policy process does not match the requested contest."))
    return ar1
end

"""Validate that an AR(1) policy matches the declared model and process."""
function validate_ar1_contest_policy_alignment(
    policy::PTDSCEQAR1ContestPolicy, method::Symbol,
    config::StochasticPTDConfig, process::AbstractIncomeProcess)
    _ar1_policy_scope(policy, :ptd_sceq, method, config, process)
    return nothing
end

function validate_ar1_contest_policy_alignment(
    policy::PTDOSAAR1ContestPolicy, method::Symbol,
    config::StochasticPTDConfig, process::AbstractIncomeProcess)
    ar1 = _ar1_policy_scope(policy, :ptd_osa, method, config, process)
    _validate_ar1_five_node_base_rule(ar1, policy.base_rule)
    return nothing
end

function validate_ar1_contest_policy_alignment(
    policy::G2EGMAR1ContestPolicy, method::Symbol,
    config::StochasticPTDConfig, process::AbstractIncomeProcess)
    ar1 = _ar1_policy_scope(policy, :g2egm, method, config, process)
    _validate_ar1_five_node_base_rule(ar1, policy.base_rule)
    backend = _g2egm_module()
    if isnothing(policy.options)
        backend.validate_alignment(policy.solution, config, ar1,
            policy.base_rule, policy.grids)
    else
        backend.validate_alignment(policy.solution, config, ar1,
            policy.base_rule, policy.grids; options=policy.options)
    end
    expected = String(backend.solution_fingerprint(policy.solution))
    policy.policy_fingerprint == expected || throw(ArgumentError(
        "The AR(1) G2EGM policy fingerprint does not match its solution."))
    return nothing
end

"""
    build_ar1_contest_policy(method, config, process; kwargs...)

Construct one scalar AR(1) competitor. `g2egm_grids` is mandatory for
`:g2egm`; a declared contest grid is converted to the solver-owned
`Ptd.G2EGM.AR1G2EGMGrid` at this boundary.
"""
function build_ar1_contest_policy(method::Symbol,
    config::StochasticPTDConfig, process::AbstractIncomeProcess;
    base_rule=nothing, ptd_osa_options::PTDOSAOptions=PTDOSAOptions(),
    g2egm_grids=nothing, g2egm_options=nothing)::AbstractAR1ContestPolicy
    ar1 = _validate_ar1_process(process)

    if method === :ptd_sceq
        !isnothing(base_rule) &&
            _validate_ar1_five_node_base_rule(ar1, base_rule)
        return PTDSCEQAR1ContestPolicy(config, ar1,
            PTDPlanCache(verify_integrity_on_hit=false))
    elseif method === :ptd_osa
        rule = _resolved_ar1_rule(ar1, base_rule)
        return PTDOSAAR1ContestPolicy(config, ar1, rule, ptd_osa_options,
            PTDPlanCache(verify_integrity_on_hit=false),
            _PTDOSAContestDiagnostics())
    elseif method === :g2egm
        isnothing(g2egm_grids) && throw(ArgumentError(
            "The AR(1) G2EGM comparator requires an explicit grid specification."))
        rule = _resolved_ar1_rule(ar1, base_rule)
        backend = _g2egm_module()
        solver_grids = g2egm_grids isa AR1G2ContestGridConfiguration ?
            _materialize_g2egm_grids(g2egm_grids) : g2egm_grids
        solver_options = g2egm_options isa NamedTuple ?
            _materialize_g2egm_options(g2egm_options,
                g2egm_grids isa AR1G2ContestGridConfiguration ?
                    g2egm_grids.barycentric_weight_floor : -0.25) :
            g2egm_options
        solution = isnothing(solver_options) ?
            backend.build_ar1_solution(config, ar1;
                base_rule=rule, grid=solver_grids) :
            backend.build_ar1_solution(config, ar1;
                base_rule=rule, grid=solver_grids,
                options=solver_options)
        if isnothing(solver_options)
            backend.validate_alignment(solution, config, ar1,
                rule, solver_grids)
        else
            backend.validate_alignment(solution, config, ar1,
                rule, solver_grids; options=solver_options)
        end
        fingerprint = String(backend.solution_fingerprint(solution))
        occursin(r"^[0-9a-f]{64}$", fingerprint) || throw(ArgumentError(
            "Ptd.G2EGM.solution_fingerprint must return lowercase SHA-256 hex."))
        return G2EGMAR1ContestPolicy(config, ar1, rule, solver_grids,
            solver_options, solution, fingerprint)
    end
    throw(ArgumentError(
        "Unknown AR(1) contest method $method; use :ptd_sceq, :ptd_osa, or :g2egm."))
end

function (policy::PTDSCEQAR1ContestPolicy)(
    observed::ObservedState)::ActionResult
    _validate_ar1_observed(policy.config, observed)
    return ptd_sceq(policy.config, policy.process, observed;
        cache=policy.cache, evaluate_objective=false)
end

function (policy::PTDOSAAR1ContestPolicy)(
    observed::ObservedState)::ActionResult
    _validate_ar1_observed(policy.config, observed)
    result = ptd_osa_with_diagnostics(policy.config, policy.process, observed,
        policy.base_rule; options=policy.options, cache=policy.cache)
    _record_ptd_osa_diagnostics!(policy.diagnostics, result.diagnostics)
    return result.action
end

function (policy::G2EGMAR1ContestPolicy)(
    observed::ObservedState)::ActionResult
    _validate_ar1_observed(policy.config, observed)
    consumption = Float64(
        _g2egm_module().action(policy.solution, observed))
    tolerance = 64eps(max(observed.cash, 1.0))
    isfinite(consumption) &&
        -tolerance <= consumption <= observed.cash + tolerance ||
        throw(ErrorException(
            "The AR(1) G2EGM policy returned infeasible consumption."))
    consumption = clamp(consumption, 0.0, observed.cash)
    return ActionResult(consumption, observed.cash - consumption,
        nothing, :g2egm, policy.policy_fingerprint)
end

"""Construct all three policy adapters used by the AR(1) contest."""
function build_ar1_contest_policies(config::StochasticPTDConfig,
    process::AbstractIncomeProcess; base_rule=nothing,
    ptd_osa_options::PTDOSAOptions=PTDOSAOptions(), g2egm_grids=nothing,
    g2egm_options=nothing)
    ar1 = _validate_ar1_process(process)
    rule = _resolved_ar1_rule(ar1, base_rule)
    return AR1ContestPolicies(
        build_ar1_contest_policy(:ptd_sceq, config, ar1;
            base_rule=rule, ptd_osa_options),
        build_ar1_contest_policy(:ptd_osa, config, ar1;
            base_rule=rule, ptd_osa_options),
        build_ar1_contest_policy(:g2egm, config, ar1;
            base_rule=rule, ptd_osa_options, g2egm_grids, g2egm_options))
end

"""Return state-only action closures for an AR(1) policy bundle."""
ar1_contest_actions(policies::AR1ContestPolicies) =
    (ptd_sceq=policies.ptd_sceq,
     ptd_osa=policies.ptd_osa,
     g2egm=policies.g2egm)

"""Return serializable diagnostics for an AR(1) contest policy."""
ar1_contest_diagnostics(policy::PTDSCEQAR1ContestPolicy) =
    _cache_diagnostics(:ptd_sceq, policy.cache)

function ar1_contest_diagnostics(policy::PTDOSAAR1ContestPolicy)
    verified = verify_plan_cache_integrity!(policy.cache)
    stats = plan_cache_statistics(policy.cache)
    aggregate = policy.diagnostics
    return lock(aggregate.lock) do
        (method=:ptd_osa, cache_entries=stats.entries,
         cache_hits=stats.hits, cache_misses=stats.misses,
         cache_builds=stats.builds, verified_cache_entries=verified,
         calls=aggregate.calls,
         certified_calls=aggregate.certified_calls,
         certification_failures=aggregate.certification_failures,
         fallback_uses=aggregate.fallback_uses,
         total_structural_boundaries=aggregate.total_structural_boundaries,
         maximum_structural_boundaries=aggregate.maximum_structural_boundaries,
         total_interior_roots=aggregate.total_interior_roots,
         maximum_interior_roots=aggregate.maximum_interior_roots,
         total_floor_derivative_exclusions=
            aggregate.total_floor_derivative_exclusions,
         total_candidates=aggregate.total_candidates,
         maximum_candidates=aggregate.maximum_candidates,
         total_cells=aggregate.total_cells,
         total_nodes=aggregate.total_nodes,
         maximum_certification_gap=aggregate.maximum_certification_gap)
    end
end

function _checkpoint_g2egm_diagnostic(value, field::Symbol=:root)
    if value isa AbstractFloat
        isfinite(value) && return Float64(value)
        value == Inf && startswith(String(field), "minimum_") &&
            return nothing
        throw(ErrorException(
            "G2EGM diagnostic $field is nonfinite and is not an unavailable minimum sentinel."))
    elseif value isa NamedTuple
        return (; (name => _checkpoint_g2egm_diagnostic(
                    getproperty(value, name), name) for name in keys(value))...)
    elseif value isa AbstractDict
        return Dict(key => _checkpoint_g2egm_diagnostic(
                        item, Symbol(key)) for (key, item) in pairs(value))
    elseif value isa Tuple
        return map(item -> _checkpoint_g2egm_diagnostic(item, field), value)
    elseif value isa AbstractArray
        return map(item -> _checkpoint_g2egm_diagnostic(item, field), value)
    elseif value === nothing || value isa Union{Bool,Integer,Symbol,AbstractString}
        return value
    end
    throw(ErrorException(
        "G2EGM diagnostic $field has unsupported type $(typeof(value))."))
end

function ar1_contest_diagnostics(policy::G2EGMAR1ContestPolicy)
    return (method=:g2egm,
        implementation=:pure_julia_g2egm,
        policy_fingerprint=policy.policy_fingerprint,
        solver=_checkpoint_g2egm_diagnostic(
            _g2egm_module().diagnostics(policy.solution)))
end

"""Reset mutable diagnostics for an AR(1) contest policy."""
function reset_ar1_contest_diagnostics!(
    policy::PTDSCEQAR1ContestPolicy)
    reset_plan_cache_statistics!(policy.cache)
    return ar1_contest_diagnostics(policy)
end

function reset_ar1_contest_diagnostics!(
    policy::PTDOSAAR1ContestPolicy)
    reset_plan_cache_statistics!(policy.cache)
    _reset_ptd_osa_diagnostics!(policy.diagnostics)
    return ar1_contest_diagnostics(policy)
end

function reset_ar1_contest_diagnostics!(
    policy::G2EGMAR1ContestPolicy)
    _g2egm_module().reset_query_diagnostics!(policy.solution)
    return ar1_contest_diagnostics(policy)
end

"""Describe the quadrature rule used by an AR(1) contest policy."""
function ar1_contest_quadrature_audit(policy::PTDSCEQAR1ContestPolicy)
    validate_ar1_contest_policy_alignment(
        policy, :ptd_sceq, policy.config, policy.process)
    return (method=:ptd_sceq, uses_quadrature=false)
end

function ar1_contest_quadrature_audit(policy::PTDOSAAR1ContestPolicy)
    validate_ar1_contest_policy_alignment(
        policy, :ptd_osa, policy.config, policy.process)
    transformed = DcEgm.distribution_quadrature(
        Normal(0.0, policy.process.log_std), policy.base_rule)
    return (method=:ptd_osa, uses_quadrature=true,
        nodes_per_dimension=policy.base_rule.nodes_per_dimension,
        base_nodes=policy.base_rule.nodes,
        base_weights=policy.base_rule.weights,
        base_node_bits=_quadrature_bits(policy.base_rule.nodes),
        base_weight_bits=_quadrature_bits(policy.base_rule.weights),
        distribution_nodes=transformed.nodes,
        distribution_weights=transformed.weights,
        distribution_node_bits=_quadrature_bits(transformed.nodes),
        distribution_weight_bits=_quadrature_bits(transformed.weights))
end

function ar1_contest_quadrature_audit(policy::G2EGMAR1ContestPolicy)
    validate_ar1_contest_policy_alignment(
        policy, :g2egm, policy.config, policy.process)
    audit = _g2egm_module().quadrature_audit(policy.solution)
    nodes = audit.standard_normal_nodes
    weights = audit.normalized_weights
    return merge((method=:g2egm, uses_quadrature=true,
        base_nodes=nodes,
        base_weights=weights,
        base_node_bits=_quadrature_bits(nodes),
        base_weight_bits=_quadrature_bits(weights)), audit)
end
