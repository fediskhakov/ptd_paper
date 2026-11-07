"""
Pure, nonanticipative policy adapters for the paper's IID welfare contest.

Each adapter is constructed before hidden paths are evaluated and is callable
with exactly one `ObservedState`.
"""

import Distributions: LogNormal, params

export validate_iid_contest_policy_alignment

const IID_DCEGM_ASSET_MAXIMUM = 60_000.0
const IID_DCEGM_DIAGNOSTIC_POINTS = 3_201
const IID_DCEGM_REPORTED_POINTS = 6_401
const IID_DCEGM_VALIDATION_POINTS = 12_801
const IID_DCEGM_GRID_SCHEMA = :single_log1p_closed_range

"""Supertype for policies participating in the paper's IID contest."""
abstract type AbstractIIDContestPolicy end

"""PTD-SCEQ adapter with a policy-local deterministic-plan cache."""
struct PTDSCEQIIDContestPolicy <: AbstractIIDContestPolicy
    config::StochasticPTDConfig
    process::IIDLognormalProcess
    cache::PTDPlanCache
end

"""Mutable, thread-safe aggregate of PTD-OSA structural-search diagnostics."""
mutable struct _PTDOSAContestDiagnostics
    calls::Int
    certified_calls::Int
    certification_failures::Int
    fallback_uses::Int
    total_structural_boundaries::Int
    maximum_structural_boundaries::Int
    total_interior_roots::Int
    maximum_interior_roots::Int
    total_floor_derivative_exclusions::Int
    total_candidates::Int
    maximum_candidates::Int
    total_cells::Int
    total_nodes::Int
    maximum_certification_gap::Float64
    lock::ReentrantLock
end

_PTDOSAContestDiagnostics() = _PTDOSAContestDiagnostics(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, ReentrantLock())

"""PTD-OSA adapter with a policy-local plan cache and call audit."""
struct PTDOSAIIDContestPolicy <: AbstractIIDContestPolicy
    config::StochasticPTDConfig
    process::IIDLognormalProcess
    base_rule::GaussHermiteRule{1,5}
    options::PTDOSAOptions
    cache::PTDPlanCache
    diagnostics::_PTDOSAContestDiagnostics
end

"""Solved-once IID DCEGM adapter."""
struct DCEGMIIDContestPolicy <: AbstractIIDContestPolicy
    config::StochasticPTDConfig
    process::IIDLognormalProcess
    base_rule::GaussHermiteRule{1,5}
    egm_rule::DcEgm.EGMQuadratureRule{5}
    solution::DcEgm.EGMSolution{DcEgm.AgentIID}
    asset_grid::Vector{Float64}
    policy_fingerprint::String
end

"""Bundle of the three IID contest policies."""
struct IIDContestPolicies
    ptd_sceq::PTDSCEQIIDContestPolicy
    ptd_osa::PTDOSAIIDContestPolicy
    dcegm::DCEGMIIDContestPolicy
end

"""
    iid_dcegm_asset_grid(points=IID_DCEGM_REPORTED_POINTS;
                         asset_maximum=IID_DCEGM_ASSET_MAXIMUM)

Construct the paper's single-log end-of-period asset grid, matching the
`scaleGrid(..., logorder=1)` convention in DCEGM.jl,

`a(s) = exp(s * log(1 + a_max)) - 1`,  `s in [0,1]`.

The transformation keeps fine absolute resolution near the borrowing limit
while spanning the ex-ante resource bound needed by the high-saving CRRA
corner. The grid depends only on declared numerical settings, never on an
evaluation seed or a hidden shock path.
"""
function iid_dcegm_asset_grid(points::Integer=IID_DCEGM_REPORTED_POINTS;
    asset_maximum::Real=IID_DCEGM_ASSET_MAXIMUM)
    n = Int(points)
    upper = Float64(asset_maximum)
    n >= 2 || throw(ArgumentError(
        "the IID DCEGM asset grid requires at least two points"))
    isfinite(upper) && upper > 0.0 || throw(ArgumentError(
        "the IID DCEGM asset maximum must be finite and positive"))
    grid = exp.(range(0.0, log1p(upper); length=n)) .- 1.0
    # Preserve the declared closed domain bit-for-bit despite exp/log rounding.
    grid[1] = 0.0
    grid[end] = upper
    all(diff(grid) .> 0.0) || throw(ArgumentError(
        "the requested IID DCEGM log grid is not strictly increasing in Float64"))
    return grid
end

"""The reported/timed ordinary IID DCEGM grid."""
iid_contest_asset_grid() = iid_dcegm_asset_grid()

_quadrature_scalar(value::Real) = Float64(value)
function _quadrature_scalar(value::Tuple)
    length(value) == 1 || throw(ArgumentError(
        "An IID quadrature node must have exactly one coordinate."))
    return Float64(only(value))
end
_quadrature_bits(values) = Tuple(reinterpret(UInt64,
    _quadrature_scalar(value)) for value in values)

function _validate_mean_one_iid(process::AbstractIncomeProcess)
    process isa IIDLognormalProcess || throw(ArgumentError(
        "The nested policy contest accepts only IID lognormal income."))
    iid = process::IIDLognormalProcess
    mean_income = exp(iid.log_mean + iid.log_std^2 / 2.0)
    isapprox(mean_income, 1.0; rtol=8eps(Float64), atol=8eps(Float64)) ||
        throw(ArgumentError(
            "The IID contest requires the mean-one lognormal parameterization."))
    return iid
end

function _validate_iid_observed(config::StochasticPTDConfig,
    observed::ObservedState)
    _validate_observed(config, observed)
    observed.exogenous isa IIDState || throw(ArgumentError(
        "An IID contest policy requires ObservedState(..., IIDState())."))
    return nothing
end

function _validate_contest_asset_grid(asset_grid)
    values = Float64[value for value in asset_grid]
    length(values) >= 2 || throw(ArgumentError(
        "The DCEGM asset grid must contain at least two points."))
    first(values) == 0.0 || throw(ArgumentError(
        "The DCEGM asset grid must start at zero."))
    all(isfinite, values) && all(>=(0.0), values) &&
        all(diff(values) .> 0.0) || throw(ArgumentError(
            "The DCEGM asset grid must be finite, nonnegative, and increasing."))
    return values
end

function _validate_five_node_base_rule(process::IIDLognormalProcess, rule)
    rule isa GaussHermiteRule{1,5} || throw(ArgumentError(
        "The IID contest requires exactly five scalar Gauss-Hermite nodes."))
    expected = gauss_hermite_rule(process, 5)
    _quadrature_bits(rule.nodes) == _quadrature_bits(expected.nodes) &&
        _quadrature_bits(rule.weights) == _quadrature_bits(expected.weights) ||
        throw(ArgumentError(
            "The supplied rule is not the canonical five-node Gauss-Hermite base rule."))
    return rule::GaussHermiteRule{1,5}
end

function _resolved_base_rule(process::IIDLognormalProcess, base_rule)
    rule = isnothing(base_rule) ? gauss_hermite_rule(process, 5) : base_rule
    return _validate_five_node_base_rule(process, rule)
end

function _validate_egm_alignment(config::StochasticPTDConfig,
    process::IIDLognormalProcess, solution::DcEgm.EGMSolution{DcEgm.AgentIID},
    asset_grid::Vector{Float64})
    agent = solution.agent
    fields = agent.bf
    fields.β == config.beta && fields.γ == config.gamma &&
        fields.r == config.return_rate && fields.xFloor == config.cash_floor &&
        fields.tMax == config.terminal.final_date || throw(ErrorException(
            "The DCEGM model is not aligned with the contest configuration."))
    fields.a == asset_grid || throw(ErrorException(
        "The solved DCEGM asset grid is not the requested contest grid."))
    location, scale = params(agent.d)
    Float64(location) == process.log_mean && Float64(scale) == process.log_std ||
        throw(ErrorException(
            "The solved DCEGM distribution is not the contest income process."))
    return nothing
end

function _dcegm_policy_fingerprint(config::StochasticPTDConfig,
    process::IIDLognormalProcess, asset_grid::Vector{Float64},
    base_rule::GaussHermiteRule{1,5}, egm_rule::DcEgm.EGMQuadratureRule{5})
    base_token = join((join(_float_token.(node), ',') * ":" *
        _float_token(weight) for (node, weight) in
        zip(base_rule.nodes, base_rule.weights)), ';')
    transformed_token = join((_float_token(node) * ":" * _float_token(weight)
        for (node, weight) in zip(egm_rule.nodes, egm_rule.weights)), ';')
    grid_token = join(_float_token.(asset_grid), ',')
    return _sha_fingerprint("iid_dcegm_policy_v1", _config_token(config),
        _process_token(process), base_token, transformed_token, grid_token)
end

function _validate_policy_scope(policy::AbstractIIDContestPolicy,
    expected_method::Symbol, method::Symbol, config::StochasticPTDConfig,
    process::AbstractIncomeProcess)
    method === expected_method || throw(ArgumentError(
        "Policy $(typeof(policy)) must be registered as $expected_method, not $method."))
    iid = _validate_mean_one_iid(process)
    _config_token(policy.config) == _config_token(config) ||
        throw(ArgumentError(
            "The policy configuration does not match the requested contest configuration."))
    _process_token(policy.process) == _process_token(iid) ||
        throw(ArgumentError(
            "The policy income process does not match the requested IID panel."))
    return iid
end

"""
    validate_iid_contest_policy_alignment(policy, method, config, process)

Fail-closed post-construction check used by the production runner. It verifies
the registered method and exact contest configuration/process. PTD-OSA additionally
checks its canonical five-node rule. DCEGM also checks the solved preference,
horizon, distribution, asset grid, transformed income nodes, and stable policy
fingerprint.
"""
function validate_iid_contest_policy_alignment(
    policy::PTDSCEQIIDContestPolicy, method::Symbol,
    config::StochasticPTDConfig, process::AbstractIncomeProcess)
    _validate_policy_scope(policy, :ptd_sceq, method, config, process)
    return nothing
end

function validate_iid_contest_policy_alignment(
    policy::PTDOSAIIDContestPolicy, method::Symbol,
    config::StochasticPTDConfig, process::AbstractIncomeProcess)
    iid = _validate_policy_scope(policy, :ptd_osa, method, config, process)
    _validate_five_node_base_rule(iid, policy.base_rule)
    return nothing
end

function validate_iid_contest_policy_alignment(
    policy::DCEGMIIDContestPolicy, method::Symbol,
    config::StochasticPTDConfig, process::AbstractIncomeProcess)
    iid = _validate_policy_scope(policy, :dcegm, method, config, process)
    _validate_five_node_base_rule(iid, policy.base_rule)
    _validate_contest_asset_grid(policy.asset_grid)
    _validate_egm_alignment(
        policy.config, policy.process, policy.solution, policy.asset_grid)
    expected_rule = DcEgm.distribution_quadrature(
        LogNormal(iid.log_mean, iid.log_std), policy.base_rule)
    _quadrature_bits(policy.egm_rule.nodes) ==
        _quadrature_bits(expected_rule.nodes) &&
        _quadrature_bits(policy.egm_rule.weights) ==
        _quadrature_bits(expected_rule.weights) || throw(ArgumentError(
            "The DCEGM transformed quadrature rule does not match its IID panel."))
    expected_fingerprint = _dcegm_policy_fingerprint(policy.config,
        policy.process, policy.asset_grid, policy.base_rule, policy.egm_rule)
    policy.policy_fingerprint == expected_fingerprint || throw(ArgumentError(
        "The DCEGM policy fingerprint does not match its solved inputs."))
    return nothing
end

"""
    build_iid_contest_policy(method, config, process; kwargs...)

Construct exactly one IID competitor. `:ptd_sceq` and `:ptd_osa` each own a
separate `PTDPlanCache`; `:dcegm` constructs and solves one IID EGM model once.
The DCEGM defaults are the reported 6,401-point single-log grid on
`[0,60_000]` and the same canonical five-node standard-normal rule used by
PTD-OSA.
"""
function build_iid_contest_policy(method::Symbol,
    config::StochasticPTDConfig, process::AbstractIncomeProcess;
    asset_grid=nothing, base_rule=nothing,
    ptd_osa_options::PTDOSAOptions=PTDOSAOptions())::AbstractIIDContestPolicy
    iid = _validate_mean_one_iid(process)

    if method === :ptd_sceq
        !isnothing(asset_grid) && _validate_contest_asset_grid(asset_grid)
        !isnothing(base_rule) && _validate_five_node_base_rule(iid, base_rule)
        return PTDSCEQIIDContestPolicy(config, iid,
            PTDPlanCache(verify_integrity_on_hit=false))
    elseif method === :ptd_osa
        !isnothing(asset_grid) && _validate_contest_asset_grid(asset_grid)
        rule = _resolved_base_rule(iid, base_rule)
        return PTDOSAIIDContestPolicy(config, iid, rule, ptd_osa_options,
            PTDPlanCache(verify_integrity_on_hit=false),
            _PTDOSAContestDiagnostics())
    elseif method === :dcegm
        rule = _resolved_base_rule(iid, base_rule)
        assets = isnothing(asset_grid) ? iid_contest_asset_grid() :
            _validate_contest_asset_grid(asset_grid)
        distribution = LogNormal(iid.log_mean, iid.log_std)
        egm_rule = DcEgm.distribution_quadrature(distribution, rule)
        _quadrature_bits(egm_rule.weights) == _quadrature_bits(rule.weights) ||
            throw(ErrorException(
                "DCEGM did not preserve the shared normalized quadrature weights."))
        agent = DcEgm.build_agent(model="iid", β=config.beta,
            γ=config.gamma, r=config.return_rate, xFloor=config.cash_floor,
            tMax=config.terminal.final_date, a=copy(assets), d=distribution)
        solution = DcEgm.solve_egm!(agent; quadrature=egm_rule)
        _validate_egm_alignment(config, iid, solution, assets)
        fingerprint = _dcegm_policy_fingerprint(
            config, iid, assets, rule, egm_rule)
        return DCEGMIIDContestPolicy(config, iid, rule, egm_rule,
            solution, assets, fingerprint)
    end
    throw(ArgumentError(
        "Unknown IID contest method $method; use :ptd_sceq, :ptd_osa, or :dcegm."))
end

function (policy::PTDSCEQIIDContestPolicy)(observed::ObservedState)::ActionResult
    _validate_iid_observed(policy.config, observed)
    return ptd_sceq(policy.config, policy.process, observed;
        cache=policy.cache, evaluate_objective=false)
end

function _record_ptd_osa_diagnostics!(aggregate::_PTDOSAContestDiagnostics,
    diagnostics::PTDOSADiagnostics)
    lock(aggregate.lock) do
        boundary_count = length(diagnostics.structural_boundaries)
        root_count = length(diagnostics.interior_roots)
        aggregate.calls += 1
        aggregate.certified_calls += diagnostics.certified
        aggregate.certification_failures += !diagnostics.certified
        aggregate.fallback_uses += diagnostics.fallback_used
        aggregate.total_structural_boundaries += boundary_count
        aggregate.maximum_structural_boundaries = max(
            aggregate.maximum_structural_boundaries, boundary_count)
        aggregate.total_interior_roots += root_count
        aggregate.maximum_interior_roots = max(
            aggregate.maximum_interior_roots, root_count)
        aggregate.total_floor_derivative_exclusions +=
            diagnostics.floor_derivative_exclusions
        aggregate.total_candidates += diagnostics.candidate_count
        aggregate.maximum_candidates = max(
            aggregate.maximum_candidates, diagnostics.candidate_count)
        aggregate.total_cells += diagnostics.cell_count
        aggregate.total_nodes += diagnostics.node_count
        aggregate.maximum_certification_gap = max(
            aggregate.maximum_certification_gap,
            diagnostics.certification_gap)
    end
    return nothing
end

function (policy::PTDOSAIIDContestPolicy)(observed::ObservedState)::ActionResult
    _validate_iid_observed(policy.config, observed)
    result = ptd_osa_with_diagnostics(policy.config, policy.process, observed,
        policy.base_rule; options=policy.options, cache=policy.cache)
    _record_ptd_osa_diagnostics!(policy.diagnostics, result.diagnostics)
    return result.action
end

function (policy::DCEGMIIDContestPolicy)(
    observed::ObservedState)::ActionResult
    _validate_iid_observed(policy.config, observed)
    consumption = observed.date == policy.config.terminal.final_date ?
        observed.cash : DcEgm.action(
            policy.solution, observed.cash; t=observed.date)
    tolerance = 64eps(max(observed.cash, 1.0))
    -tolerance <= consumption <= observed.cash + tolerance || throw(ErrorException(
        "DCEGM returned infeasible consumption for the observed state."))
    consumption = clamp(consumption, 0.0, observed.cash)
    # The contest evaluator needs the policy action, not a second interpolated
    # value query. `nothing` records that the objective was deliberately not
    # evaluated, keeping the DCEGM workload and timer free of unused work.
    return ActionResult(consumption, observed.cash - consumption,
        nothing, :dcegm, policy.policy_fingerprint)
end

"""Construct all IID methods with independent caches and solver state."""
function build_iid_contest_policies(config::StochasticPTDConfig,
    process::AbstractIncomeProcess; asset_grid=nothing, base_rule=nothing,
    ptd_osa_options::PTDOSAOptions=PTDOSAOptions())
    iid = _validate_mean_one_iid(process)
    rule = _resolved_base_rule(iid, base_rule)
    return IIDContestPolicies(
        build_iid_contest_policy(:ptd_sceq, config, iid;
            asset_grid=asset_grid, base_rule=rule, ptd_osa_options=ptd_osa_options),
        build_iid_contest_policy(:ptd_osa, config, iid;
            asset_grid=asset_grid, base_rule=rule, ptd_osa_options=ptd_osa_options),
        build_iid_contest_policy(:dcegm, config, iid;
            asset_grid=asset_grid, base_rule=rule, ptd_osa_options=ptd_osa_options))
end

"""Return the exact named tuple expected by `evaluate_contest_paths`."""
iid_contest_actions(policies::IIDContestPolicies) =
    (ptd_sceq=policies.ptd_sceq,
     ptd_osa=policies.ptd_osa,
     dcegm=policies.dcegm)

function _cache_diagnostics(method::Symbol, cache::PTDPlanCache)
    verified = verify_plan_cache_integrity!(cache)
    stats = plan_cache_statistics(cache)
    return (method=method, cache_entries=stats.entries,
        cache_hits=stats.hits, cache_misses=stats.misses,
        cache_builds=stats.builds, verified_cache_entries=verified)
end

"""Return a JLD2-safe immutable diagnostic snapshot for one adapter."""
iid_contest_diagnostics(policy::PTDSCEQIIDContestPolicy) =
    _cache_diagnostics(:ptd_sceq, policy.cache)

function iid_contest_diagnostics(policy::PTDOSAIIDContestPolicy)
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

"""
Convert an EGM floating diagnostic to its checkpoint representation.

Finite values, including signed zero, are returned unchanged. EGM initializes
some inapplicable minimum-margin fields to positive infinity; only callers that
identify such a field may encode that value as `nothing`. NaN, negative
infinity, and nonfinite maxima or masses indicate a diagnostics bug and are
rejected.
"""
function _checkpoint_diagnostic(value::Float64, field::Symbol;
    positive_infinity_is_unavailable::Bool=false)
    isnan(value) && throw(ErrorException(
        "DCEGM diagnostic $field is NaN and cannot be checkpointed."))
    isfinite(value) && return value
    positive_infinity_is_unavailable && value == Inf && return nothing
    throw(ErrorException(
        "DCEGM diagnostic $field is nonfinite and is not an unavailable sentinel."))
end

function _egm_solve_diagnostics(diagnostics::DcEgm.EGMSolveDiagnostics)
    return (quadrature_expectations=diagnostics.quadrature_expectations,
        quadgk_fallbacks=diagnostics.quadgk_fallbacks,
        common_grid_projections=diagnostics.common_grid_projections,
        continuation_queries=diagnostics.continuation_queries,
        continuation_wealth_extrapolations=
            diagnostics.continuation_wealth_extrapolations,
        continuation_income_clamps=diagnostics.continuation_income_clamps,
        max_continuation_wealth_extrapolation=
            _checkpoint_diagnostic(
                diagnostics.max_continuation_wealth_extrapolation,
                :max_continuation_wealth_extrapolation),
        max_continuation_log_income_clamp=
            _checkpoint_diagnostic(
                diagnostics.max_continuation_log_income_clamp,
                :max_continuation_log_income_clamp),
        quadrature_continuation_mass=
            _checkpoint_diagnostic(diagnostics.quadrature_continuation_mass,
                :quadrature_continuation_mass),
        quadrature_income_clamp_mass=_checkpoint_diagnostic(
            diagnostics.quadrature_income_clamp_mass,
            :quadrature_income_clamp_mass),
        min_continuation_lower_wealth_margin=
            _checkpoint_diagnostic(
                diagnostics.min_continuation_lower_wealth_margin,
                :min_continuation_lower_wealth_margin;
                positive_infinity_is_unavailable=true),
        min_continuation_upper_wealth_margin=
            _checkpoint_diagnostic(
                diagnostics.min_continuation_upper_wealth_margin,
                :min_continuation_upper_wealth_margin;
                positive_infinity_is_unavailable=true),
        min_continuation_lower_log_income_margin=
            _checkpoint_diagnostic(
                diagnostics.min_continuation_lower_log_income_margin,
                :min_continuation_lower_log_income_margin;
                positive_infinity_is_unavailable=true),
        min_continuation_upper_log_income_margin=
            _checkpoint_diagnostic(
                diagnostics.min_continuation_upper_log_income_margin,
                :min_continuation_upper_log_income_margin;
                positive_infinity_is_unavailable=true))
end

function _egm_query_diagnostics(diagnostics::DcEgm.EGMQueryDiagnostics)
    return lock(diagnostics.lock) do
        (consumption_queries=diagnostics.consumption_queries,
         value_queries=diagnostics.value_queries,
         material_wealth_extrapolations=
            diagnostics.material_wealth_extrapolations,
         lower_wealth_boundary_contacts=
            diagnostics.lower_wealth_boundary_contacts,
         upper_wealth_boundary_contacts=
            diagnostics.upper_wealth_boundary_contacts,
         material_income_clamps=diagnostics.material_income_clamps,
         lower_income_boundary_contacts=
            diagnostics.lower_income_boundary_contacts,
         upper_income_boundary_contacts=
            diagnostics.upper_income_boundary_contacts,
         max_wealth_extrapolation=_checkpoint_diagnostic(
            diagnostics.max_wealth_extrapolation,
            :max_wealth_extrapolation),
         max_income_clamp=_checkpoint_diagnostic(
            diagnostics.max_income_clamp, :max_income_clamp),
         min_lower_wealth_margin=_checkpoint_diagnostic(
            diagnostics.min_lower_wealth_margin, :min_lower_wealth_margin;
            positive_infinity_is_unavailable=true),
         min_upper_wealth_margin=_checkpoint_diagnostic(
            diagnostics.min_upper_wealth_margin, :min_upper_wealth_margin;
            positive_infinity_is_unavailable=true),
         min_lower_income_margin=_checkpoint_diagnostic(
            diagnostics.min_lower_income_margin, :min_lower_income_margin;
            positive_infinity_is_unavailable=true),
         min_upper_income_margin=_checkpoint_diagnostic(
            diagnostics.min_upper_income_margin, :min_upper_income_margin;
            positive_infinity_is_unavailable=true))
    end
end

function iid_contest_diagnostics(policy::DCEGMIIDContestPolicy)
    return (method=:dcegm, policy_fingerprint=policy.policy_fingerprint,
        asset_points=length(policy.asset_grid),
        asset_minimum=first(policy.asset_grid),
        asset_maximum=last(policy.asset_grid),
        solve=_egm_solve_diagnostics(policy.solution.solve_diagnostics),
        query=_egm_query_diagnostics(policy.solution.query_diagnostics))
end

iid_contest_diagnostics(policies::IIDContestPolicies) =
    (ptd_sceq=iid_contest_diagnostics(policies.ptd_sceq),
     ptd_osa=iid_contest_diagnostics(policies.ptd_osa),
     dcegm=iid_contest_diagnostics(policies.dcegm))

function _reset_ptd_osa_diagnostics!(aggregate::_PTDOSAContestDiagnostics)
    lock(aggregate.lock) do
        aggregate.calls = 0
        aggregate.certified_calls = 0
        aggregate.certification_failures = 0
        aggregate.fallback_uses = 0
        aggregate.total_structural_boundaries = 0
        aggregate.maximum_structural_boundaries = 0
        aggregate.total_interior_roots = 0
        aggregate.maximum_interior_roots = 0
        aggregate.total_floor_derivative_exclusions = 0
        aggregate.total_candidates = 0
        aggregate.maximum_candidates = 0
        aggregate.total_cells = 0
        aggregate.total_nodes = 0
        aggregate.maximum_certification_gap = 0.0
    end
    return nothing
end

"""
Reset evaluation-local counters without clearing caches or rebuilding a solver.

Call this only at an evaluation barrier, after all tasks using the adapter have
joined. DCEGM solve diagnostics intentionally remain lifetime setup metadata.
"""
function reset_iid_contest_diagnostics!(policy::PTDSCEQIIDContestPolicy)
    reset_plan_cache_statistics!(policy.cache)
    return iid_contest_diagnostics(policy)
end

function reset_iid_contest_diagnostics!(policy::PTDOSAIIDContestPolicy)
    reset_plan_cache_statistics!(policy.cache)
    _reset_ptd_osa_diagnostics!(policy.diagnostics)
    return iid_contest_diagnostics(policy)
end

function reset_iid_contest_diagnostics!(policy::DCEGMIIDContestPolicy)
    DcEgm.reset_query_diagnostics!(policy.solution)
    return iid_contest_diagnostics(policy)
end

function reset_iid_contest_diagnostics!(policies::IIDContestPolicies)
    return (
        ptd_sceq=reset_iid_contest_diagnostics!(policies.ptd_sceq),
        ptd_osa=reset_iid_contest_diagnostics!(policies.ptd_osa),
        dcegm=reset_iid_contest_diagnostics!(policies.dcegm))
end

"""Return the base-rule and transformed-rule audit for one adapter."""
function iid_contest_quadrature_audit(policy::PTDSCEQIIDContestPolicy)
    validate_iid_contest_policy_alignment(
        policy, :ptd_sceq, policy.config, policy.process)
    return (method=:ptd_sceq, uses_quadrature=false)
end

function iid_contest_quadrature_audit(policy::PTDOSAIIDContestPolicy)
    validate_iid_contest_policy_alignment(
        policy, :ptd_osa, policy.config, policy.process)
    income_rule = DcEgm.distribution_quadrature(
        LogNormal(policy.process.log_mean, policy.process.log_std),
        policy.base_rule)
    return (method=:ptd_osa, uses_quadrature=true,
        nodes_per_dimension=policy.base_rule.nodes_per_dimension,
        base_nodes=policy.base_rule.nodes,
        base_weights=policy.base_rule.weights,
        base_node_bits=_quadrature_bits(policy.base_rule.nodes),
        base_weight_bits=_quadrature_bits(policy.base_rule.weights),
        distribution_nodes=income_rule.nodes,
        distribution_weights=income_rule.weights,
        distribution_node_bits=_quadrature_bits(income_rule.nodes),
        distribution_weight_bits=_quadrature_bits(income_rule.weights))
end

function iid_contest_quadrature_audit(policy::DCEGMIIDContestPolicy)
    validate_iid_contest_policy_alignment(
        policy, :dcegm, policy.config, policy.process)
    return (method=:dcegm, uses_quadrature=true,
        nodes_per_dimension=policy.base_rule.nodes_per_dimension,
        base_nodes=policy.base_rule.nodes,
        base_weights=policy.base_rule.weights,
        base_node_bits=_quadrature_bits(policy.base_rule.nodes),
        base_weight_bits=_quadrature_bits(policy.base_rule.weights),
        distribution_nodes=policy.egm_rule.nodes,
        distribution_weights=policy.egm_rule.weights,
        distribution_node_bits=_quadrature_bits(policy.egm_rule.nodes),
        distribution_weight_bits=_quadrature_bits(policy.egm_rule.weights))
end

function iid_contest_quadrature_audit(oa::PTDOSAIIDContestPolicy,
    dcegm::DCEGMIIDContestPolicy)
    _config_token(oa.config) == _config_token(dcegm.config) ||
        throw(ArgumentError(
            "PTD-OSA and DCEGM quadrature audits require the same contest configuration."))
    _process_token(oa.process) == _process_token(dcegm.process) ||
        throw(ArgumentError(
            "PTD-OSA and DCEGM quadrature audits require the same IID panel."))
    oa_audit = iid_contest_quadrature_audit(oa)
    dcegm_audit = iid_contest_quadrature_audit(dcegm)
    return (nodes_bit_identical=
                oa_audit.base_node_bits == dcegm_audit.base_node_bits,
        weights_bit_identical=
                oa_audit.base_weight_bits == dcegm_audit.base_weight_bits ==
                oa_audit.distribution_weight_bits ==
                dcegm_audit.distribution_weight_bits,
        income_nodes_bit_identical=
                oa_audit.distribution_node_bits ==
                dcegm_audit.distribution_node_bits,
        ptd_osa=oa_audit, dcegm=dcegm_audit)
end

iid_contest_quadrature_audit(policies::IIDContestPolicies) =
    iid_contest_quadrature_audit(policies.ptd_osa, policies.dcegm)
