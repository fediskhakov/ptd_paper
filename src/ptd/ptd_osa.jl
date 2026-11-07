"""
One-step-ahead policy-tree descendant (PTD-OSA) policy.

PTD-OSA integrates only the next true innovation. At every quadrature node it
attaches a deterministic conditional-mean PTD continuation beginning at the next
date, then globally compares all structural boundaries and all interior Euler
roots. Evaluation paths and random-number generators never enter this file.
"""

if !isdefined(@__MODULE__, :PTDPlan)
    include("ptd_sceq.jl")
end

const PTD_OSA_ROOT_XRTOL = 1e-12

"""Numerical tolerances and the independent bounded-search effort for PTD-OSA."""
struct PTDOSAOptions
    tie_atol::Float64
    tie_rtol::Float64
    certification_atol::Float64
    direct_iterations::Int

    function PTDOSAOptions(; tie_atol::Real=1e-12, tie_rtol::Real=1e-10,
        certification_atol::Real=1e-8, direct_iterations::Integer=32)
        atol = Float64(tie_atol)
        rtol = Float64(tie_rtol)
        cert = Float64(certification_atol)
        iterations = Int(direct_iterations)
        all(isfinite, (atol, rtol, cert)) &&
            atol >= 0.0 && rtol >= 0.0 && cert >= 0.0 || throw(ArgumentError(
            "PTD-OSA tolerances must be finite and nonnegative."))
        iterations >= 8 || throw(ArgumentError(
            "The direct bounded check needs at least eight iterations."))
        return new(atol, rtol, cert, iterations)
    end
end

"""Auditable structural-search and independent bounded-check diagnostics."""
struct PTDOSADiagnostics
    structural_boundaries::Tuple
    interior_roots::Tuple
    node_count::Int
    cell_count::Int
    floor_derivative_exclusions::Int
    candidate_count::Int
    direct_saving::Float64
    direct_objective::Float64
    certification_gap::Float64
    certified::Bool
    fallback_used::Bool
    tail_start_date::Int
end

"""A PTD-OSA action together with the numerical checks used to select it."""
struct PTDOSAResult
    action::ActionResult
    diagnostics::PTDOSADiagnostics
end

"""Internal decision from comparing structural enumeration and bounded search."""
struct _PTDOSACertification
    saving::Float64
    value::Float64
    certification_gap::Float64
    certified::Bool
    fallback_used::Bool
end

"""One quadrature weight, realized next income, and deterministic continuation."""
struct _PTDOSANodeTail
    weight::Float64
    next_income::Float64
    plan::PTDPlan
end

_ptd_osa_utility(config::StochasticPTDConfig, consumption::Float64) =
    config.gamma == 1.0 ? log(consumption) :
    (consumption^(1.0 - config.gamma) - 1.0) / (1.0 - config.gamma)

_ptd_osa_marginal_utility(config::StochasticPTDConfig, consumption::Float64) =
    consumption <= 0.0 ? Inf : consumption^(-config.gamma)

function _validate_ptd_osa_rule(process::AbstractIncomeProcess,
    rule::GaussHermiteRule{D}) where {D}
    D == innovation_dimension(process) || throw(ArgumentError(
        "The quadrature dimension does not match the income process."))
    all(>(0.0), rule.weights) || throw(ArgumentError(
        "PTD-OSA requires strictly positive quadrature weights."))
    isapprox(sum(rule.weights), 1.0; rtol=8eps(Float64), atol=8eps(Float64)) ||
        throw(ArgumentError("PTD-OSA quadrature weights must sum to one."))
    return nothing
end

"""
Build one immutable conditional-mean PTD tail for every next-shock node. Each
plan is integrity-checked exactly once here. Subsequent private value queries do
not recompute its SHA fingerprint.
"""
function _build_ptd_osa_tails(config::StochasticPTDConfig,
    process::AbstractIncomeProcess, observed::ObservedState,
    rule::GaussHermiteRule;
    cache::Union{Nothing,PTDPlanCache}=nothing)
    _validate_observed(config, observed)
    observed.date < config.terminal.final_date || throw(ArgumentError(
        "A terminal decision has no PTD-OSA continuation tails."))
    _validate_ptd_osa_rule(process, rule)

    tails = _PTDOSANodeTail[]
    sizehint!(tails, length(rule.weights))
    for (node, weight) in zip(rule.nodes, rule.weights)
        # Saving is zero in this call, so z is exactly next income. Using the
        # shared transition prevents a second implementation of shock timing.
        transition = true_transition(
            config, process, observed, observed.cash, node)
        next_observed_state = ObservedState(
            observed.date + 1, transition.xnext, transition.snext)
        path = nominal_path(config, process, next_observed_state)
        if isnothing(cache)
            plan = build_ptd_plan(config, path)
            _assert_plan_integrity(plan)
        else
            plan = cached_ptd_plan!(cache, config, path)
        end
        push!(tails, _PTDOSANodeTail(weight, transition.z, plan))
    end
    return tails
end

"""Pure, unchecked value query used only after `_build_ptd_osa_tails` validation."""
function _fast_ptd_osa_plan_value(plan::PTDPlan, cash::Float64)
    agent = _plan_agent(plan)
    current_cash = cash
    result = 0.0
    for period in 1:agent.tMax
        consumption = c_x(agent, current_cash; i=1, t=period)
        result += agent.β^(period - 1) * u(agent, consumption)
        if period < agent.tMax
            assets = current_cash - consumption
            current_cash = xNext_a(agent, assets; i=1, t=period)
        end
    end
    return result
end

"""PTD-OSA objective parameterized by current saving `a` in `[0,x]`."""
function _ptd_osa_objective(config::StochasticPTDConfig,
    observed::ObservedState, tails::Vector{_PTDOSANodeTail}, saving::Float64)
    0.0 <= saving <= observed.cash || throw(ArgumentError(
        "PTD-OSA saving must lie in [0, current cash]."))
    consumption = observed.cash - saving
    current_utility = _ptd_osa_utility(config, consumption)
    continuation = 0.0
    gross_return = 1.0 + config.return_rate
    for tail in tails
        next_cash = max(gross_return * saving + tail.next_income,
            config.cash_floor)
        continuation += tail.weight * _fast_ptd_osa_plan_value(tail.plan, next_cash)
    end
    return current_utility + config.beta * continuation
end

function _mapped_boundary!(boundaries::Vector{Float64}, mapped::Float64,
    current_cash::Float64)
    isfinite(mapped) || return nothing
    if 0.0 < mapped < current_cash
        push!(boundaries, mapped)
    end
    return nothing
end

"""Map all floor switches and finite next-date PTD boundaries into saving."""
function _ptd_osa_structural_boundaries(config::StochasticPTDConfig,
    observed::ObservedState, tails::Vector{_PTDOSANodeTail})
    boundaries = Float64[0.0, observed.cash]
    gross_return = 1.0 + config.return_rate
    for tail in tails
        floor_saving = (config.cash_floor - tail.next_income) / gross_return
        _mapped_boundary!(boundaries, floor_saving, observed.cash)
        policy = _plan_agent(tail.plan).policy[1, 1]
        for segment in policy
            boundary = segment.xStart
            if isfinite(boundary) && boundary > config.cash_floor
                mapped = (boundary - tail.next_income) / gross_return
                _mapped_boundary!(boundaries, mapped, observed.cash)
            end
        end
    end
    sort!(boundaries)
    unique!(boundaries)
    return boundaries
end

"""Fixed floor/policy-segment signature on one open structural cell."""
function _ptd_osa_cell_branches(config::StochasticPTDConfig,
    tails::Vector{_PTDOSANodeTail}, saving::Float64)
    gross_return = 1.0 + config.return_rate
    branches = Vector{Union{Nothing,PolicySegment}}(undef, length(tails))
    for (index, tail) in pairs(tails)
        next_resources = gross_return * saving + tail.next_income
        if next_resources < config.cash_floor
            branches[index] = nothing
        else
            policy = _plan_agent(tail.plan).policy[1, 1]
            branches[index] = policy[findfirst(policy, next_resources)]
        end
    end
    return branches
end

"""
Current Euler gap on a fixed cell. Floor-binding nodes contribute exactly zero
to the continuation derivative; they are never evaluated at the floor policy's
marginal value.
"""
function _ptd_osa_euler_gap(config::StochasticPTDConfig,
    observed::ObservedState, tails::Vector{_PTDOSANodeTail},
    branches::Vector{Union{Nothing,PolicySegment}}, saving::Float64)
    consumption = observed.cash - saving
    consumption <= 0.0 && return Inf
    expected_marginal = 0.0
    gross_return = 1.0 + config.return_rate
    for (tail, segment) in zip(tails, branches)
        isnothing(segment) && continue
        next_resources = gross_return * saving + tail.next_income
        next_consumption = segment.b0 + segment.b1 * next_resources
        expected_marginal += tail.weight *
            _ptd_osa_marginal_utility(config, next_consumption)
    end
    return _ptd_osa_marginal_utility(config, consumption) -
        config.beta * gross_return * expected_marginal
end

function _opposite_sign(left::Float64, right::Float64)
    (isnan(left) || isnan(right) || left == 0.0 || right == 0.0) && return false
    return signbit(left) != signbit(right)
end

function _brent_ptd_osa_root(gap, lower::Float64, upper::Float64,
    gap_lower::Float64, gap_upper::Float64)
    _opposite_sign(gap_lower, gap_upper) || throw(ArgumentError(
        "A Brent bracket must have opposite Euler-gap signs."))
    checked_gap = saving -> begin
        value = gap(saving)
        isnan(value) && throw(ErrorException(
            "The PTD-OSA Euler gap became NaN inside a structural cell."))
        return value
    end
    return find_zero(checked_gap, (lower, upper), Roots.Brent();
        xrtol=PTD_OSA_ROOT_XRTOL)
end

function _monotone_gap(left::Float64, middle::Float64, right::Float64)
    any(isnan, (left, middle, right)) && return false
    scale = max(1.0,
        isfinite(left) ? abs(left) : 1.0,
        isfinite(middle) ? abs(middle) : 1.0,
        isfinite(right) ? abs(right) : 1.0)
    tolerance = 128eps(Float64) * scale
    return left <= middle + tolerance && middle <= right + tolerance
end

"""Fallback enumeration if a sampled cell violates the monotone-gap invariant."""
function _scan_ptd_osa_roots(gap, lower::Float64, upper::Float64; subdivisions::Int=128)
    points = range(lower, upper; length=subdivisions + 1)
    values = [gap(Float64(point)) for point in points]
    roots = Float64[]
    for index in 1:subdivisions
        left = Float64(points[index])
        right = Float64(points[index+1])
        left_value = values[index]
        right_value = values[index+1]
        if left_value == 0.0 && lower < left < upper
            push!(roots, left)
        end
        if _opposite_sign(left_value, right_value)
            root = _brent_ptd_osa_root(gap, left, right, left_value, right_value)
            lower < root < upper && push!(roots, root)
        end
    end
    sort!(roots)
    unique!(roots)
    return roots
end

function _enumerate_ptd_osa_roots(config::StochasticPTDConfig,
    observed::ObservedState, tails::Vector{_PTDOSANodeTail},
    boundaries::Vector{Float64})
    roots = Float64[]
    floor_exclusions = 0
    monotone_certified = true
    for cell in 1:length(boundaries)-1
        lower = boundaries[cell]
        upper = boundaries[cell+1]
        lower < upper || continue
        midpoint = lower + (upper - lower) / 2.0
        branches = _ptd_osa_cell_branches(config, tails, midpoint)
        floor_exclusions += count(isnothing, branches)
        gap = saving -> _ptd_osa_euler_gap(
            config, observed, tails, branches, saving)
        left_value = gap(lower)
        middle_value = gap(midpoint)
        right_value = gap(upper)
        monotone = _monotone_gap(left_value, middle_value, right_value)
        monotone_certified &= monotone
        if monotone
            if _opposite_sign(left_value, right_value)
                root = _brent_ptd_osa_root(
                    gap, lower, upper, left_value, right_value)
                lower < root < upper && push!(roots, root)
            end
        else
            append!(roots, _scan_ptd_osa_roots(gap, lower, upper))
        end
    end
    sort!(roots)
    unique!(roots)
    return roots, floor_exclusions, monotone_certified
end

"""
Select the largest saving whose value is within tolerance of the raw global
maximum. The two-pass comparison prevents a chain of pairwise ties from drifting
arbitrarily far below the actual maximum.
"""
function _best_ptd_osa_pairs(pairs, options::PTDOSAOptions)
    valid = [(Float64(saving), Float64(value)) for (saving, value) in pairs
        if !isnan(value)]
    isempty(valid) && throw(ErrorException(
        "Every PTD-OSA candidate objective was NaN."))
    raw_maximum = maximum(last, valid)
    if isfinite(raw_maximum)
        tolerance = max(options.tie_atol,
            options.tie_rtol * max(1.0, abs(raw_maximum)))
        eligible = filter(pair -> pair[2] >= raw_maximum - tolerance, valid)
    else
        eligible = filter(pair -> pair[2] == raw_maximum, valid)
    end
    selected = reduce((left, right) ->
        left[1] >= right[1] ? left : right, eligible)
    return selected
end

function _best_ptd_osa_candidate(objective, candidates, options::PTDOSAOptions)
    pairs = [(Float64(candidate), objective(Float64(candidate)))
        for candidate in candidates]
    return _best_ptd_osa_pairs(pairs, options)
end

"""Derivative-free bounded maximization on one structural cell."""
function _golden_ptd_osa_candidate(objective, lower::Float64, upper::Float64,
    options::PTDOSAOptions)
    lower == upper && return lower, objective(lower)
    original_lower = lower
    original_upper = upper
    ratio = (sqrt(5.0) - 1.0) / 2.0
    left_inner = upper - ratio * (upper - lower)
    right_inner = lower + ratio * (upper - lower)
    left_value = objective(left_inner)
    right_value = objective(right_inner)
    for _ in 1:options.direct_iterations
        if !isnan(right_value) && (isnan(left_value) || right_value >= left_value)
            lower = left_inner
            left_inner = right_inner
            left_value = right_value
            right_inner = lower + ratio * (upper - lower)
            right_value = objective(right_inner)
        else
            upper = right_inner
            right_inner = left_inner
            right_value = left_value
            left_inner = upper - ratio * (upper - lower)
            left_value = objective(left_inner)
        end
    end
    midpoint = lower + (upper - lower) / 2.0
    return _best_ptd_osa_candidate(objective,
        (original_lower, original_upper, left_inner, right_inner, midpoint),
        options)
end

"""Independent direct bounded search over every structural cell."""
function _direct_ptd_osa_global(objective, boundaries::Vector{Float64},
    options::PTDOSAOptions)
    if length(boundaries) == 1
        saving = boundaries[1]
        return saving, objective(saving)
    end
    candidates = Tuple{Float64,Float64}[]
    for cell in 1:length(boundaries)-1
        lower = boundaries[cell]
        upper = boundaries[cell+1]
        lower < upper || continue
        push!(candidates,
            _golden_ptd_osa_candidate(objective, lower, upper, options))
    end
    return _best_ptd_osa_pairs(candidates, options)
end

"""
Resolve the independent bounded-search certificate without ever discarding a
known better structural candidate.  On certification failure the enumerated
and direct results enter one final global comparison using the same save-more
tie rule as every other PTD-OSA candidate comparison.
"""
function _resolve_ptd_osa_certification(enumerated_saving::Float64,
    enumerated_value::Float64, direct_saving::Float64,
    direct_value::Float64, monotone_certified::Bool, options::PTDOSAOptions)

    certification_gap = direct_value == enumerated_value ? 0.0 :
        abs(direct_value - enumerated_value)
    certified = monotone_certified &&
        certification_gap <= options.certification_atol
    fallback_used = !certified
    if fallback_used
        chosen_saving, chosen_value = _best_ptd_osa_pairs(
            ((enumerated_saving, enumerated_value),
             (direct_saving, direct_value)), options)
    else
        chosen_saving, chosen_value = enumerated_saving, enumerated_value
    end
    return _PTDOSACertification(chosen_saving, chosen_value,
        certification_gap, certified, fallback_used)
end

function _ptd_osa_policy_fingerprint(config::StochasticPTDConfig,
    process::AbstractIncomeProcess, observed::ObservedState,
    rule::GaussHermiteRule, tails::Vector{_PTDOSANodeTail}, options::PTDOSAOptions)
    rule_token = join((join(_float_token.(node), ',') * ":" *
        _float_token(weight) for (node, weight) in zip(rule.nodes, rule.weights)), ';')
    tail_token = join((tail.plan.fingerprint for tail in tails), ';')
    return _sha_fingerprint("ptd_osa", _config_token(config),
        _process_token(process), observed.date, _float_token(observed.cash),
        _state_token(observed.exogenous), rule_token, tail_token,
        _float_token(options.tie_atol), _float_token(options.tie_rtol),
        _float_token(options.certification_atol), options.direct_iterations)
end

function _terminal_ptd_osa_result(config::StochasticPTDConfig,
    process::AbstractIncomeProcess, observed::ObservedState,
    rule::GaussHermiteRule)
    _validate_observed(config, observed)
    _validate_ptd_osa_rule(process, rule)
    objective = _ptd_osa_utility(config, observed.cash)
    fingerprint = _sha_fingerprint("ptd_osa_terminal", _config_token(config),
        _process_token(process), observed.date, _float_token(observed.cash),
        _state_token(observed.exogenous))
    action = ActionResult(observed.cash, 0.0, objective,
        :ptd_osa, fingerprint)
    diagnostics = PTDOSADiagnostics((0.0,), (), length(rule.weights), 0, 0, 1,
        0.0, objective, 0.0, true, false, observed.date)
    return PTDOSAResult(action, diagnostics)
end

"""
    ptd_osa_with_diagnostics(config, process, observed, rule; options=PTDOSAOptions())

Compute the nonanticipative one-step-ahead PTD action and return its structural
search diagnostics. A derivative-free bounded maximization independently checks
every structural cell. If it finds a materially better value than the complete
boundary/root enumeration, that direct result is used and `fallback_used=true`.
"""
function ptd_osa_with_diagnostics(config::StochasticPTDConfig,
    process::AbstractIncomeProcess, observed::ObservedState,
    rule::GaussHermiteRule; options::PTDOSAOptions=PTDOSAOptions(),
    cache::Union{Nothing,PTDPlanCache}=nothing)
    _validate_observed(config, observed)
    _validate_ptd_osa_rule(process, rule)
    if observed.date == config.terminal.final_date
        return _terminal_ptd_osa_result(config, process, observed, rule)
    end

    tails = _build_ptd_osa_tails(
        config, process, observed, rule; cache=cache)
    boundaries = _ptd_osa_structural_boundaries(config, observed, tails)
    roots, floor_exclusions, monotone_certified =
        _enumerate_ptd_osa_roots(config, observed, tails, boundaries)
    candidates = sort!(unique!(vcat(copy(boundaries), roots)))
    objective = saving -> _ptd_osa_objective(
        config, observed, tails, Float64(saving))
    enumerated_saving, enumerated_value =
        _best_ptd_osa_candidate(objective, candidates, options)
    direct_saving, direct_value =
        _direct_ptd_osa_global(objective, boundaries, options)
    resolution = _resolve_ptd_osa_certification(
        enumerated_saving, enumerated_value, direct_saving, direct_value,
        monotone_certified, options)
    chosen_saving = resolution.saving
    chosen_value = resolution.value
    consumption = observed.cash - chosen_saving
    fingerprint = _ptd_osa_policy_fingerprint(
        config, process, observed, rule, tails, options)
    action = ActionResult(consumption, chosen_saving, chosen_value,
        :ptd_osa, fingerprint)
    diagnostics = PTDOSADiagnostics(Tuple(boundaries), Tuple(roots),
        length(tails), max(length(boundaries) - 1, 0), floor_exclusions,
        length(candidates), direct_saving, direct_value,
        resolution.certification_gap, resolution.certified,
        resolution.fallback_used, observed.date + 1)
    return PTDOSAResult(action, diagnostics)
end

"""
    ptd_osa(config, process, observed, rule; options=PTDOSAOptions())

Return the PTD-OSA action. The method accepts no RNG, seed, realized future path,
or evaluation-path object. Use `ptd_osa_with_diagnostics` for certification data.
"""
function ptd_osa(config::StochasticPTDConfig,
    process::AbstractIncomeProcess, observed::ObservedState,
    rule::GaussHermiteRule; options::PTDOSAOptions=PTDOSAOptions(),
    cache::Union{Nothing,PTDPlanCache}=nothing)::ActionResult
    return ptd_osa_with_diagnostics(
        config, process, observed, rule; options=options, cache=cache).action
end
