include("model.jl")
using QuadGK
import Distributions: pdf, cdf, quantile, params, Normal, LogNormal

const EGM_INTEGRATION_RTOL = 1e-8
const EGM_DIAGNOSTIC_ATOL = 1e-12
const EGM_DIAGNOSTIC_RTOL = 1e-12
# The asset endpoint is the solver's declared numerical domain. A fourfold
# buffer retains ordinary endogenous-grid overshoot while excluding remote FOC
# points generated solely by vanishing nonbinding tail probabilities.
const EGM_STATE_DOMAIN_MULTIPLIER = 4.0


function _mark_agent_mutated!(agent::Agent)
    epoch = agent.bf.solve_epoch[]
    epoch < typemax(UInt64) || throw(ErrorException(
        "The EGM agent solve epoch is exhausted."))
    agent.bf.solve_epoch[] = epoch + UInt64(1)
    return agent.bf.solve_epoch[]
end


"""
Finite-node expectation rule. Nodes are scalar draws in the distribution's own
units; weights are strictly positive and normalized on construction.
"""
struct EGMQuadratureRule{N}
    nodes::NTuple{N,Float64}
    weights::NTuple{N,Float64}

    function EGMQuadratureRule(nodes::NTuple{N,Float64},
        weights::NTuple{N,Float64}, ::Val{:validated}) where {N}
        return new{N}(nodes, weights)
    end
end

function EGMQuadratureRule(nodes, weights)
    scalar_nodes = Float64[_scalar_quadrature_node(node) for node in nodes]
    scalar_weights = Float64[weight for weight in weights]
    !isempty(scalar_nodes) && length(scalar_nodes) == length(scalar_weights) ||
        throw(ArgumentError("Quadrature nodes and weights must have the same positive length."))
    all(isfinite, scalar_nodes) || throw(ArgumentError(
        "Every quadrature node must be finite."))
    all(weight -> isfinite(weight) && weight > 0.0, scalar_weights) ||
        throw(ArgumentError("Every quadrature weight must be finite and positive."))

    # Scale first so a valid collection such as `[floatmax(), floatmax()]`
    # cannot overflow while being normalized. Reject relative weights that are
    # too small to remain strictly positive in Float64 after scaling.
    scalar_weights ./= maximum(scalar_weights)
    total = sum(scalar_weights)
    isfinite(total) && total > 0.0 || throw(ArgumentError(
        "Quadrature weights could not be normalized finitely."))
    scalar_weights ./= total
    all(weight -> isfinite(weight) && weight > 0.0, scalar_weights) ||
        throw(ArgumentError("Normalized quadrature weights must remain positive."))
    isapprox(sum(scalar_weights), 1.0; rtol=8eps(Float64), atol=8eps(Float64)) ||
        throw(ArgumentError("Normalized quadrature weights must sum to one."))
    return EGMQuadratureRule(Tuple(scalar_nodes), Tuple(scalar_weights),
        Val(:validated))
end


_scalar_quadrature_node(node::Real) = Float64(node)
function _scalar_quadrature_node(node::Tuple)
    length(node) == 1 || throw(ArgumentError(
        "DCEGM supports scalar quadrature only; each tuple node must have length one."))
    return Float64(only(node))
end


"""
Build a rule from any object exposing `nodes` and `weights` fields.

If the source weights are already positive, finite, and normalized, their
`Float64` bit patterns are retained exactly. This lets PTD and DCEGM consume
one shared reported quadrature rule without a second normalization changing
the numerical experiment. Non-normalized field-compatible rules continue to
use the overflow-safe normalization in the public two-argument constructor.
"""
function EGMQuadratureRule(rule; transform=identity)
    hasproperty(rule, :nodes) && hasproperty(rule, :weights) || throw(ArgumentError(
        "A field-compatible quadrature rule must expose nodes and weights."))
    nodes = Float64[_scalar_quadrature_node(
        transform(_scalar_quadrature_node(node)))
        for node in getproperty(rule, :nodes)]
    weights = Float64[weight for weight in getproperty(rule, :weights)]
    !isempty(nodes) && length(nodes) == length(weights) || throw(ArgumentError(
        "Quadrature nodes and weights must have the same positive length."))

    normalized = all(isfinite, nodes) &&
        all(weight -> isfinite(weight) && weight > 0.0, weights) &&
        isapprox(sum(weights), 1.0;
            rtol=8eps(Float64), atol=8eps(Float64))
    if normalized
        return EGMQuadratureRule(Tuple(nodes), Tuple(weights), Val(:validated))
    end
    return EGMQuadratureRule(nodes, weights)
end


_distribution_node(d::Normal, node::Float64) = begin
    location, scale = params(d)
    return Float64(location + scale * node)
end

_distribution_node(d::LogNormal, node::Float64) = begin
    location, scale = params(d)
    return Float64(exp(location + scale * node))
end

function _distribution_node(d::ContinuousUnivariateDistribution,
    node::Float64)

    probability = cdf(Normal(), node)
    0.0 < probability < 1.0 || throw(ArgumentError(
        "The generic quantile transform cannot represent this Gauss-Hermite " *
        "tail probability in Float64. Pass an explicitly transformed " *
        "EGMQuadratureRule instead."))
    transformed = Float64(quantile(d, probability))
    isfinite(transformed) || throw(ArgumentError(
        "The transformed quadrature node must be finite."))
    return transformed
end

"""
Map standard-normal Gauss-Hermite nodes into a continuous scalar law.

Normal and lognormal laws use their exact affine/log-affine transforms. This is
important for high-order rules: converting a 31-node normal tail through a
Float64 CDF first would round its probability to zero or one. Other laws use a
quantile transform when its tail probability is representable. The adapter
accepts `GaussHermiteRule` through its public `nodes` and `weights` fields,
without coupling the EGM module to PTD internals.
"""
function distribution_quadrature(d::ContinuousUnivariateDistribution, rule)
    transform = node -> _distribution_node(d, node)
    return EGMQuadratureRule(rule; transform=transform)
end


function _quadrature_expectation(rule::EGMQuadratureRule, f,
    diagnostics::Union{Nothing,EGMSolveDiagnostics}=nothing)

    !isnothing(diagnostics) && (diagnostics.quadrature_expectations += 1)
    total = 0.0
    for (node, weight) in zip(rule.nodes, rule.weights)
        total += weight * f(node, weight)
    end
    return total
end


function _continuation_domain(agent::Union{AgentIID,AgentAR1},
    child_t::Int64)

    child_t < agent.bf.tMax || return nothing
    interpolant = agent isa AgentIID ? agent.bf.policy[child_t] :
        agent.bf.policy[child_t, 1]
    states = collect(Interpolations.knots(interpolant))
    return (first(states), last(states))
end


function _diagnostic_tolerance(values::Float64...)
    return EGM_DIAGNOSTIC_ATOL + EGM_DIAGNOSTIC_RTOL *
        max(1.0, maximum(abs, values))
end


"""Record every nonterminal continuation-policy/value query made while solving."""
function _record_solve_continuation!(diagnostics::Union{Nothing,EGMSolveDiagnostics},
    agent::Union{AgentIID,AgentAR1}, wealth::Float64, child_t::Int64,
    domain::Union{Nothing,Tuple{Float64,Float64}};
    income::Union{Nothing,Float64}=nothing,
    quadrature_weight::Union{Nothing,Float64}=nothing)

    (isnothing(diagnostics) || isnothing(domain) || child_t >= agent.bf.tMax) &&
        return nothing
    lower, upper = domain
    lower_margin, upper_margin = wealth - lower, upper - wealth
    wealth_distance = max(-lower_margin, -upper_margin, 0.0)
    tolerance = _diagnostic_tolerance(wealth, lower, upper)

    diagnostics.continuation_queries += 1
    diagnostics.min_continuation_lower_wealth_margin = min(
        diagnostics.min_continuation_lower_wealth_margin, lower_margin)
    diagnostics.min_continuation_upper_wealth_margin = min(
        diagnostics.min_continuation_upper_wealth_margin, upper_margin)
    if wealth_distance > tolerance
        diagnostics.continuation_wealth_extrapolations += 1
        diagnostics.max_continuation_wealth_extrapolation = max(
            diagnostics.max_continuation_wealth_extrapolation, wealth_distance)
    end
    if !isnothing(quadrature_weight)
        diagnostics.quadrature_continuation_mass += quadrature_weight
    end

    if agent isa AgentAR1 && !isnothing(income)
        log_income = log(income)
        lower_log, upper_log = log(first(agent.y)), log(last(agent.y))
        lower_log_margin = log_income - lower_log
        upper_log_margin = upper_log - log_income
        clamp_distance = max(-lower_log_margin, -upper_log_margin, 0.0)
        income_tolerance = _diagnostic_tolerance(log_income, lower_log, upper_log)
        diagnostics.min_continuation_lower_log_income_margin = min(
            diagnostics.min_continuation_lower_log_income_margin,
            lower_log_margin)
        diagnostics.min_continuation_upper_log_income_margin = min(
            diagnostics.min_continuation_upper_log_income_margin,
            upper_log_margin)
        if clamp_distance > income_tolerance
            diagnostics.continuation_income_clamps += 1
            diagnostics.max_continuation_log_income_clamp = max(
                diagnostics.max_continuation_log_income_clamp, clamp_distance)
            if !isnothing(quadrature_weight)
                diagnostics.quadrature_income_clamp_mass += quadrature_weight
            end
        end
    end
    return nothing
end

"""Integrate `f(x)` against a continuous distribution, optionally at a kink."""
function _expectation(d, f; split::Union{Nothing,Float64}=nothing,
    rtol::Float64=EGM_INTEGRATION_RTOL,
    diagnostics::Union{Nothing,EGMSolveDiagnostics}=nothing)

    !isnothing(diagnostics) && (diagnostics.quadgk_fallbacks += 1)
    lower, upper = Float64(minimum(d)), Float64(maximum(d))
    points = Float64[lower]
    if !isnothing(split) && lower < split < upper
        push!(points, split)
    end
    push!(points, upper)
    return quadgk(x -> f(x) * pdf(d, x), points...; rtol=rtol)[1]
end

"""Return the numerical wealth domain represented by an EGM asset grid."""
function _state_domain(agent::Agent)
    scale = max(agent.bf.a[end], agent.bf.xFloor)
    cap = scale > floatmax(Float64) / EGM_STATE_DOMAIN_MULTIPLIER ?
          floatmax(Float64) : EGM_STATE_DOMAIN_MULTIPLIER * scale
    return scale, cap
end


"""
Choose a positive corner-grid endpoint whose utility is finite and whose scale
is meaningful relative to the supplied asset grid. Values arbitrarily close to
zero are not useful interpolation knots: for log utility they contaminate the
entire first line segment, and for CRRA coefficients at or above two they can
overflow to `-Inf`.
"""
function _corner_lower_bound(agent::Agent, scale::Float64)
    gridStep = min(agent.bf.a[2], scale)
    lower = max(eps(scale), gridStep / 1024.0)

    if agent.bf.γ > 1.0
        # Keep c^(1-γ) comfortably below floatmax. The bound matters only for
        # very large risk aversion; for conventional calibrations the grid-based
        # lower bound above is tighter.
        logSafe = (-log(floatmax(Float64)) + log(16.0)) / (agent.bf.γ - 1.0)
        lower = max(lower, exp(logSafe))
    end

    while !isfinite(u(agent, lower))
        lower = sqrt(lower * scale)
        lower < scale || throw(ArgumentError(
            "No finite-utility positive corner state exists on the EGM domain."))
    end
    return lower
end


"""Split filtered interior points into contiguous economic candidates."""
function _interior_candidates(x::Vector{Float64}, c::Vector{Float64},
    v::Vector{Float64}, valid::BitVector;
    regimes=nothing)::Vector{Grids1d}

    isnothing(regimes) || length(regimes) == length(valid) ||
        throw(DimensionMismatch(
            "An interior regime label is required for every Euler point."))

    candidates = Grids1d[]
    firstIndex = 1
    while firstIndex <= length(valid)
        while firstIndex <= length(valid) && !valid[firstIndex]
            firstIndex += 1
        end
        firstIndex > length(valid) && break

        lastIndex = firstIndex
        while lastIndex < length(valid) && valid[lastIndex+1] &&
              (isnothing(regimes) ||
               isequal(regimes[lastIndex+1], regimes[firstIndex]))
            lastIndex += 1
        end
        if lastIndex > firstIndex
            push!(candidates, Grids1d(x[firstIndex:lastIndex],
                c[firstIndex:lastIndex], v[firstIndex:lastIndex]))
        end
        firstIndex = lastIndex + 1
    end
    return candidates
end


"""Policy-switch states represented by adjacent floating-point knots."""
function _iid_policy_discontinuities(agent::AgentIID, child_t::Int64)

    child_t < agent.bf.tMax || return Float64[]
    policy = agent.bf.policy[child_t]
    states = collect(Interpolations.knots(policy))
    consumption = policy(states)
    boundaries = Float64[]
    for index in 1:length(states)-1
        lower, upper = states[index], states[index+1]
        scale = max(abs(lower), abs(upper), 1.0)
        adjacent = upper - lower <= 8eps(scale)
        jump = !isapprox(consumption[index], consumption[index+1];
            rtol=1e-10, atol=1e-12)
        adjacent && jump && push!(boundaries, upper)
    end
    return sort!(unique!(boundaries))
end


"""
The finite-node continuation regime at one end-of-period asset choice.

Regime zero is floor binding. Positive regimes identify the open intervals
separated by genuine discontinuities of the child policy. Such a discontinuity
makes expected marginal value jump when a quadrature atom crosses it, just as a
primitive floor activation does.
"""
function _iid_continuation_signature(agent::AgentIID, a::Float64,
    quadrature::EGMQuadratureRule{N},
    policy_discontinuities::Vector{Float64}) where {N}

    R = 1.0 + agent.bf.r
    return ntuple(N) do index
        nextCash = R * a + quadrature.nodes[index]
        nextCash > agent.bf.xFloor || return 0
        return searchsortedlast(policy_discontinuities, nextCash) + 1
    end
end


"""
Return the distinct positive saving levels where a finite IID quadrature atom
hits the next-period cash floor. These primitive structural kinks receive exact
fixed-saving candidates even when the exogenous asset grid misses them.
"""
function _iid_structural_kink_assets(agent::AgentIID,
    quadrature::EGMQuadratureRule)

    R = 1.0 + agent.bf.r
    kinks = Float64[
        (agent.bf.xFloor - income) / R for income in quadrature.nodes
        if 0.0 < (agent.bf.xFloor - income) / R <= agent.bf.a[end]
    ]
    return sort!(unique!(kinks))
end


"""
Construct a singularity-safe grid for the borrowing-constrained branch.

The corner candidate uses the asset grid's resolution so its piecewise-linear
value approximation is balanced with the Euler candidate.  Only the interval
below the first positive asset knot receives geometric refinement.
"""
function _corner_grid(agent::Agent, lower::Float64,
    upper::Float64)::Vector{Float64}

    firstAsset = clamp(agent.bf.a[2], lower, upper)
    logarithmic = exp.(range(log(lower), log(firstAsset); length=16))
    # `exp(log(bound))` can round one ULP outside `bound`.  An overshoot at the
    # upper endpoint leaves a microscopic corner-only envelope interval, whose
    # slope would then control linear extrapolation and spuriously imply c=x at
    # every state beyond the endogenous grid.
    logarithmic[1], logarithmic[end] = lower, firstAsset

    assets = agent.bf.a[(firstAsset .<= agent.bf.a) .&
                        (agent.bf.a .<= upper)]
    assetSteps = diff(agent.bf.a)
    firstStep = assetSteps[1]
    extensionStep = all(step -> isapprox(step, firstStep; rtol=1e-12),
                        assetSteps) ? firstStep : maximum(assetSteps)
    extension = collect((agent.bf.a[end] + extensionStep):extensionStep:upper)
    return sort!(unique!(vcat(Float64[lower, upper], logarithmic, assets,
        extension)))
end


"""
Value of one fixed-saving Bellman candidate at current state `x`.
"""
function _fixed_saving_value(agent::Agent, x::Float64,
    saving::Float64, continuation::Float64)

    consumption = x - saving
    consumption > 0.0 || return -Inf
    return u(agent, consumption) + agent.bf.β * continuation
end


"""Whether the higher-saving `rightChoice` wins under the envelope tie rule."""
function _fixed_saving_right_wins(agent::Agent, state::Float64,
    leftChoice::Tuple{Float64,Float64},
    rightChoice::Tuple{Float64,Float64})

    leftValue = _fixed_saving_value(agent, state, leftChoice...)
    rightValue = _fixed_saving_value(agent, state, rightChoice...)
    isnan(leftValue) && throw(ErrorException(
        "A fixed-saving Bellman candidate evaluated to NaN."))
    isnan(rightValue) && throw(ErrorException(
        "A fixed-saving Bellman candidate evaluated to NaN."))
    return rightValue > leftValue || rightValue == leftValue ||
        isapprox(rightValue, leftValue;
            rtol=ENVELOPE_VALUE_RTOL, atol=ENVELOPE_VALUE_ATOL)
end


"""
Locate the first representable state where the higher-saving right choice wins.

The Boolean bisection is robust when CRRA utility overflows to `-Inf` near zero
consumption. It also locates the left edge of the configured numerical-tie band,
where the common save-more rule begins to select `rightChoice`.
"""
function _fixed_saving_crossing(agent::Agent,
    leftChoice::Tuple{Float64,Float64},
    rightChoice::Tuple{Float64,Float64}, lower::Float64, upper::Float64)

    lower <= upper || throw(ArgumentError(
        "A fixed-saving crossing requires an ordered state bracket."))
    _fixed_saving_right_wins(agent, lower, leftChoice, rightChoice) &&
        return lower
    _fixed_saving_right_wins(agent, upper, leftChoice, rightChoice) ||
        return nothing
    left, right = lower, upper
    for _ in 1:160
        nextfloat(left) >= right && return right
        midpoint = left + (right - left) / 2.0
        (midpoint == left || midpoint == right) && return right
        if _fixed_saving_right_wins(
            agent, midpoint, leftChoice, rightChoice)
            right = midpoint
        else
            left = midpoint
        end
    end
    return right
end


"""
Upper envelope of all corner and structural fixed-saving candidates.

Every pair of CRRA fixed-saving curves crosses at most once because their value
difference has a constant derivative sign. Sorting choices costs
`O(N log N)` and the subsequent monotone upper-hull stack is `O(N)`, apart from
fixed-cost scalar bisections. The retained switching roots are added to one
common state grid, avoiding the quadratic-in-knots cost of sending many shifted
fixed-choice grids through the generic polyline envelope. The final comparison
with Euler branches still uses the generic envelope and is not claimed linear.
"""
function _fixed_saving_envelope(agent::Agent,
    choices::Vector{Tuple{Float64,Float64}}, minState::Float64,
    maxState::Float64)::Grids1d

    !isempty(choices) || throw(ArgumentError(
        "At least the zero-saving corner candidate is required."))
    all(choice -> isfinite(choice[1]) && choice[1] >= 0.0 &&
                  isfinite(choice[2]), choices) || throw(ArgumentError(
        "Fixed-saving candidates must contain finite nonnegative saving and finite continuation values."))
    all(isfinite, (minState, maxState)) && 0.0 <= minState < maxState ||
        throw(ArgumentError(
            "The fixed-saving state domain must be finite, nonnegative, and ordered."))
    ordered = sort(copy(choices); by=first)
    deduplicated = Tuple{Float64,Float64}[]
    for choice in ordered
        if !isempty(deduplicated) && choice[1] == deduplicated[end][1]
            choice[2] > deduplicated[end][2] &&
                (deduplicated[end] = choice)
        else
            push!(deduplicated, choice)
        end
    end
    first(deduplicated)[1] == 0.0 || throw(ArgumentError(
        "Fixed-saving choices must include the zero-saving corner."))

    # CRRA gives ordered derivatives: if a₂>a₁, then
    # d[Q(x,a₂)-Q(x,a₁)]/dx = u'(x-a₂)-u'(x-a₁)>0. The upper
    # envelope can therefore be built with the same monotone stack used for an
    # upper hull of affine functions, without examining every pair at every
    # state knot.
    hull = Tuple{Float64,Float64}[]
    starts = Float64[]
    for choice in deduplicated
        # Saving a is feasible at every state x>a. Delaying feasibility by the
        # corner-grid width changes the Bellman choice for gamma<1, whose utility
        # has a finite limit at zero consumption.
        feasibleStart = max(minState, nextfloat(choice[1]))
        feasibleStart <= maxState || continue
        if isempty(hull)
            push!(hull, choice)
            push!(starts, minState)
            continue
        end

        retained = true
        crossing = feasibleStart
        while !isempty(hull)
            incumbent = hull[end]
            lower = max(feasibleStart, starts[end],
                nextfloat(incumbent[1]))
            lower <= maxState || (retained = false; break)
            candidateCrossing = _fixed_saving_crossing(
                agent, incumbent, choice, lower, maxState)
            if isnothing(candidateCrossing)
                retained = false
                break
            end
            crossing = candidateCrossing
            if length(hull) > 1 && crossing <= starts[end]
                pop!(hull)
                pop!(starts)
                continue
            end
            break
        end
        retained || continue
        push!(hull, choice)
        push!(starts, crossing)
    end

    cuts = _corner_grid(agent, minState, maxState)
    append!(cuts, starts)
    sort!(unique!(cuts))
    x, c, v = Float64[], Float64[], Float64[]
    function pushWinner!(state::Float64)
        !isempty(x) && state <= x[end] && return nothing
        winner = clamp(searchsortedlast(starts, state), 1, length(hull))
        saving, continuation = hull[winner]
        push!(x, state)
        push!(c, state - saving)
        push!(v, _fixed_saving_value(agent, state, saving, continuation))
        return nothing
    end
    for index in eachindex(cuts)
        state = cuts[index]
        winner = clamp(searchsortedlast(starts, state), 1, length(hull))
        if index > 1
            left = cuts[index-1]
            midpoint = left + (state - left) / 2.0
            leftWinner = clamp(searchsortedlast(starts, midpoint), 1, length(hull))
            if leftWinner != winner
                leftLimit = prevfloat(state)
                left < leftLimit && pushWinner!(leftLimit)
            end
        end
        pushWinner!(state)
    end
    return Grids1d(x, c, v)
end


"""
Build the DCEGM upper envelope while keeping corner and interior curves
separate. Interior Euler points outside the numerical state domain are omitted
in contiguous runs, preventing tiny nonbinding tail probabilities from making
the corner grid span astronomically remote wealth states.
"""
function _with_corner(agent::Agent, xInt::Vector{Float64}, cInt::Vector{Float64},
    vInt::Vector{Float64}, cornerContinuation::Float64;
    interior_regimes=nothing,
    fixed_saving_continuations::Vector{Tuple{Float64,Float64}}=
        Tuple{Float64,Float64}[])::Grids1d

    isfinite(cornerContinuation) || throw(ArgumentError(
        "The borrowing-constrained continuation value must be finite."))
    domain, stateCap = _state_domain(agent)
    minState = _corner_lower_bound(agent, domain)
    valid = BitVector(isfinite.(xInt) .& isfinite.(cInt) .& isfinite.(vInt) .&
                      (minState .<= xInt) .& (xInt .<= stateCap) .&
                      (cInt .> 0.0) .& (cInt .<= xInt))
    candidates = _interior_candidates(xInt, cInt, vInt, valid;
        regimes=interior_regimes)
    validStates = xInt[valid]

    if isempty(fixed_saving_continuations)
        # Deterministic and AR(1) callers retain their original corner-only
        # construction and endogenous upper domain.
        maxState = isempty(validStates) ? domain :
            max(domain, maximum(validStates))
        xCorner = _corner_grid(agent, minState, maxState)
        vCorner = u(agent, xCorner) .+ agent.bf.β * cornerContinuation
        all(isfinite, vCorner) || throw(ArgumentError(
            "The borrowing-constrained value grid contains a nonfinite value."))
        pushfirst!(candidates, Grids1d(xCorner, copy(xCorner), vCorner))
    else
        # The IID fixed-saving Bellman candidates span the declared numerical
        # state cap. This prevents a remote Euler branch from being the accidental
        # source of continuation-domain coverage and makes the boundary
        # independent of whether a coarse grid samples a narrow activation regime.
        fixedChoices = vcat([(0.0, cornerContinuation)],
            fixed_saving_continuations)
        fixedEnvelope = _fixed_saving_envelope(
            agent, fixedChoices, minState, stateCap)
        all(isfinite, fixedEnvelope.v) || throw(ArgumentError(
            "The fixed-saving value envelope contains a nonfinite value."))
        pushfirst!(candidates, fixedEnvelope)
    end
    return upper_envelope(candidates)
end

# EGM step for agent with deterministic incomes.

"""
    egm_step!(agent::AgentD; t)
    
Using EGM algorithm to update the deterministic `agent`'s 
policy and value functions at period `t`.
"""
function egm_step!(agent::AgentD; t::Int64)
    _mark_agent_mutated!(agent)
    # Next period wealth level without social insurance.
    zNext = zNext_a(agent, agent.bf.a, t=t)
    # Euler points exist only where the next-period floor is not binding.
    # Reaching the floor exactly with positive assets is dominated by consuming
    # those assets today; Euler candidates therefore require a strictly
    # nonbinding next-period floor.
    eligible = zNext .> agent.bf.xFloor
    assets = agent.bf.a[eligible]
    zNext = zNext[eligible]
    cNext = c_x(agent, zNext, t=t + 1) # Next period consumption.
    # Current consumption based on Euler equation.
    cInt = inv_mu(agent, agent.bf.β * (1 + agent.bf.r) * mu(agent, cNext))

    xInt = assets .+ cInt # Endogenous current wealth level.
    # Current values without social insurance.
    vInt = u(agent, cInt) .+ agent.bf.β .* v_x(agent, zNext, t=t + 1)

    xNextCorner = xNext_a(agent, 0.0, t=t)
    grids = _with_corner(agent, xInt, cInt, vInt, v_x(agent, xNextCorner, t=t + 1))

    agent.bf.policy[t] = linear_interpolation(grids.x, grids.c,
        extrapolation_bc=Line())
    agent.bf.value[t] = linear_interpolation(grids.x, grids.v,
        extrapolation_bc=Line())
    return grids
end


# EGM step for IID type agent.

"""
    rhs(agent::AgentIID, a::Float64; t::Int64) 

The r.h.s. of the stochastic euler equation for IID `agent` at time `t`
with end-of-period asset level `a`.
"""
function rhs(agent::AgentIID, a::Float64; t::Int64,
    rtol::Float64=EGM_INTEGRATION_RTOL,
    quadrature::Union{Nothing,EGMQuadratureRule}=nothing,
    diagnostics::Union{Nothing,EGMSolveDiagnostics}=nothing,
    continuation_domain::Union{Nothing,Tuple{Float64,Float64}}=nothing)

    R = 1.0 + agent.bf.r
    if !isnothing(quadrature)
        expectation = _quadrature_expectation(quadrature, (yNext, weight) -> begin
            zNext = R * a + yNext
            zNext > agent.bf.xFloor || return 0.0
            _record_solve_continuation!(diagnostics, agent, zNext, t + 1,
                continuation_domain; quadrature_weight=weight)
            return mu(agent, c_x(agent, zNext, t=t + 1))
        end, diagnostics)
        return agent.bf.β * R * max(expectation, 0.0)
    end

    threshold = agent.bf.xFloor - R * a
    lower, upper = Float64(minimum(agent.d)), Float64(maximum(agent.d))
    integrationLower = max(lower, threshold)
    integrationLower >= upper && return 0.0

    !isnothing(diagnostics) && (diagnostics.quadgk_fallbacks += 1)
    expectation = quadgk(yNext -> begin
            zNext = R * a + yNext
            _record_solve_continuation!(diagnostics, agent, zNext, t + 1,
                continuation_domain)
            mu(agent, c_x(agent, zNext, t=t + 1)) * pdf(agent.d, yNext)
        end,
        integrationLower, upper; rtol=rtol)[1]
    return agent.bf.β * R * max(expectation, 0.0)
end


"""
    ev_next(agent::AgentIID, a::Float64; t::Int64)

The expected next period value for IID `agent` at time `t` 
with end-of-period asset level `a`.
"""
function ev_next(agent::AgentIID, a::Float64; t::Int64,
    rtol::Float64=EGM_INTEGRATION_RTOL,
    quadrature::Union{Nothing,EGMQuadratureRule}=nothing,
    diagnostics::Union{Nothing,EGMSolveDiagnostics}=nothing,
    continuation_domain::Union{Nothing,Tuple{Float64,Float64}}=nothing)

    R = 1.0 + agent.bf.r
    if !isnothing(quadrature)
        return _quadrature_expectation(quadrature, (yNext, weight) -> begin
            xNext = max(R * a + yNext, agent.bf.xFloor)
            _record_solve_continuation!(diagnostics, agent, xNext, t + 1,
                continuation_domain; quadrature_weight=weight)
            return v_x(agent, xNext, t=t + 1)
        end, diagnostics)
    end
    threshold = agent.bf.xFloor - R * a
    return _expectation(agent.d,
        yNext -> begin
            xNext = max(R * a + yNext, agent.bf.xFloor)
            _record_solve_continuation!(diagnostics, agent, xNext, t + 1,
                continuation_domain)
            return v_x(agent, xNext, t=t + 1)
        end;
        split=threshold, rtol=rtol, diagnostics=diagnostics)
end


"""
    egm_step!(agent::AgentIID; t::Int64)

For the `agent` with i.i.d. income distribution, 
use EGM algorithm to update the `agent`'s policy and value functions at period `t`.
"""
function egm_step!(agent::AgentIID; t::Int64,
    quadrature::Union{Nothing,EGMQuadratureRule}=nothing,
    diagnostics::Union{Nothing,EGMSolveDiagnostics}=nothing)
    _mark_agent_mutated!(agent)
    continuation_domain = _continuation_domain(agent, t + 1)
    # Current comsumptions based on the stochastic Euler equation.
    cInt = Vector{Float64}(undef, length(agent.bf.a))
    # Endogenous current wealth levels.
    xInt = Vector{Float64}(undef, length(agent.bf.a))
    # Current values without social insurance.
    vInt = Vector{Float64}(undef, length(agent.bf.a))
    # Expected continuation values are also the fixed-saving Bellman intercepts.
    evInt = Vector{Float64}(undef, length(agent.bf.a))

    for (i, a) in enumerate(agent.bf.a)
        cInt[i] = inv_mu(agent, rhs(agent, a, t=t, quadrature=quadrature,
            diagnostics=diagnostics, continuation_domain=continuation_domain))
        xInt[i] = a + cInt[i]
        evInt[i] = ev_next(agent, a, t=t,
            quadrature=quadrature, diagnostics=diagnostics,
            continuation_domain=continuation_domain)
        vInt[i] = u(agent, cInt[i]) + agent.bf.β * evInt[i]
    end

    # Fedor Iskhakov's secondary envelope, and the independent DCEGM.jl port,
    # split loopbacks of a continuous Euler polyline before comparing branches.
    # Finite quadrature plus a floor needs one additional step: a node crossing
    # either the floor or an inherited policy jump changes the marginal-value
    # regime discontinuously, so those adjacent Euler points must never share a
    # polyline. The global fixed-saving hull below is the corresponding bounded
    # Bellman correction: every declared asset choice participates, and exact
    # primitive floor kinks are inserted even when the asset grid misses them.
    # Inherited child-policy jumps split Euler loci, but their optimal kink
    # savings remain a bounded asset-grid approximation. They are therefore
    # declared for the paper's reported 6,401-point comparator
    # rather than described as exact.
    policy_discontinuities = isnothing(quadrature) ? Float64[] :
        _iid_policy_discontinuities(agent, t + 1)
    interior_regimes = isnothing(quadrature) ? nothing :
        [_iid_continuation_signature(
            agent, a, quadrature, policy_discontinuities)
         for a in agent.bf.a]
    floor_kink_continuations = isnothing(quadrature) ?
        Tuple{Float64,Float64}[] :
        [(saving, ev_next(agent, saving, t=t, quadrature=quadrature,
            diagnostics=diagnostics,
            continuation_domain=continuation_domain))
         for saving in _iid_structural_kink_assets(agent, quadrature)]
    fixed_saving_continuations = collect(zip(agent.bf.a, evInt))
    append!(fixed_saving_continuations, floor_kink_continuations)

    grids = _with_corner(agent, xInt, cInt, vInt, evInt[1];
        interior_regimes, fixed_saving_continuations)

    agent.bf.policy[t] = linear_interpolation(grids.x, grids.c,
        extrapolation_bc=Line())
    agent.bf.value[t] = linear_interpolation(grids.x, grids.v,
        extrapolation_bc=Line())

    return grids
end


# EGM step for AR1 type agent.

"""
    rhs(agent::AgentAR1, a::Float64, yNow::Float64; t::Int64)::Float64

The r.h.s. of the stochastic euler equation for AR1 `agent` at time `t`
with end-of-period asset level `a` and current income `yNow`.
"""
function rhs(agent::AgentAR1, a::Float64, yNow::Float64; t::Int64,
    rtol::Float64=EGM_INTEGRATION_RTOL,
    quadrature::Union{Nothing,EGMQuadratureRule}=nothing,
    diagnostics::Union{Nothing,EGMSolveDiagnostics}=nothing,
    continuation_domain::Union{Nothing,Tuple{Float64,Float64}}=nothing)::Float64

    yNow > 0.0 || throw(ArgumentError("Current AR(1) income must be positive."))
    R = 1.0 + agent.bf.r
    if !isnothing(quadrature)
        expectation = _quadrature_expectation(quadrature, (ε, weight) -> begin
            yNext = exp(agent.ρ * log(yNow) + ε)
            zNext = R * a + yNext
            zNext > agent.bf.xFloor || return 0.0
            _record_solve_continuation!(diagnostics, agent, zNext, t + 1,
                continuation_domain; income=yNext, quadrature_weight=weight)
            return mu(agent, c_x(agent, zNext, yNext, t=t + 1))
        end, diagnostics)
        return agent.bf.β * R * max(expectation, 0.0)
    end

    requiredIncome = agent.bf.xFloor - R * a
    lower, upper = Float64(minimum(agent.d)), Float64(maximum(agent.d))
    threshold = requiredIncome > 0.0 ?
                log(requiredIncome) - agent.ρ * log(yNow) : lower
    integrationLower = max(lower, threshold)
    integrationLower >= upper && return 0.0

    !isnothing(diagnostics) && (diagnostics.quadgk_fallbacks += 1)
    expectation = quadgk(ε -> begin
            yNext = exp(agent.ρ * log(yNow) + ε)
            zNext = R * a + yNext
            _record_solve_continuation!(diagnostics, agent, zNext, t + 1,
                continuation_domain; income=yNext)
            mu(agent, c_x(agent, zNext, yNext, t=t + 1)) * pdf(agent.d, ε)
        end, integrationLower, upper; rtol=rtol)[1]
    return agent.bf.β * R * max(expectation, 0.0)
end


"""
    ev_next(agent::AgentAR1, a::Float64, yNow::Float64; t::Int64)::Float64

The expected next period value for AR1 `agent` at time `t`
with end-of-period asset level `a` and current income `yNow`.
"""
function ev_next(agent::AgentAR1, a::Float64, yNow::Float64; t::Int64,
    rtol::Float64=EGM_INTEGRATION_RTOL,
    quadrature::Union{Nothing,EGMQuadratureRule}=nothing,
    diagnostics::Union{Nothing,EGMSolveDiagnostics}=nothing,
    continuation_domain::Union{Nothing,Tuple{Float64,Float64}}=nothing)::Float64

    yNow > 0.0 || throw(ArgumentError("Current AR(1) income must be positive."))
    R = 1.0 + agent.bf.r
    if !isnothing(quadrature)
        return _quadrature_expectation(quadrature, (ε, weight) -> begin
            yNext = exp(agent.ρ * log(yNow) + ε)
            xNext = max(R * a + yNext, agent.bf.xFloor)
            _record_solve_continuation!(diagnostics, agent, xNext, t + 1,
                continuation_domain; income=yNext, quadrature_weight=weight)
            return v_x(agent, xNext, yNext; t=t + 1)
        end, diagnostics)
    end

    requiredIncome = agent.bf.xFloor - R * a
    threshold = requiredIncome > 0.0 ?
                log(requiredIncome) - agent.ρ * log(yNow) : nothing
    return _expectation(agent.d, ε -> begin
            yNext = exp(agent.ρ * log(yNow) + ε)
            xNext = max(R * a + yNext, agent.bf.xFloor)
            _record_solve_continuation!(diagnostics, agent, xNext, t + 1,
                continuation_domain; income=yNext)
            v_x(agent, xNext, yNext; t=t + 1)
        end; split=threshold, rtol=rtol, diagnostics=diagnostics)
end


"""
    egm_step_y(agent::AgentAR1, y::Float64; t::Int64)::Grids1d

For the AR1 type `agent` at period `t` with fixed current income level `y`,
use EGM algorithm to update the `agent`'s policy and value functions.
# Returns
- `grids`: the x-c-v grids at period `t` and fixed `y` on the upper envelope.
"""
function egm_step_y(agent::AgentAR1, y::Float64; t::Int64,
    quadrature::Union{Nothing,EGMQuadratureRule}=nothing,
    diagnostics::Union{Nothing,EGMSolveDiagnostics}=nothing,
    continuation_domain::Union{Nothing,Tuple{Float64,Float64}}=nothing)::Grids1d
    n_a = length(agent.bf.a)
    # Current comsumptions based on the stochastic Euler equation.
    cInt = Vector{Float64}(undef, n_a)
    # Endogenous current wealth levels.
    xInt = Vector{Float64}(undef, n_a)
    # Current values without social insurance.
    vInt = Vector{Float64}(undef, n_a)

    for (i, a) in enumerate(agent.bf.a)
        cInt[i] = inv_mu(agent, rhs(agent, a, y, t=t,
            quadrature=quadrature, diagnostics=diagnostics,
            continuation_domain=continuation_domain))
        xInt[i] = a + cInt[i]
        vInt[i] = u(agent, cInt[i]) + agent.bf.β * ev_next(agent, a, y, t=t,
            quadrature=quadrature, diagnostics=diagnostics,
            continuation_domain=continuation_domain)
    end

    grids = _with_corner(agent, xInt, cInt, vInt, ev_next(agent, 0.0, y, t=t,
        quadrature=quadrature, diagnostics=diagnostics,
        continuation_domain=continuation_domain))

    return grids
end


"""
Project persistent-income slices onto one common wealth grid.

This is an original one-dimensional Julia adaptation inspired by the common-grid
reconstruction in Druedahl and Jørgensen's G2EGM
(https://github.com/JeppeDruedahl/G2EGM): each conditional endogenous grid receives
its own upper envelope, and the resulting slices are then represented on a
shared state grid before interpolation over the persistent state. Restricting
the shared grid to the intersection of slice domains avoids hidden extrapolation
during the projection itself.
"""
function _common_state_projection(slices::Vector{Grids1d}, y::Vector{Float64},
    diagnostics::Union{Nothing,EGMSolveDiagnostics}=nothing)::Grids2d

    length(slices) == length(y) || throw(ArgumentError(
        "There must be one endogenous-grid slice per persistent-state node."))
    lower = maximum(first(slice.x) for slice in slices)
    upper = minimum(last(slice.x) for slice in slices)
    lower < upper || throw(ArgumentError(
        "Persistent-state EGM slices have no common nondegenerate wealth domain."))

    common = Float64[lower, upper]
    for slice in slices
        append!(common, (state for state in slice.x if lower < state < upper))
    end
    sort!(unique!(common))

    C = Matrix{Float64}(undef, length(common), length(y))
    V = similar(C)
    for i in eachindex(slices)
        slice = slices[i]
        C[:, i] = linear_interpolation(slice.x, slice.c)(common)
        V[:, i] = linear_interpolation(slice.x, slice.v)(common)
    end
    !isnothing(diagnostics) && (diagnostics.common_grid_projections += 1)
    return Grids2d(common, copy(y), C, V)
end


"""
    egm_step!(agent::AgentAR1; t::Int64)

Solve every conditional AR(1) slice, apply its upper envelope, and project the
slices to one common wealth grid before persistent-state interpolation.
"""
function egm_step!(agent::AgentAR1; t::Int64,
    quadrature::Union{Nothing,EGMQuadratureRule}=nothing,
    diagnostics::Union{Nothing,EGMSolveDiagnostics}=nothing)

    _mark_agent_mutated!(agent)
    continuation_domain = _continuation_domain(agent, t + 1)
    slices = [egm_step_y(agent, income, t=t, quadrature=quadrature,
        diagnostics=diagnostics, continuation_domain=continuation_domain)
        for income in agent.y]
    grids = _common_state_projection(slices, agent.y, diagnostics)

    # Dict-backed storage is mutated serially. Every conditional interpolant now
    # has exactly the same wealth knots, so interpolation across income never
    # mixes different endogenous-grid domains.
    for i in eachindex(agent.y)
        agent.bf.policy[t, i] = linear_interpolation(grids.x, grids.C[:, i],
            extrapolation_bc=Line())
        agent.bf.value[t, i] = linear_interpolation(grids.x, grids.V[:, i],
            extrapolation_bc=Line())
    end
    return grids
end


# EGM setup and solution adapters.

"""Solve a deterministic EGM agent and return its query adapter."""
function solve_egm!(agent::AgentD; material_atol::Real=1e-10,
    material_rtol::Real=1e-8)

    diagnostics = EGMSolveDiagnostics()
    state_grids = Dict{Int,Vector{Float64}}()
    for t in agent.bf.tMax-1:-1:1
        grids = egm_step!(agent, t=t)
        state_grids[t] = copy(grids.x)
    end
    return EGMSolution(agent, state_grids, diagnostics;
        material_atol=material_atol, material_rtol=material_rtol)
end


function _prepare_quadrature(agent::Union{AgentIID,AgentAR1}, quadrature)
    isnothing(quadrature) && return nothing
    quadrature isa EGMQuadratureRule && return quadrature
    # A raw field-compatible rule is interpreted as standard-normal GH nodes.
    # Use `EGMQuadratureRule(nodes, weights)` for already transformed draws.
    return distribution_quadrature(agent.d, quadrature)
end


"""
    solve_egm!(agent::Union{AgentIID,AgentAR1}; quadrature=nothing)

Solve a stochastic EGM agent and return a separate query adapter. Passing a
positive normalized finite-node rule is the reported workflow. Omitting it uses
the explicitly diagnosed adaptive-QuadGK fallback retained for validation.
"""
function solve_egm!(agent::A; quadrature=nothing, material_atol::Real=1e-10,
    material_rtol::Real=1e-8) where {A<:Union{AgentIID,AgentAR1}}

    prepared = _prepare_quadrature(agent, quadrature)
    diagnostics = EGMSolveDiagnostics()
    state_grids = Dict{Int,Vector{Float64}}()
    for t in agent.bf.tMax-1:-1:1
        grids = egm_step!(agent, t=t, quadrature=prepared,
            diagnostics=diagnostics)
        state_grids[t] = copy(grids.x)
    end
    return EGMSolution(agent, state_grids, diagnostics;
        material_atol=material_atol, material_rtol=material_rtol)
end


"""Backward-compatible solver name; now returns an `EGMSolution` adapter."""
egm!(agent::Agent; kwargs...) = solve_egm!(agent; kwargs...)
