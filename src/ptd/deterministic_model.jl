const FT = Union{Float64,Vector{Float64}}

"""
An affine policy segment ``(x_L, x_R, b_0, b_1)`` in the paper's policy
collection.

# Fields
- `xStart::Float64`: The infimum of the state variable on the policy segment.

- `xEnd::Float64`: The supremum of the state variable on the policy segment.

- `b0::Float64`: The intercept b0 in the affine policy function b0 + b1x.

- `b1::Float64`: The slope b1 in the affine policy function b0 + b1x.
"""
@kwdef struct PolicySegment
    xStart::Float64
    xEnd::Float64
    b0::Float64
    b1::Float64
end

# NaN policy segment.
const NaNPolicySegment = PolicySegment(xStart=NaN64, xEnd=NaN64, b0=NaN64, b1=NaN64)
Base.isnan(ps::PolicySegment) = (isnan(ps.xStart) || isnan(ps.xEnd)
                                 || isnan(ps.b0) || isnan(ps.b1))
# Comparison (used for sorting).
Base.isless(ps1::PolicySegment, ps2::PolicySegment) = ps1.xStart < ps2.xStart

# Define NaN policy collection.
const NaNPolicyCollection = [NaNPolicySegment,]
# Check if pc is NaNPolicyCollection.
Base.isnan(pc::Vector{PolicySegment}) = isnan(pc[1]::PolicySegment)

# Last period policy collection (only one policy segment: consuming all).
const LastPolicyCollection = [PolicySegment(xStart=0.0, xEnd=Inf, b0=0.0, b1=1.0),]

# Policy collections are produced by an ordered envelope: at a shared boundary,
# the segment to the right is the (weakly) higher-saving choice.  Keep the
# numerical tolerance local to invariant checks; coverage itself remains exact
# so a genuine gap can never be hidden by a large wealth scale.
const POLICY_INVARIANT_RTOL = 128 * eps(Float64)

_policy_consumption(ps::PolicySegment, x::Float64) = ps.b0 + ps.b1 * x
_policy_saving(ps::PolicySegment, x::Float64) = x - _policy_consumption(ps, x)

function _policy_tolerance(xs::Float64...)
    scale = 1.0
    for x in xs
        isfinite(x) && (scale = max(scale, abs(x)))
    end
    return POLICY_INVARIANT_RTOL * scale
end

function _policy_invariant_error(context::AbstractString, detail::AbstractString,
    policy::Vector{PolicySegment})
    throw(ErrorException("Deterministic PTD policy invariant failed ($context): " *
                         "$detail. policy=$policy"))
end

"""
    validate_policy_collection(policy; context="policy", require_coverage=true)

Check the executable assumptions used by the deterministic ordered envelope.
A solved collection must cover `[0, Inf)` exactly, be feasible segment by
segment, and have weakly increasing saving at every boundary.
"""
function validate_policy_collection(policy::Vector{PolicySegment};
    context::AbstractString="policy", require_coverage::Bool=true)

    isempty(policy) && _policy_invariant_error(context, "collection is empty", policy)

    for (j, ps) in pairs(policy)
        isnan(ps) && _policy_invariant_error(context, "segment $j contains NaN", policy)
        isfinite(ps.xStart) ||
            _policy_invariant_error(context, "segment $j has non-finite xStart", policy)
        ps.xStart >= 0.0 ||
            _policy_invariant_error(context, "segment $j starts below zero", policy)
        (isfinite(ps.xEnd) || ps.xEnd == Inf64) ||
            _policy_invariant_error(context, "segment $j has invalid xEnd", policy)
        ps.xEnd > ps.xStart ||
            _policy_invariant_error(context, "segment $j has an empty domain", policy)
        all(isfinite, (ps.b0, ps.b1)) ||
            _policy_invariant_error(context, "segment $j has non-finite coefficients", policy)
        0.0 <= ps.b1 <= 1.0 ||
            _policy_invariant_error(context,
                "segment $j has slope $(ps.b1) outside [0,1]", policy)

        cStart = _policy_consumption(ps, ps.xStart)
        aStart = ps.xStart - cStart
        tol = _policy_tolerance(ps.xStart, cStart, aStart)
        cStart >= -tol ||
            _policy_invariant_error(context,
                "segment $j implies negative consumption at its left endpoint", policy)
        aStart >= -tol ||
            _policy_invariant_error(context,
                "segment $j implies negative saving at its left endpoint", policy)

        if j > 1
            elder = policy[j-1]
            if elder.xEnd != ps.xStart
                relation = elder.xEnd < ps.xStart ? "gap" : "overlap"
                _policy_invariant_error(context,
                    "$relation between segments $(j-1) and $j: " *
                    "$(elder.xEnd) != $(ps.xStart)", policy)
            end

            boundary = ps.xStart
            elderSaving = _policy_saving(elder, boundary)
            youngerSaving = _policy_saving(ps, boundary)
            savingTol = _policy_tolerance(boundary, elderSaving, youngerSaving)
            youngerSaving >= elderSaving - savingTol ||
                _policy_invariant_error(context,
                    "saving falls at boundary $boundary between segments " *
                    "$(j-1) and $j ($elderSaving -> $youngerSaving)", policy)
        end
    end

    if require_coverage
        policy[1].xStart == 0.0 ||
            _policy_invariant_error(context, "collection does not start at zero", policy)
        policy[end].xEnd == Inf64 ||
            _policy_invariant_error(context, "collection does not end at Inf", policy)
    end

    return nothing
end


"""
The CRRA consumption--saving illustration solved by PTD. Each row of `incomes`
is one deterministic income path. The only borrowing constraint is nonnegative
saving, the transition includes a cash floor, and terminal wealth is consumed
in full. General affine consumption constraints and terminal bequests are not
implemented by this type; the paper's broader class needs separate hypotheses.

# Fields
- `β::Float64`: The discount factor.

- `γ::Float64`: The coefficient of relative risk aversion.

- `r::Float64`: The net interest rate.

- `xFloor::Float64`: The social insurance level (consumption floor).

- `incomes::Matrix{Float64}`: All of the income paths, where `income[i, t]`
is the income at income path `i` and period `t`.

- `tMax::Int64`: _t=1,..., tMax_ in the model.

- `logType::Bool`: True if the problem has a log utility function.

- `policy::Matrix{Vector{PolicySegment}}`: policy functions at different periods,
    where `policy[i, t]` is the array of policy segments for income path `i` at
    period `t`.  The fixed-size matrix permits safe threaded writes to distinct
    income paths.

- `vFloor::Matrix{Float64}`: value function evaluated at the consumption floor
    for each income path and period (this avoids repeated computation).
"""
struct DeterministicProblem
    β::Float64
    γ::Float64
    r::Float64
    xFloor::Float64
    incomes::Matrix{Float64}
    tMax::Int64
    logType::Bool
    policy::Matrix{Vector{PolicySegment}}
    vFloor::Matrix{Float64}
end


"""
    DeterministicProblem(; β::Float64, γ::Float64, r::Float64, xFloor::Float64,
    incomes::Matrix{Float64})::DeterministicProblem

Build a deterministic PTD problem,
where `income[i, t]` is the income at income path i and period t.
"""
function DeterministicProblem(; β::Float64, γ::Float64, r::Float64, xFloor::Float64,
    incomes::Matrix{Float64})::DeterministicProblem

    all(isfinite, (β, γ, r, xFloor)) ||
        throw(ArgumentError("The scalar model parameters must be finite."))
    β > 0.0 || throw(ArgumentError("Discount factor β must be positive."))
    γ > 0.0 || throw(ArgumentError("CRRA coefficient γ must be positive."))
    r > -1.0 || throw(ArgumentError("Gross return 1+r must be positive."))
    xFloor >= 0.0 || throw(ArgumentError("Consumption floor must be nonnegative."))
    !isempty(incomes) || throw(ArgumentError("Income paths must be nonempty."))
    all(isfinite, incomes) || throw(ArgumentError("Income paths must be finite."))

    if γ == 1.0
        logType = true
    else
        logType = false
    end

    n_y, tMax = size(incomes)
    # A fixed-size matrix is safe for threaded writes to distinct rows. Dict is
    # not safe for concurrent mutation, even when keys are pre-initialized.
    policy = [copy(NaNPolicyCollection) for _ in 1:n_y, _ in 1:tMax]
    # Last period policy: consuming all the wealth.
    for i in 1:n_y
        policy[i, tMax] = copy(LastPolicyCollection)
    end

    vFloor = fill(NaN64, n_y, tMax)

    return DeterministicProblem(β, γ, r, xFloor, incomes, tMax, logType, policy, vFloor)
end


"""
    truncate(problem::DeterministicProblem; t::Int64)::DeterministicProblem

Truncate `problem`'s profile from period `t` to the final period,
and create a new `DeterministicProblem`.
"""
function truncate(problem::DeterministicProblem; t::Int64)::DeterministicProblem

    1 <= t <= problem.tMax || throw(ArgumentError(
        "The truncation date must lie in 1:$(problem.tMax)."))
    # The truncated agent's first period is the original agent's t-th period.
    policy = deepcopy(problem.policy[:, t:end])
    vFloor = copy(problem.vFloor[:, t:end])

    tMax = problem.tMax - t + 1
    incomes = problem.incomes[:, t:end]

    return DeterministicProblem(problem.β, problem.γ, problem.r, problem.xFloor,
        incomes, tMax, problem.logType, policy, vFloor)
end


"""
    u(problem::DeterministicProblem, c::T)::T where {T<:FT}

CRRA flow utility at consumption `c`, evaluated elementwise for a vector.
"""
function u(problem::DeterministicProblem, c::T)::T where {T<:FT}
    if problem.logType
        return log.(c)
    else
        return (c .^ (1.0 - problem.γ) .- 1.0) / (1.0 - problem.γ)
    end
end


"""
    mu(problem::DeterministicProblem, c::T)::T where {T<:FT}

Marginal CRRA utility at consumption `c`, elementwise for a vector.
"""
function mu(problem::DeterministicProblem, c::T)::T where {T<:FT}
    if problem.logType
        return 1.0 ./ c
    else
        return c .^ (-problem.γ)
    end
end


"""
    inv_mu(problem::DeterministicProblem, marginal_utility)

Invert marginal utility for a scalar or vector.
"""
function inv_mu(problem::DeterministicProblem, mu::T)::T where {T<:FT}
    if problem.logType
        return 1 ./ mu
    else
        return mu .^ (-1.0 / problem.γ)
    end
end


"""
    c_x(ps::PolicySegment, x::Float64)::Float64

Consumption at wealth `x` on the closed domain of segment `ps`.
"""
function c_x(ps::PolicySegment, x::Float64)::Float64
    if ps.xStart <= x <= ps.xEnd
        return ps.b0 + ps.b1 * x
    else
        throw(ArgumentError("State x=$(x) must be in the interval " *
                            "[xStart=$(ps.xStart), xEnd=$(ps.xEnd)] " *
                            "of the policy segment."))
    end
end


"""
    findfirst(policy_t::Vector{PolicySegment}, x::Float64)::Int64

Find the index of the first policy segment in `policy_t` that contains state `x`.
"""
function Base.findfirst(policy_t::Vector{PolicySegment}, x::Float64)::Int64
    isfinite(x) && x >= 0.0 ||
        throw(ArgumentError("State x must be finite and nonnegative; received x=$x."))
    res = Base.findfirst(ps -> ps.xStart <= x < ps.xEnd, policy_t)
    if isnothing(res)
        throw(ArgumentError("x is not inside any policy segment, " *
                            "x = $x, policy_t = $policy_t"))
    else
        return res
    end
end


"""
    c_x(policy_t::Vector(PolicySegment), x::Float64)::Float64

Given a `policy_t` (time t policy collection) array
composed of many `PolicySegment`s,
calculate an problem's consumption at state `x`.
"""
function c_x(policy_t::Vector{PolicySegment}, x::Float64)::Float64
    ind = findfirst(policy_t, x)::Int64
    ps = policy_t[ind]::PolicySegment
    return ps.b0 + ps.b1 * x
end


"""
    c_x(problem::DeterministicProblem, x::Float64; i::Int64, t::Int64)::Float64

An `problem`'s  consumption at state `x` at income path `i` and period `t`.
"""
function c_x(problem::DeterministicProblem, x::Float64; i::Int64, t::Int64)::Float64
    c_x(problem.policy[i, t]::Vector{PolicySegment}, x::Float64)
end


"""
    zNext_a(problem::DeterministicProblem, a::Float64; i::Int64, t::Int64)::Float64

Given an `problem`'s end-of-period asset level `a`
at the income path `i` and the end of period `t`,
find the next period (t+1) wealth level of the problem **without** social insurance.
"""
function zNext_a(problem::DeterministicProblem, a::Float64; i::Int64, t::Int64)::Float64
    if t < problem.tMax
        return a * (1 + problem.r) + problem.incomes[i, t+1] # Add next period income!
    elseif t == problem.tMax
        return 0.0 # No wealth at T+1.
    else
        throw(ArgumentError("Period t must be between 1, $(problem.tMax)."))
    end
end


"""
    xNext_a(problem::DeterministicProblem, a::Float64; i::Int64, t::Int64)::Float64

Given an `problem`'s end-of-period asset level `a`
at the income path `i` and the end of period `t`,
find the next period (t+1) wealth level of the problem **with** social insurance.
"""
function xNext_a(problem::DeterministicProblem, a::Float64; i::Int64, t::Int64)::Float64
    if t < problem.tMax
        # Next period wealth with social insurance!
        return max(a * (1 + problem.r) + problem.incomes[i, t+1], problem.xFloor)
    elseif t == problem.tMax
        return 0.0 # No wealth and social insurance at T+1.
    else
        throw(ArgumentError("Period t must between 1 and $(problem.tMax)."))
    end

end


"""
    v_x(problem::DeterministicProblem, x::Float64; i::Int64, t::Int64)::Float64

Evaluate discounted flow utility along the solved path from wealth `x`.
This illustration has terminal value `u(x)`, so summing through the terminal
consume-all action is complete. A model with bequests would also need its
separate terminal payoff.
"""
function v_x(problem::DeterministicProblem, x::Float64; i::Int64, t::Int64)::Float64

    flag = (x == problem.xFloor) # Whether x == xFloor at t.

    if flag && !isnan(problem.vFloor[i, t])
        return problem.vFloor[i, t] # Avoid repeated computation.
    else
        v = 0.0
        for j in t:problem.tMax
            c = c_x(problem, x; i=i, t=j)
            v += problem.β^(j - t) * u(problem, c)
            if j < problem.tMax
                a = x - c
                x = xNext_a(problem, a; i=i, t=j)
            end
        end
        # Save floor value if x == problem.xFloor.
        flag ? (problem.vFloor[i, t] = v) : nothing
        return v
    end

end
