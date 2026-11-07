include("envelope.jl")
import Distributions: ContinuousUnivariateDistribution, Distribution

# Union Type for Float64 and Vector{Float64}
const FT = Union{Float64,Vector{Float64}}

# NaN extrapolation object.
const NaNitp1d = linear_interpolation([-Inf64, 0.0], [-Inf64, 0.0],
    extrapolation_bc=Line())

"""
Abstract type for all agents in both deterministic and stochastic models.
"""
abstract type Agent end

"""
Abstract type for all agents in the stochastic models.
"""
abstract type RA <: Agent end

"""
Base fields shared by all kinds of agents 
in deterministoc model and stochastic model.

# Fields
- `β::Float64`: The discount factor.

- `γ::Float64`: The CRRA coefficient.

- `r::Float64`: The net interest rate.

- `xFloor::Float64`: The social insurance level (consumption floor).

- `tMax::Int64`: _t = 1, ..., tMax_ in the model.

- `logType::Bool`: True if the agent has log utility.

- `a::Vector{Float64}`: The exogenous end-of-period asset vector.

- `value::Dict{Int64,Interpolations.Extrapolation}`: 
The dict (t is key) of value function extrapolation objects for each period.

- `policy::Dict{Int64,Interpolations.Extrapolation} or 
Dict{Any,Interpolations.Extrapolation}`: 
The dict of consumption policy function extrapolation objects for each period:
`t` is the key for `AgentD` and `AgentIID`, while `(t, ind_y)` is the key for
`AgentAR1`.
"""
struct BaseFields{T<:Interpolations.Extrapolation}
    β::Float64
    γ::Float64
    r::Float64
    xFloor::Float64
    tMax::Int64
    logType::Bool
    a::Vector{Float64}
    value::Union{Dict{Int,T},Dict{Tuple{Int,Int},T}}
    policy::Union{Dict{Int,T},Dict{Tuple{Int,Int},T}}
    solve_epoch::Base.RefValue{UInt64}
end


"""
A deterministic agent in the deterministic consumption-saving model
with social insurance facing a fixed income path.

# Fields
- `bf::BaseFields`: The base fields of the agent.

- `incomes::Vector{Float64}`: The fixed income path.
"""
struct AgentD <: Agent
    bf::BaseFields
    incomes::Vector{Float64}
end


"""
An agent in the stochastic consumption-saving model
with social insurance facing an i.i.d. income distribution.

# Fields
- `bf::BaseFields`: The base fields of the agent.

- `d::ContinuousUnivariateDistribution`: The i.i.d. income distribution.
"""
struct AgentIID <: RA
    bf::BaseFields
    d::ContinuousUnivariateDistribution
end


"""
An agent in the stochastic consumption-saving model with social insurance
facing a scalar persistent log-AR(1) income process.

# Fields
- `bf::BaseFields`: The base fields of the agent.

- `ρ::Float64`: The autoregressive coefficient of the log income process.

- `d::ContinuousUnivariateDistribution`: The distribution of the white noise in
the log income process.

- `y::Vector{Float64}`: The exogenous income grid.
"""
struct AgentAR1 <: RA
    bf::BaseFields
    ρ::Float64
    d::ContinuousUnivariateDistribution
    y::Vector{Float64}
end


"""
Counters describing how an EGM solution was constructed.

The signed minimum margins diagnose contact with solved continuation domains.
For finite quadrature, `quadrature_income_clamp_mass` records accumulated node
weight at AR(1) income clamps and `quadrature_continuation_mass` is its exposure
denominator; both can exceed one because they accumulate across expectations.
"""
mutable struct EGMSolveDiagnostics
    quadrature_expectations::Int
    quadgk_fallbacks::Int
    common_grid_projections::Int
    continuation_queries::Int
    continuation_wealth_extrapolations::Int
    continuation_income_clamps::Int
    max_continuation_wealth_extrapolation::Float64
    max_continuation_log_income_clamp::Float64
    quadrature_continuation_mass::Float64
    quadrature_income_clamp_mass::Float64
    min_continuation_lower_wealth_margin::Float64
    min_continuation_upper_wealth_margin::Float64
    min_continuation_lower_log_income_margin::Float64
    min_continuation_upper_log_income_margin::Float64
end

EGMSolveDiagnostics() = EGMSolveDiagnostics(0, 0, 0, 0, 0, 0,
    0.0, 0.0, 0.0, 0.0, Inf, Inf, Inf, Inf)


"""
Thread-safe post-solve query diagnostics. Signed minimum margins retain even
sub-tolerance boundary use; contact counters use the configured material query
tolerance, while extrapolation/clamp counters exclude distances within it.
"""
mutable struct EGMQueryDiagnostics
    consumption_queries::Int
    value_queries::Int
    material_wealth_extrapolations::Int
    lower_wealth_boundary_contacts::Int
    upper_wealth_boundary_contacts::Int
    material_income_clamps::Int
    lower_income_boundary_contacts::Int
    upper_income_boundary_contacts::Int
    max_wealth_extrapolation::Float64
    max_income_clamp::Float64
    min_lower_wealth_margin::Float64
    min_upper_wealth_margin::Float64
    min_lower_income_margin::Float64
    min_upper_income_margin::Float64
    lock::ReentrantLock
end

EGMQueryDiagnostics() = EGMQueryDiagnostics(0, 0, 0, 0, 0, 0, 0, 0,
    0.0, 0.0, Inf, Inf, Inf, Inf, ReentrantLock())


"""
Solved EGM adapter. Solving owns the state grids and integration diagnostics;
queries own a separate, resettable diagnostics object.
"""
struct EGMSolution{A<:Agent}
    agent::A
    state_grids::Dict{Int,Vector{Float64}}
    solve_diagnostics::EGMSolveDiagnostics
    query_diagnostics::EGMQueryDiagnostics
    material_atol::Float64
    material_rtol::Float64
    solve_epoch::UInt64

    function EGMSolution(agent::A, state_grids::Dict{Int,Vector{Float64}},
        solve_diagnostics::EGMSolveDiagnostics; material_atol::Real=1e-10,
        material_rtol::Real=1e-8) where {A<:Agent}

        atol, rtol = Float64(material_atol), Float64(material_rtol)
        all(isfinite, (atol, rtol)) && atol >= 0.0 && rtol >= 0.0 ||
            throw(ArgumentError("Material extrapolation tolerances must be finite and nonnegative."))
        all(grid -> length(grid) >= 2 && all(isfinite, grid) &&
                    all(diff(grid) .> 0.0), values(state_grids)) ||
            throw(ArgumentError("Every solved EGM state grid must be finite and strictly increasing."))
        return new{A}(agent, state_grids, solve_diagnostics,
            EGMQueryDiagnostics(), atol, rtol, agent.bf.solve_epoch[])
    end
end


"""Validate and return a scalar continuous shock distribution."""
function _continuous_distribution(d, name::String)::ContinuousUnivariateDistribution
    d isa ContinuousUnivariateDistribution || throw(ArgumentError(
        "$name must be a continuous univariate distribution."))
    return d
end




"""
   build_agent(; model::String, β::Float64, γ::Float64, r::Float64,
    xFloor::Float64, tMax::Int64, a::Vector{Float64},
    last_grids::Grids, kwargs...)::Agent

Build an `Agent` in the `model`:
- `model::String`: The model type can only takes one of 
`"deterministic"`, `"iid"`, and `"ar1"`.

- `β::Float64`: The discount factor.

- `γ::Float64`: The CRRA coefficient.

- `r::Float64`: The net interest rate.

- `xFloor::Float64`: The social insurance level (consumption floor).

- `tMax::Int64`: _t = 1, ..., tMax_ in the model.

- `a::Vector{Float64}`: The exogenous end-of-period asset vector.

- `kwargs...`: Other keyword arguments for different models.
They should be 
    - `incomes` if `model` is `"deterministic"`;
    - `d` if `model` is `"iid"`;
    - `ρ`, `d`, and `y` if `model` is `"ar1"`.
"""
function build_agent(; model::String, β::Float64, γ::Float64, r::Float64,
    xFloor::Float64, tMax::Int64, a::Vector{Float64}, kwargs...)::Agent

    all(isfinite, (β, γ, r, xFloor)) ||
        throw(ArgumentError("The scalar model parameters must be finite."))
    β > 0.0 || throw(ArgumentError("Discount factor β must be positive."))
    γ > 0.0 || throw(ArgumentError("CRRA coefficient γ must be positive."))
    r > -1.0 || throw(ArgumentError("Gross return 1+r must be positive."))
    xFloor >= 0.0 || throw(ArgumentError("Consumption floor must be nonnegative."))
    tMax >= 1 || throw(ArgumentError("The horizon must contain at least one period."))
    length(a) >= 2 || throw(ArgumentError("The asset grid needs at least two points."))
    a[1] == 0.0 || throw(ArgumentError("The asset grid must start at zero."))
    all(isfinite, a) && all(a .>= 0.0) ||
        throw(ArgumentError("The asset grid must be finite and nonnegative."))
    @assert begin
        all(a[1:end-1] .< a[2:end])
    end "a must be strictly increasing."

    logType = utility_type(γ)::Bool

    # initialize the last period policy and value function.
    policy, value = init_policy_value(model, tMax; kwargs...)
    bf = BaseFields(β, γ, r, xFloor, tMax, logType, a, value, policy,
        Ref(UInt64(0)))

    return build_agent(bf, model; kwargs...)

end


"""
    build_agent(bf::BaseFields, model::String; kwargs...)::Agent

Build an agent with the base fields `bf` in the `model`.
"""
function build_agent(bf::BaseFields, model::String; kwargs...)::Agent
    if model == "deterministic"
        incomes = kwargs[:incomes]::Vector{Float64}
        length(incomes) == bf.tMax ||
            throw(ArgumentError("The deterministic income path must have tMax entries."))
        all(isfinite, incomes) ||
            throw(ArgumentError("The deterministic income path must be finite."))
        return AgentD(bf, incomes)
    elseif model == "iid"
        d = _continuous_distribution(kwargs[:d], "The IID income distribution")
        return AgentIID(bf, d)
    elseif model == "ar1"
        y = kwargs[:y]::Vector{Float64}
        length(y) >= 2 && all(isfinite, y) && all(y .> 0.0) &&
            all(diff(y) .> 0.0) || throw(ArgumentError(
                "The AR(1) income grid must contain at least two finite, increasing positive levels."))
        ρ = kwargs[:ρ]::Float64
        isfinite(ρ) || throw(ArgumentError("The AR(1) coefficient must be finite."))
        d = _continuous_distribution(kwargs[:d], "The AR(1) innovation distribution")
        return AgentAR1(bf, ρ, d, y)
    else
        throw(ArgumentError("Model type must be one of " *
                            """ "deterministic", "iid", and "ar1"."""))
    end
end


"""
    utility_type(γ::Float64)::Bool

Check the utility type of an agent with CRRA coefficient `γ`.
"""
function utility_type(γ::Float64)::Bool
    if γ == 1.0
        return true
    elseif 0.0 < γ
        return false
    else
        throw(ArgumentError("CRRA coefficeint γ must " *
                            "be in the interval (0, +∞)."))
    end
end


"""
    init_policy_value(model::String, tMax::Int64)

Initialize the policy and value function dicts 
for agent in `model` with the final period `tMax` grids `last_grid`.
"""
function init_policy_value(model::String, tMax::Int64; kwargs...)
    if model in ["deterministic", "iid"]
        # We don't need to store the policy and value functions at tMax 
        # since we have analytical solutions.
        policy = Dict(i => NaNitp1d for i in 1:tMax-1)
        value = Dict(i => NaNitp1d for i in 1:tMax-1)
    elseif model == "ar1"
        policy = Dict((i, j) => NaNitp1d
                      for i in 1:tMax-1, j in 1:length(kwargs[:y]))
        value = Dict((i, j) => NaNitp1d
                     for i in 1:tMax-1, j in 1:length(kwargs[:y]))
    else
        throw(ArgumentError("Wrong model type."))
    end

    return policy, value
end


# u, mu and inv_mu functions for agents of all types.

"""
    u(agent::Agent, c::T)::T where T<:FT

utility (Float or Vector) of an `agent` consuming a `c` (Float or Vector).
"""
function u(agent::Agent, c::T)::T where {T<:FT}
    if agent.bf.logType
        return log.(c)
    else
        return (c .^ (1.0 - agent.bf.γ) .- 1.0) / (1.0 - agent.bf.γ)
    end
end


"""
    mu(agent::Agent, c::T)::T where {T<:FT}

marginal utility Vector/Float of an `agent` consuming Vector/Float `c`.
"""
function mu(agent::Agent, c::T)::T where {T<:FT}
    if agent.bf.logType
        return 1.0 ./ c
    else
        return c .^ (-agent.bf.γ)
    end
end


"""
    inv_mu(agent::Agent, mu::T}::T where {T<:FT}

comsumption vector of an `agent` at the marginal utility vector `mu`.
"""
function inv_mu(agent::Agent, mu::T)::T where {T<:FT}
    if agent.bf.logType
        return 1 ./ mu
    else
        return mu .^ (-1.0 / agent.bf.γ)
    end
end


# The following functions are agent type specific.

_feasible_consumption(c::Float64, x::Float64) = clamp(c, 0.0, x)
_feasible_consumption(c::Vector{Float64}, x::Vector{Float64}) = clamp.(c, 0.0, x)

## AgentD

"""
    c_x(agent::AgentD, x::T; t::Int64)::T where {T<:FT}

An `agent`'s consumptions at states `x` at `t`.
"""
function c_x(agent::AgentD, x::T; t::Int64)::T where {T<:FT}
    return t == agent.bf.tMax ? x : _feasible_consumption(agent.bf.policy[t](x), x)
end


"""
    zNext_a(agent::AgentD, a::T; t::Int64)::T where {T<:FT}

Given an deterministic `agent`'s end-of-period assets level (Vector or Float) `a` 
at the the end of period `t`,
find the next period (t+1) wealth level of the agentD **without** social insurance.
"""
function zNext_a(agent::AgentD, a::T; t::Int64)::T where {T<:FT}
    if t < agent.bf.tMax
        return a .* (1 + agent.bf.r) .+ agent.incomes[t+1] # add next period income!
    elseif t == agent.bf.tMax
        return a isa Float64 ? 0.0 : zeros(length(a)) # no wealth at T+1
    else
        throw(ArgumentError("Period t must be between 1 and $(agent.bf.tMax)."))
    end
end


"""
    xNext_a(agent::AgentD, a::T; t::Int64)::T where {T<:FT}

Given a deterministic `agent`'s end-of-period assets level (Vector or Float) `a` 
at the the end of period `t`,
find the next period (t+1) wealth levels of the agentD **with** social insurance.
"""
function xNext_a(agent::AgentD, a::T; t::Int64)::T where {T<:FT}
    if t < agent.bf.tMax
        # next period income with social insurance!
        return max.(a .* (1 .+ agent.bf.r) .+ agent.incomes[t+1], agent.bf.xFloor)
    elseif t == agent.bf.tMax
        # no wealth and social insurance at T+1
        return a isa Float64 ? 0.0 : zeros(length(a))
    else
        throw(ArgumentError("Period t must be between 1 and $(agent.bf.tMax)."))
    end
end


"""
    v_x(agent::AgentD, x::T; t::Int64)::T where {T<:FT}

A deterministic `agent`'s values at states (Vector or Float) `x`, period `t`.
"""
function v_x(agent::AgentD, x::T; t::Int64)::T where {T<:FT}
    if t < agent.bf.tMax
        return agent.bf.value[t](x)
    elseif t == agent.bf.tMax
        return u(agent, x) # Agent will consume all the wealth at the final period.
    end
end


## AgentIID & AgentAR1

"""
    zNext_a(agent::RA, a::T, yNext::T; t::Int64)::T where {T<:FT}

Given a random `agent`'s end-of-period assets level (Vector or Float) `a`
at the the end of period `t` and the next period income (Vector or Float) `yNext`,
find the next period (t+1) wealth level **without** social insurance.
"""
function zNext_a(agent::RA, a::T, yNext::T; t::Int64)::T where {T<:FT}
    if t < agent.bf.tMax
        return a .* (1 + agent.bf.r) .+ yNext
    elseif t == agent.bf.tMax
        return a isa Float64 ? 0.0 : zeros(length(a)) # No wealth at tMax+1.
    else
        throw(ArgumentError("Period t must be between 1 and $(agent.bf.tMax)."))
    end
end


"""
    xNext_a(agent::RA, a::T, yNext::T; t::Int64)::T where {T<:FT}

Given a random `agent`'s end-of-period assets level (Vector or Float) `a`
at the the end of period `t` and the next period income (Vector or Float) `yNext`,
find the next period (t+1) wealth levels **with** social insurance.
"""
function xNext_a(agent::RA, a::T, yNext::T; t::Int64)::T where {T<:FT}
    if t < agent.bf.tMax
        # Next period income with social insurance.
        return max.(a .* (1 .+ agent.bf.r) .+ yNext, agent.bf.xFloor)
    elseif t == agent.bf.tMax
        return a isa Float64 ? 0.0 : zeros(length(a)) # No wealth at tMax+1.
    else
        throw(ArgumentError("Period t must be between 1 and $(agent.bf.tMax)."))
    end
end


# AgentIID

"""
    c_x(agent::AgentIID, x::T; t::Int64)::T where {T<:FT}

An IID type `agent`'s consumptions at states `x` at `t`.
"""
function c_x(agent::AgentIID, x::T; t::Int64)::T where {T<:FT}
    return t == agent.bf.tMax ? x : _feasible_consumption(agent.bf.policy[t](x), x)
end


"""
    v_x(agent::AgentIID, x::T; t::Int64)::T where {T<:FT}

An IID type `agent`'s values at states (Vector or Float) `x`, period `t`.
"""
function v_x(agent::AgentIID, x::T; t::Int64)::T where {T<:FT}
    if t < agent.bf.tMax
        return agent.bf.value[t](x)
    elseif t == agent.bf.tMax
        return u(agent, x) # Agent will consume all the wealth at the final period.
    else
        throw(ArgumentError("Period t must be between 1 and $(agent.bf.tMax)."))
    end
end


## AgentAR1
"""
    c_x(agent::AgentAR1, x::Float64, y::Float64; t::Int64)::Float64

An AR1 type `agent`'s consumptions at state (`x`, `y`) at `t`.
"""
function c_x(agent::AgentAR1, x::Float64, y::Float64; t::Int64)::Float64
    if t == agent.bf.tMax
        return x
    else
        i = searchsortedlast(agent.y, y)
        if i == 0
            return _feasible_consumption(agent.bf.policy[t, 1](x), x)
        elseif i ≥ length(agent.y)
            return _feasible_consumption(agent.bf.policy[t, length(agent.y)](x), x)
        else # 2D linear interpolation.
            c1 = agent.bf.policy[t, i](x)
            c2 = agent.bf.policy[t, i+1](x)
            w1 = (agent.y[i+1] - y) / (agent.y[i+1] - agent.y[i])
            w2 = 1.0 - w1
            return _feasible_consumption(w1 * c1 + w2 * c2, x)
        end
    end
end


"""
    v_x(agent::AgentAR1, x::Float64, y::Float64; t::Int64)::Float64

An AR1 type `agent`'s value at state (`x`, `y`), period `t`.
"""
function v_x(agent::AgentAR1, x::Float64, y::Float64; t::Int64)::Float64
    if t == agent.bf.tMax # Agent will consume all the wealth at the final period.
        return u(agent, x)
    else
        i = searchsortedlast(agent.y, y)
        if i == 0
            return agent.bf.value[t, 1](x)
        elseif i ≥ length(agent.y)
            return agent.bf.value[t, length(agent.y)](x)
        else # 2D Linear interpolation.
            v1 = agent.bf.value[t, i](x)
            v2 = agent.bf.value[t, i+1](x)
            w1 = (agent.y[i+1] - y) / (agent.y[i+1] - agent.y[i])
            w2 = 1.0 - w1
            return w1 * v1 + w2 * v2
        end
    end
end


"""Reset query diagnostics stored on a solved DCEGM model."""
function reset_query_diagnostics!(solution::EGMSolution)
    diagnostics = solution.query_diagnostics
    lock(diagnostics.lock) do
        diagnostics.consumption_queries = 0
        diagnostics.value_queries = 0
        diagnostics.material_wealth_extrapolations = 0
        diagnostics.lower_wealth_boundary_contacts = 0
        diagnostics.upper_wealth_boundary_contacts = 0
        diagnostics.material_income_clamps = 0
        diagnostics.lower_income_boundary_contacts = 0
        diagnostics.upper_income_boundary_contacts = 0
        diagnostics.max_wealth_extrapolation = 0.0
        diagnostics.max_income_clamp = 0.0
        diagnostics.min_lower_wealth_margin = Inf
        diagnostics.min_upper_wealth_margin = Inf
        diagnostics.min_lower_income_margin = Inf
        diagnostics.min_upper_income_margin = Inf
    end
    return diagnostics
end


function _query_tolerance(solution::EGMSolution, values::Real...)
    scale = max(1.0, maximum(abs, values))
    return solution.material_atol + solution.material_rtol * scale
end


function _validate_solution_query(solution::EGMSolution, x::Float64, t::Int64)
    solution.solve_epoch == solution.agent.bf.solve_epoch[] || throw(ErrorException(
        "This EGM solution is stale because its agent has since been re-solved. " *
        "Build one fresh agent per numerical configuration."))
    isfinite(x) && x >= 0.0 || throw(ArgumentError(
        "Queried wealth must be finite and nonnegative."))
    1 <= t <= solution.agent.bf.tMax || throw(ArgumentError(
        "Period t must be between 1 and $(solution.agent.bf.tMax)."))
    if t < solution.agent.bf.tMax && !haskey(solution.state_grids, t)
        throw(ArgumentError("Period $t has not been solved by this EGM adapter."))
    end
    return nothing
end


function _record_query!(solution::EGMSolution, x::Float64, t::Int64;
    value_query::Bool, income::Union{Nothing,Float64}=nothing)

    _validate_solution_query(solution, x, t)
    wealth_distance = 0.0
    lower_wealth_margin = Inf
    upper_wealth_margin = Inf
    lower_wealth_contact = false
    upper_wealth_contact = false
    if t < solution.agent.bf.tMax
        grid = solution.state_grids[t]
        lower_wealth_margin = x - first(grid)
        upper_wealth_margin = last(grid) - x
        tolerance = _query_tolerance(solution, x, first(grid), last(grid))
        wealth_distance = max(-lower_wealth_margin, -upper_wealth_margin, 0.0)
        wealth_distance <= tolerance &&
            (wealth_distance = 0.0)
        lower_wealth_contact = abs(lower_wealth_margin) <= tolerance
        upper_wealth_contact = abs(upper_wealth_margin) <= tolerance
    end

    income_distance = 0.0
    lower_income_margin = Inf
    upper_income_margin = Inf
    lower_income_contact = false
    upper_income_contact = false
    if !isnothing(income)
        isfinite(income) && income > 0.0 || throw(ArgumentError(
            "Queried AR(1) income must be finite and positive."))
    end
    if !isnothing(income) && t < solution.agent.bf.tMax
        agent = solution.agent::AgentAR1
        lower_income_margin = income - first(agent.y)
        upper_income_margin = last(agent.y) - income
        tolerance = _query_tolerance(solution, income,
            first(agent.y), last(agent.y))
        income_distance = max(-lower_income_margin, -upper_income_margin, 0.0)
        income_distance <= tolerance && (income_distance = 0.0)
        lower_income_contact = abs(lower_income_margin) <= tolerance
        upper_income_contact = abs(upper_income_margin) <= tolerance
    end

    diagnostics = solution.query_diagnostics
    lock(diagnostics.lock) do
        if value_query
            diagnostics.value_queries += 1
        else
            diagnostics.consumption_queries += 1
        end
        if wealth_distance > 0.0
            diagnostics.material_wealth_extrapolations += 1
            diagnostics.max_wealth_extrapolation = max(
                diagnostics.max_wealth_extrapolation, wealth_distance)
        end
        diagnostics.lower_wealth_boundary_contacts += lower_wealth_contact
        diagnostics.upper_wealth_boundary_contacts += upper_wealth_contact
        diagnostics.min_lower_wealth_margin = min(
            diagnostics.min_lower_wealth_margin, lower_wealth_margin)
        diagnostics.min_upper_wealth_margin = min(
            diagnostics.min_upper_wealth_margin, upper_wealth_margin)
        if income_distance > 0.0
            diagnostics.material_income_clamps += 1
            diagnostics.max_income_clamp = max(diagnostics.max_income_clamp,
                income_distance)
        end
        diagnostics.lower_income_boundary_contacts += lower_income_contact
        diagnostics.upper_income_boundary_contacts += upper_income_contact
        diagnostics.min_lower_income_margin = min(
            diagnostics.min_lower_income_margin, lower_income_margin)
        diagnostics.min_upper_income_margin = min(
            diagnostics.min_upper_income_margin, upper_income_margin)
    end
    return nothing
end


function c_x(solution::EGMSolution{A}, x::Real; t::Int64)::Float64 where
    {A<:Union{AgentD,AgentIID}}

    wealth = Float64(x)
    _record_query!(solution, wealth, t; value_query=false)
    return c_x(solution.agent, wealth; t=t)
end


function v_x(solution::EGMSolution{A}, x::Real; t::Int64)::Float64 where
    {A<:Union{AgentD,AgentIID}}

    wealth = Float64(x)
    _record_query!(solution, wealth, t; value_query=true)
    return v_x(solution.agent, wealth; t=t)
end


function c_x(solution::EGMSolution{AgentAR1}, x::Real, y::Real;
    t::Int64)::Float64

    wealth, income = Float64(x), Float64(y)
    _record_query!(solution, wealth, t; value_query=false, income=income)
    return c_x(solution.agent, wealth, income; t=t)
end


function v_x(solution::EGMSolution{AgentAR1}, x::Real, y::Real;
    t::Int64)::Float64

    wealth, income = Float64(x), Float64(y)
    _record_query!(solution, wealth, t; value_query=true, income=income)
    return v_x(solution.agent, wealth, income; t=t)
end


"""Return the DCEGM consumption action at the requested state."""
action(solution::EGMSolution, args...; kwargs...) = c_x(solution, args...; kwargs...)

"""Return the DCEGM value at the requested state."""
value_at(solution::EGMSolution, args...; kwargs...) = v_x(solution, args...; kwargs...)
