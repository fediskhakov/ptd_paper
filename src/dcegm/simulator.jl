using Random
include("solver.jl")


function _validate_initial_wealth(xList::Vector{Float64})
    all(isfinite, xList) ||
        throw(ArgumentError("Initial wealth levels must be finite."))
    all(xList .>= 0.0) ||
        throw(ArgumentError("Initial wealth levels must be nonnegative."))
    return nothing
end


function _validate_positive_initial_state(values::Vector{Float64}, name::String)
    all(isfinite, values) && all(values .> 0.0) ||
        throw(ArgumentError("Initial $name levels must be finite and positive."))
    return nothing
end


"""
Parameters of the life cycle with i.i.d. income distribution.

# Fields
- `agent`: An instance of `AgentIID`.

- `xList`: A list of initial wealth levels for each agent.
"""
struct IIDModel
    agent::AgentIID
    xList::Vector{Float64}

    function IIDModel(agent::AgentIID, xList::Vector{Float64})
        _validate_initial_wealth(xList)
        return new(agent, xList)
    end
end

IIDModel(agent::AgentIID, xList::AbstractVector{<:Real}) =
    IIDModel(agent, collect(Float64, xList))


"""
Parameters of the life cycle model with AR(1) log income process.

# Fields
- `agent`: An instance of `AgentAR1`.

- `xList`: A list of initial wealth levels for each simulation.

- `yList`: A list of initial income levels for each simulation.

- `ρ`: AR(1) coefficient in the log income process. This compatibility field
  is required to equal `agent.ρ`.

- `d`: The continuous white-noise distribution. This compatibility field is
  required to equal `agent.d`.
"""
struct AR1Model
    agent::AgentAR1
    xList::Vector{Float64}
    yList::Vector{Float64}
    ρ::Float64
    d::ContinuousUnivariateDistribution

    function AR1Model(agent::AgentAR1, xList::Vector{Float64},
        yList::Vector{Float64}, ρ::Float64,
        d::ContinuousUnivariateDistribution)

        length(xList) == length(yList) || throw(ArgumentError(
            "xList and yList must have the same length."))
        _validate_initial_wealth(xList)
        _validate_positive_initial_state(yList, "income")
        isfinite(ρ) || throw(ArgumentError("The simulation AR(1) coefficient must be finite."))
        ρ == agent.ρ || throw(ArgumentError(
            "The simulation AR(1) coefficient must equal the solved agent coefficient."))
        isequal(d, agent.d) || throw(ArgumentError(
            "The simulation innovation distribution must equal the solved agent distribution."))
        return new(agent, xList, yList, agent.ρ, agent.d)
    end
end

AR1Model(agent::AgentAR1, xList::Vector{Float64}, yList::Vector{Float64}) =
    AR1Model(agent, xList, yList, agent.ρ, agent.d)

AR1Model(agent::AgentAR1, xList::AbstractVector{<:Real},
    yList::AbstractVector{<:Real}) =
    AR1Model(agent, collect(Float64, xList), collect(Float64, yList))

function AR1Model(agent::AgentAR1, xList::AbstractVector{<:Real},
    yList::AbstractVector{<:Real}, ρ::Real, d::Distribution)

    continuousD = _continuous_distribution(d, "The simulation innovation distribution")
    return AR1Model(agent, collect(Float64, xList), collect(Float64, yList),
        Float64(ρ), continuousD)
end


"""
    init_simulation(model::IIDModel; seed::Int64=42)

Initialize the simulation of the model with i.i.d. income distribution.

# Returns
- `T`: number of periods.
- `res::Dict{Char,Matrix{Float64}}`: simulated income, wealth, and consumption paths.
"""
function init_simulation(model::IIDModel; seed::Int64=42)
    rng = Random.MersenneTwister(seed)

    N = length(model.xList) # Number of initial wealth levels.
    T = model.agent.bf.tMax # Number of periods.
    y = rand(rng, model.agent.d, (N, T)) # Realizations of income paths for all agents.
    res = Dict('y' => y,
        'x' => Matrix{Float64}(undef, N, T),
        'c' => Matrix{Float64}(undef, N, T)) # Simulation results.
    res['x'][:, 1] .= model.xList # Wealth levels at t=1.

    return T, res
end


"""
    init_simulation(model::AR1Model; seed::Int64=42)

Initialize the simulation of the model with AR(1) income distribution.

# Returns
- `T`: number of periods.
- `res::Dict{Char,Matrix{Float64}}`: simulated income, wealth, and consumption paths.
"""
function init_simulation(model::AR1Model; seed::Int64=42)
    rng = Random.MersenneTwister(seed)

    N = length(model.xList) # Number of initial wealth levels.
    T = model.agent.bf.tMax # Number of periods.

    ln_y = Matrix{Float64}(undef, N, T)
    ln_y[:, 1] .= log.(model.yList) # Log incomes at t=1.
    # white noises on all realization of all the income paths.
    wn = rand(rng, model.agent.d, (N, T))
    for t in 2:T
        ln_y[:, t] .= model.agent.ρ * ln_y[:, t-1] .+ wn[:, t]
    end
    y = exp.(ln_y) # Realizations of income paths for all agents.

    res = Dict('y' => y,
        'x' => Matrix{Float64}(undef, N, T),
        'c' => Matrix{Float64}(undef, N, T)) # Simulation results saved here.
    res['x'][:, 1] .= model.xList # Wealth levels at t=1.

    return T, res
end


"""
    sim_xNext(agent::Agent, x::Float64, y::Float64, t::Int64)::Float64

Simulate `agent`'s next period wealth level given current wealth `x`,
consumption `c`, and income `yNext` at the next period.
"""
function sim_xNext(agent::Agent, x::Float64, c::Float64, yNext::Float64)::Float64
    zNext = (1 + agent.bf.r) * (x - c) + yNext
    xNext = max(agent.bf.xFloor, zNext)
    return xNext
end


function _validated_external_paths(model, income_paths::AbstractMatrix{<:Real};
    positive::Bool)

    paths = Matrix{Float64}(income_paths)
    size(paths) == (length(model.xList), model.agent.bf.tMax) ||
        throw(ArgumentError("External income paths must have size " *
            "(length(xList), tMax)."))
    all(isfinite, paths) || throw(ArgumentError(
        "External income paths must be finite."))
    positive && !all(paths .> 0.0) && throw(ArgumentError(
        "External AR(1) income paths must be positive."))
    return paths
end


function _validate_solution_model(solution::EGMSolution, model)
    solution.agent === model.agent || throw(ArgumentError(
        "The solution adapter and simulation model must own the same agent."))
    return nothing
end


"""
    evaluate_paths(model::IIDModel, income_paths, solution::EGMSolution)

Evaluate evaluator-owned IID shock paths without solving, drawing random
numbers, or touching the global RNG. A single solution can therefore be reused
across contest simulations.
"""
function evaluate_paths(model::IIDModel, income_paths::AbstractMatrix{<:Real},
    solution::EGMSolution{AgentIID})

    _validate_solution_model(solution, model)
    y = _validated_external_paths(model, income_paths; positive=false)
    N, T = size(y)
    x = Matrix{Float64}(undef, N, T)
    c = similar(x)
    x[:, 1] .= model.xList
    for i in axes(x, 1)
        for t in 1:T-1
            c[i, t] = action(solution, x[i, t]; t=t)
            x[i, t+1] = sim_xNext(model.agent, x[i, t], c[i, t], y[i, t+1])
        end
        c[i, T] = x[i, T]
    end
    return Dict('y' => y, 'x' => x, 'c' => c)
end


"""Evaluate evaluator-owned positive AR(1) income paths without RNG or setup."""
function evaluate_paths(model::AR1Model, income_paths::AbstractMatrix{<:Real},
    solution::EGMSolution{AgentAR1})

    _validate_solution_model(solution, model)
    y = _validated_external_paths(model, income_paths; positive=true)
    all(isapprox.(y[:, 1], model.yList; rtol=8eps(Float64), atol=0.0)) ||
        throw(ArgumentError("The first AR(1) path column must equal the " *
            "observed initial income state in yList."))
    N, T = size(y)
    x = Matrix{Float64}(undef, N, T)
    c = similar(x)
    x[:, 1] .= model.xList
    for i in axes(x, 1)
        for t in 1:T-1
            c[i, t] = action(solution, x[i, t], y[i, t]; t=t)
            x[i, t+1] = sim_xNext(model.agent, x[i, t], c[i, t], y[i, t+1])
        end
        c[i, T] = x[i, T]
    end
    return Dict('y' => y, 'x' => x, 'c' => c)
end


"""Convenience wrapper that draws local IID paths, solves once, then evaluates."""
function simulation_iid(model::IIDModel; seed::Int64=42, quadrature=nothing)
    _, initialized = init_simulation(model, seed=seed)
    solution = solve_egm!(model.agent; quadrature=quadrature)
    return evaluate_paths(model, initialized['y'], solution)
end


"""Convenience wrapper that draws local AR(1) paths, solves once, then evaluates."""
function simulation_ar1(model::AR1Model; seed::Int64=42, quadrature=nothing)
    _, initialized = init_simulation(model, seed=seed)
    solution = solve_egm!(model.agent; quadrature=quadrature)
    return evaluate_paths(model, initialized['y'], solution)
end
