"""
Stochastic model primitives shared by PTD-SCEQ, PTD-OSA, and the simulators.

This file defines income processes, observed states, the true transition, and
Gauss-Hermite quadrature. It contains neither a policy solver nor simulation RNG.
"""

import FastGaussQuadrature: gausshermite
import SHA

"""Supertype for exogenous income processes used by stochastic PTD."""
abstract type AbstractIncomeProcess end

"""No persistent exogenous state is needed for IID income."""
struct IIDState end

"""Observed log income for the scalar log-AR(1) process."""
struct LogAR1State
    log_income::Float64

    function LogAR1State(log_income::Real)
        value = Float64(log_income)
        isfinite(value) || throw(ArgumentError("Log income must be finite."))
        return new(value)
    end
end

"""Observed log earnings and log medical expenses in the HSZ process."""
struct HSZState
    log_earnings::Float64
    log_medical::Float64

    function HSZState(log_earnings::Real, log_medical::Real)
        earnings = Float64(log_earnings)
        medical = Float64(log_medical)
        all(isfinite, (earnings, medical)) || throw(ArgumentError(
            "Log earnings and log medical expenses must be finite."))
        return new(earnings, medical)
    end
end

"""
    IIDLognormalProcess(; sigma, mean_income=1.0)

IID lognormal income. The log location is derived from `mean_income`, so
`mean_income=1` gives the paper's mean-one parameterization
`log(Y) = -sigma^2/2 + sigma * epsilon`.
"""
struct IIDLognormalProcess <: AbstractIncomeProcess
    log_mean::Float64
    log_std::Float64

    function IIDLognormalProcess(; sigma::Real, mean_income::Real=1.0)
        sigma_f = Float64(sigma)
        mean_f = Float64(mean_income)
        isfinite(sigma_f) && sigma_f >= 0.0 || throw(ArgumentError(
            "The IID log standard deviation must be finite and nonnegative."))
        isfinite(mean_f) && mean_f > 0.0 || throw(ArgumentError(
            "Mean IID income must be finite and positive."))
        return new(log(mean_f) - sigma_f^2 / 2.0, sigma_f)
    end
end

"""
    LogAR1Process(; rho, sigma, unconditional_log_mean=0.0)

Scalar Gaussian log-income process
`ell_next = mu + rho * (ell - mu) + sigma * epsilon`.
"""
struct LogAR1Process <: AbstractIncomeProcess
    rho::Float64
    log_std::Float64
    unconditional_log_mean::Float64

    function LogAR1Process(; rho::Real, sigma::Real,
        unconditional_log_mean::Real=0.0)
        rho_f = Float64(rho)
        sigma_f = Float64(sigma)
        mean_f = Float64(unconditional_log_mean)
        isfinite(rho_f) && abs(rho_f) < 1.0 || throw(ArgumentError(
            "The log-AR(1) coefficient must be finite and have absolute value below one."))
        isfinite(sigma_f) && sigma_f >= 0.0 || throw(ArgumentError(
            "The log-AR(1) innovation standard deviation must be finite and nonnegative."))
        isfinite(mean_f) || throw(ArgumentError(
            "The unconditional log-income mean must be finite."))
        return new(rho_f, sigma_f, mean_f)
    end
end

"""
    HSZProcess(; rho_earnings, sigma_earnings, rho_medical, sigma_medical,
                 unconditional_log_earnings=0.0,
                 unconditional_log_medical=0.0)

Two-state HSZ-style process with independent standard-normal innovations in log
earnings and log medical expenses. Net income is earnings minus medical expense.
"""
struct HSZProcess <: AbstractIncomeProcess
    rho_earnings::Float64
    std_earnings::Float64
    unconditional_log_earnings::Float64
    rho_medical::Float64
    std_medical::Float64
    unconditional_log_medical::Float64

    function HSZProcess(; rho_earnings::Real, sigma_earnings::Real,
        rho_medical::Real, sigma_medical::Real,
        unconditional_log_earnings::Real=0.0,
        unconditional_log_medical::Real=0.0)
        rho_e = Float64(rho_earnings)
        std_e = Float64(sigma_earnings)
        mean_e = Float64(unconditional_log_earnings)
        rho_m = Float64(rho_medical)
        std_m = Float64(sigma_medical)
        mean_m = Float64(unconditional_log_medical)
        all(isfinite, (rho_e, std_e, mean_e, rho_m, std_m, mean_m)) ||
            throw(ArgumentError("Every HSZ process parameter must be finite."))
        abs(rho_e) < 1.0 && abs(rho_m) < 1.0 || throw(ArgumentError(
            "Both HSZ autoregressive coefficients must have absolute value below one."))
        std_e >= 0.0 && std_m >= 0.0 || throw(ArgumentError(
            "Both HSZ innovation standard deviations must be nonnegative."))
        return new(rho_e, std_e, mean_e, rho_m, std_m, mean_m)
    end
end

"""Observed date, post-transfer cash on hand, and current exogenous state."""
struct ObservedState{S}
    date::Int
    cash::Float64
    exogenous::S

    function ObservedState(date::Integer, cash::Real, exogenous::S) where {S}
        date_i = Int(date)
        cash_f = Float64(cash)
        date_i >= 1 || throw(ArgumentError("The observed date must be positive."))
        isfinite(cash_f) && cash_f >= 0.0 || throw(ArgumentError(
            "Post-transfer cash must be finite and nonnegative."))
        return new{S}(date_i, cash_f, exogenous)
    end
end

"""
Result of one true transition.

`z` is next-period cash before the floor transfer, `transfer` is the government
transfer, `xnext` is post-transfer cash, and `snext` is the next exogenous state.
"""
struct TransitionResult{S}
    z::Float64
    transfer::Float64
    xnext::Float64
    snext::S
end

"""Finite-horizon terminal rule: consume all cash at `final_date`."""
struct TerminalSpec
    final_date::Int

    function TerminalSpec(final_date::Integer)
        date = Int(final_date)
        date >= 1 || throw(ArgumentError("The terminal date must be positive."))
        return new(date)
    end
end

"""Immutable preferences, return, floor, and terminal configuration."""
struct StochasticPTDConfig
    beta::Float64
    gamma::Float64
    return_rate::Float64
    cash_floor::Float64
    terminal::TerminalSpec

    function StochasticPTDConfig(; beta::Real, gamma::Real, return_rate::Real,
        cash_floor::Real, terminal_date::Integer)
        beta_f = Float64(beta)
        gamma_f = Float64(gamma)
        return_f = Float64(return_rate)
        floor_f = Float64(cash_floor)
        all(isfinite, (beta_f, gamma_f, return_f, floor_f)) || throw(ArgumentError(
            "Every scalar stochastic-PTD configuration value must be finite."))
        beta_f > 0.0 || throw(ArgumentError("The discount factor must be positive."))
        gamma_f > 0.0 || throw(ArgumentError("CRRA must be positive."))
        return_f > -1.0 || throw(ArgumentError("The gross return must be positive."))
        floor_f >= 0.0 || throw(ArgumentError("The cash floor must be nonnegative."))
        return new(beta_f, gamma_f, return_f, floor_f, TerminalSpec(terminal_date))
    end
end

"""Number of independent standard-normal innovations in an income process."""
innovation_dimension(::IIDLognormalProcess) = 1
innovation_dimension(::LogAR1Process) = 1
innovation_dimension(::HSZProcess) = 2

"""Current income represented by an exogenous state."""
income(process::IIDLognormalProcess, ::IIDState) =
    exp(process.log_mean + process.log_std^2 / 2.0)
income(::LogAR1Process, state::LogAR1State) = exp(state.log_income)
income(::HSZProcess, state::HSZState) =
    exp(state.log_earnings) - exp(state.log_medical)

function _scalar_innovation(innovation)
    if innovation isa Real
        value = Float64(innovation)
    elseif innovation isa Tuple && length(innovation) == 1 && innovation[1] isa Real
        value = Float64(innovation[1])
    else
        throw(ArgumentError("This process requires one scalar standard-normal innovation."))
    end
    isfinite(value) || throw(ArgumentError("The innovation must be finite."))
    return value
end

function _pair_innovation(innovation)
    innovation isa Tuple && length(innovation) == 2 &&
        innovation[1] isa Real && innovation[2] isa Real || throw(ArgumentError(
        "The HSZ process requires a two-element innovation tuple."))
    pair = (Float64(innovation[1]), Float64(innovation[2]))
    all(isfinite, pair) || throw(ArgumentError("Both innovations must be finite."))
    return pair
end

function _next_income_state(process::IIDLognormalProcess, ::IIDState, innovation)
    epsilon = _scalar_innovation(innovation)
    return exp(process.log_mean + process.log_std * epsilon), IIDState()
end

function _next_income_state(process::LogAR1Process, state::LogAR1State, innovation)
    epsilon = _scalar_innovation(innovation)
    next_log = process.unconditional_log_mean +
        process.rho * (state.log_income - process.unconditional_log_mean) +
        process.log_std * epsilon
    next_state = LogAR1State(next_log)
    return income(process, next_state), next_state
end

function _next_income_state(process::HSZProcess, state::HSZState, innovation)
    epsilon_e, epsilon_m = _pair_innovation(innovation)
    next_log_e = process.unconditional_log_earnings +
        process.rho_earnings *
        (state.log_earnings - process.unconditional_log_earnings) +
        process.std_earnings * epsilon_e
    next_log_m = process.unconditional_log_medical +
        process.rho_medical *
        (state.log_medical - process.unconditional_log_medical) +
        process.std_medical * epsilon_m
    next_state = HSZState(next_log_e, next_log_m)
    return income(process, next_state), next_state
end

function _validate_observed(config::StochasticPTDConfig, observed::ObservedState)
    observed.date <= config.terminal.final_date || throw(ArgumentError(
        "The observed date is after the configured terminal date."))
    observed.cash >= config.cash_floor || throw(ArgumentError(
        "Observed post-transfer cash cannot be below the configured floor."))
    return nothing
end

"""
    true_transition(config, process, observed, consumption, innovation)

Apply the true law of motion after consumption is fixed. This is the single
authoritative implementation of the floor transition:
`z = (1+r)(x-c)+y_next`, `transfer=max(x_floor-z,0)`, and
`x_next=z+transfer`.
"""
function true_transition(config::StochasticPTDConfig,
    process::AbstractIncomeProcess, observed::ObservedState,
    consumption::Real, innovation)::TransitionResult
    _validate_observed(config, observed)
    observed.date < config.terminal.final_date || throw(ArgumentError(
        "There is no transition after the terminal decision."))
    consumption_f = Float64(consumption)
    isfinite(consumption_f) && 0.0 <= consumption_f <= observed.cash ||
        throw(ArgumentError("Consumption must be finite and lie in [0, cash]."))
    next_income, next_state = _next_income_state(
        process, observed.exogenous, innovation)
    z = (1.0 + config.return_rate) * (observed.cash - consumption_f) + next_income
    isfinite(z) || throw(ArgumentError("The pre-transfer next cash is not finite."))
    transfer = max(config.cash_floor - z, 0.0)
    return TransitionResult(z, transfer, max(z, config.cash_floor), next_state)
end

"""Construct the next observed state from a transition result."""
next_observed(observed::ObservedState, result::TransitionResult) =
    ObservedState(observed.date + 1, result.xnext, result.snext)

"""
Positive, normalized Gauss-Hermite quadrature for `D` independent standard
normal innovations. Nodes and weights are tuples, so the rule is deeply
immutable and safe to share across policy objects.
"""
struct GaussHermiteRule{D,N}
    nodes::NTuple{N,NTuple{D,Float64}}
    weights::NTuple{N,Float64}
    nodes_per_dimension::Int

    function GaussHermiteRule(nodes::NTuple{N,NTuple{D,Float64}},
        weights::NTuple{N,Float64}, nodes_per_dimension::Int,
        ::Val{:validated}) where {D,N}
        return new{D,N}(nodes, weights, nodes_per_dimension)
    end
end

function _validated_rule(nodes::NTuple{N,NTuple{D,Float64}},
    weights::NTuple{N,Float64}, nodes_per_dimension::Int) where {D,N}
    N > 0 || throw(ArgumentError("A quadrature rule must have nodes."))
    nodes_per_dimension >= 1 || throw(ArgumentError(
        "The node count per dimension must be positive."))
    all(node -> all(isfinite, node), nodes) || throw(ArgumentError(
        "Every quadrature node must be finite."))
    all(weight -> isfinite(weight) && weight > 0.0, weights) ||
        throw(ArgumentError("Every quadrature weight must be finite and positive."))
    isapprox(sum(weights), 1.0; rtol=8eps(Float64), atol=8eps(Float64)) ||
        throw(ArgumentError("Quadrature weights must sum to one."))
    return GaussHermiteRule(nodes, weights, nodes_per_dimension, Val(:validated))
end

"""
    gauss_hermite_rule(nodes_per_dimension; dimension=1)

Return positive normalized quadrature for one or two independent standard-normal
innovations. HSZ uses the two-dimensional tensor product.
"""
function gauss_hermite_rule(nodes_per_dimension::Integer; dimension::Integer=1)
    n = Int(nodes_per_dimension)
    d = Int(dimension)
    n >= 1 || throw(ArgumentError("The Gauss-Hermite node count must be positive."))
    d in (1, 2) || throw(ArgumentError(
        "The stochastic PTD core currently supports one or two shock dimensions."))
    raw_nodes, raw_weights = gausshermite(n)
    standard_nodes = sqrt(2.0) .* raw_nodes
    standard_weights = raw_weights ./ sqrt(pi)
    if d == 1
        nodes = Tuple((Float64(standard_nodes[i]),) for i in 1:n)
        weights_vector = standard_weights ./ sum(standard_weights)
        weights = Tuple(Float64(weights_vector[i]) for i in 1:n)
        return _validated_rule(nodes, weights, n)
    end

    nodes = Tuple((Float64(standard_nodes[i]), Float64(standard_nodes[j]))
        for i in 1:n for j in 1:n)
    weights_vector = [standard_weights[i] * standard_weights[j]
        for i in 1:n for j in 1:n]
    weights_vector ./= sum(weights_vector)
    weights = Tuple(Float64(weight) for weight in weights_vector)
    return _validated_rule(nodes, weights, n)
end

gauss_hermite_rule(process::AbstractIncomeProcess, nodes_per_dimension::Integer) =
    gauss_hermite_rule(nodes_per_dimension;
        dimension=innovation_dimension(process))

"""Integrate a scalar function of a standard-normal innovation tuple."""
function quadrature_expectation(f::F, rule::GaussHermiteRule) where {F}
    first_value = f(rule.nodes[1])
    result = rule.weights[1] * first_value
    for index in 2:length(rule.weights)
        result += rule.weights[index] * f(rule.nodes[index])
    end
    return result
end

quadrature_expectation(rule::GaussHermiteRule, f::F) where {F} =
    quadrature_expectation(f, rule)
