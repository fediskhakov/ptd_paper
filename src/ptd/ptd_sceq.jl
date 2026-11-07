"""
Certainty-equivalent stochastic policy-tree descendant (PTD-SCEQ) solver.

The solver constructs a deterministic conditional-mean path from the current
state, solves it with deterministic PTD, and returns only the current action.
"""

if !isdefined(@__MODULE__, :AbstractIncomeProcess)
    include("stochastic_model.jl")
end
if !isdefined(@__MODULE__, :DeterministicProblem)
    include("deterministic_solver.jl")
end

function _float_token(value::Float64)
    return string(reinterpret(UInt64, value); base=16, pad=16)
end

_state_token(::IIDState) = "iid"
_state_token(state::LogAR1State) = "ar1:" * _float_token(state.log_income)
_state_token(state::HSZState) = "hsz:" * _float_token(state.log_earnings) *
    ":" * _float_token(state.log_medical)

function _sha_fingerprint(parts...)
    payload = join(string.(parts), '\x1f')
    return bytes2hex(SHA.sha256(Vector{UInt8}(codeunits(payload))))
end

"""
An immutable, dated conditional-mean income path. The first entry is the current
date; in a deterministic `DeterministicProblem`, that first income is bookkeeping only because
the current observed cash already contains current income.
"""
struct NominalPath{S,N}
    dates::NTuple{N,Int}
    states::NTuple{N,S}
    incomes::NTuple{N,Float64}
    rule::Symbol
    fingerprint::String
end

function _conditional_variance(std::Float64, rho::Float64, horizon::Int)
    horizon == 0 && return 0.0
    return std^2 * sum(rho^(2k) for k in 0:horizon-1)
end

function _make_nominal_path(dates::NTuple{N,Int}, states::NTuple{N,S},
    incomes::NTuple{N,Float64}, process::AbstractIncomeProcess) where {S,N}
    all(isfinite, incomes) || throw(ArgumentError(
        "Every conditional-mean nominal income must be finite."))
    all(dates[index+1] == dates[index] + 1 for index in 1:N-1) ||
        throw(ArgumentError("Nominal-path dates must be consecutive."))
    process_token = _process_token(process)
    fingerprint = _sha_fingerprint("conditional_mean_income", process_token,
        join(dates, ','), join(_state_token.(states), ','),
        join(_float_token.(incomes), ','))
    return NominalPath{S,N}(dates, states, incomes,
        :conditional_mean_income, fingerprint)
end

_process_token(process::IIDLognormalProcess) = "iid:" *
    _float_token(process.log_mean) * ":" * _float_token(process.log_std)
_process_token(process::LogAR1Process) = "ar1:" * _float_token(process.rho) *
    ":" * _float_token(process.log_std) * ":" *
    _float_token(process.unconditional_log_mean)
_process_token(process::HSZProcess) = "hsz:" *
    join(_float_token.((process.rho_earnings, process.std_earnings,
        process.unconditional_log_earnings, process.rho_medical,
        process.std_medical, process.unconditional_log_medical)), ':')

"""
    nominal_path(config, process, observed)

Construct the full remaining deterministic conditional-mean income path. For
persistent processes, all conditional moments are propagated from the currently
observed state rather than reset to an unconditional state.
"""
function nominal_path(config::StochasticPTDConfig,
    process::IIDLognormalProcess, observed::ObservedState{IIDState})
    _validate_observed(config, observed)
    count = config.terminal.final_date - observed.date + 1
    dates = ntuple(index -> observed.date + index - 1, count)
    states = ntuple(_ -> IIDState(), count)
    mean_income = income(process, IIDState())
    incomes = ntuple(_ -> mean_income, count)
    return _make_nominal_path(dates, states, incomes, process)
end

function nominal_path(config::StochasticPTDConfig,
    process::LogAR1Process, observed::ObservedState{LogAR1State})
    _validate_observed(config, observed)
    count = config.terminal.final_date - observed.date + 1
    dates = ntuple(index -> observed.date + index - 1, count)
    mean_logs = ntuple(count) do index
        horizon = index - 1
        index == 1 ? observed.exogenous.log_income :
        process.unconditional_log_mean + process.rho^horizon *
        (observed.exogenous.log_income - process.unconditional_log_mean)
    end
    states = ntuple(index -> LogAR1State(mean_logs[index]), count)
    incomes = ntuple(count) do index
        horizon = index - 1
        variance = _conditional_variance(
            process.log_std, process.rho, horizon)
        exp(mean_logs[index] + variance / 2.0)
    end
    return _make_nominal_path(dates, states, incomes, process)
end

function nominal_path(config::StochasticPTDConfig,
    process::HSZProcess, observed::ObservedState{HSZState})
    _validate_observed(config, observed)
    count = config.terminal.final_date - observed.date + 1
    dates = ntuple(index -> observed.date + index - 1, count)
    mean_logs_e = ntuple(count) do index
        horizon = index - 1
        index == 1 ? observed.exogenous.log_earnings :
        process.unconditional_log_earnings + process.rho_earnings^horizon *
        (observed.exogenous.log_earnings - process.unconditional_log_earnings)
    end
    mean_logs_m = ntuple(count) do index
        horizon = index - 1
        index == 1 ? observed.exogenous.log_medical :
        process.unconditional_log_medical + process.rho_medical^horizon *
        (observed.exogenous.log_medical - process.unconditional_log_medical)
    end
    states = ntuple(index -> HSZState(mean_logs_e[index], mean_logs_m[index]), count)
    incomes = ntuple(count) do index
        horizon = index - 1
        variance_e = _conditional_variance(
            process.std_earnings, process.rho_earnings, horizon)
        variance_m = _conditional_variance(
            process.std_medical, process.rho_medical, horizon)
        exp(mean_logs_e[index] + variance_e / 2.0) -
            exp(mean_logs_m[index] + variance_m / 2.0)
    end
    return _make_nominal_path(dates, states, incomes, process)
end

function _config_token(config::StochasticPTDConfig)
    return join((_float_token(config.beta), _float_token(config.gamma),
        _float_token(config.return_rate), _float_token(config.cash_floor),
        string(config.terminal.final_date)), ':')
end

function _plan_fingerprint(agent::DeterministicProblem, path::NominalPath,
    config::StochasticPTDConfig)
    buffer = IOBuffer()
    print(buffer, _config_token(config), '|', path.fingerprint)
    for value in agent.incomes
        print(buffer, '|', _float_token(value))
    end
    for period in axes(agent.policy, 2), segment in agent.policy[1, period]
        print(buffer, '|', _float_token(segment.xStart), ',',
            _float_token(segment.xEnd), ',', _float_token(segment.b0), ',',
            _float_token(segment.b1))
    end
    return bytes2hex(SHA.sha256(take!(buffer)))
end

function _validate_solved_agent(agent::DeterministicProblem)
    size(agent.incomes, 1) == 1 || throw(ArgumentError(
        "A PTD plan must wrap exactly one nominal income path."))
    for period in 1:agent.tMax
        policy = agent.policy[1, period]
        isempty(policy) && throw(ArgumentError("A solved policy cannot be empty."))
        policy[1].xStart == 0.0 || throw(ArgumentError(
            "Every solved policy must start at zero cash."))
        isinf(policy[end].xEnd) || throw(ArgumentError(
            "Every solved policy must cover the unbounded upper domain."))
        all(!isnan(segment) for segment in policy) || throw(ArgumentError(
            "A solved policy cannot contain NaN segments."))
        all(policy[index].xEnd == policy[index+1].xStart
            for index in 1:length(policy)-1) || throw(ArgumentError(
            "Solved policy segments must have exact contiguous coverage."))
    end
    return nothing
end

"""
Validated read-only snapshot of a solved deterministic PTD plan. The underlying
`DeterministicProblem` is deliberately opaque, all public queries are pure, and every query
checks the stable policy fingerprint to detect external mutation.
"""
struct PTDPlan{S,N}
    _agent::DeterministicProblem
    path::NominalPath{S,N}
    config::StochasticPTDConfig
    fingerprint::String

    function PTDPlan(agent::DeterministicProblem, path::NominalPath{S,N},
        config::StochasticPTDConfig, fingerprint::String,
        ::Val{:validated}) where {S,N}
        return new{S,N}(agent, path, config, fingerprint)
    end
end

"""
Thread-safe memoization for complete immutable nominal-tail plans.

The cache is deliberately separate from `PTDPlan`: shared plans remain
read-only and carry their own integrity fingerprint, while cache counters and
storage are the only mutable state.
"""
mutable struct PTDPlanCache
    plans::Dict{String,PTDPlan}
    hits::Int
    misses::Int
    builds::Int
    verify_integrity_on_hit::Bool
    lock::ReentrantLock
end

PTDPlanCache(; verify_integrity_on_hit::Bool=true) = PTDPlanCache(
    Dict{String,PTDPlan}(), 0, 0, 0, verify_integrity_on_hit,
    ReentrantLock())

"""Stable request key based only on the configuration and nominal path."""
plan_request_fingerprint(config::StochasticPTDConfig, path::NominalPath) =
    _sha_fingerprint("ptd_plan_request", _config_token(config), path.fingerprint)

"""Return an immutable snapshot of cache size and counters."""
function plan_cache_statistics(cache::PTDPlanCache)
    return lock(cache.lock) do
        (; entries=length(cache.plans), hits=cache.hits,
            misses=cache.misses, builds=cache.builds)
    end
end

"""Reset hit/miss/build counters without discarding cached plans."""
function reset_plan_cache_statistics!(cache::PTDPlanCache)
    lock(cache.lock) do
        cache.hits = 0
        cache.misses = 0
        cache.builds = 0
    end
    return plan_cache_statistics(cache)
end

"""Discard cached plans and, by default, reset their audit counters."""
function clear_plan_cache!(cache::PTDPlanCache; reset_statistics::Bool=true)
    lock(cache.lock) do
        empty!(cache.plans)
        if reset_statistics
            cache.hits = 0
            cache.misses = 0
            cache.builds = 0
        end
    end
    return plan_cache_statistics(cache)
end

"""Recheck every cached plan and return the number of verified entries."""
function verify_plan_cache_integrity!(cache::PTDPlanCache)
    return lock(cache.lock) do
        for plan in values(cache.plans)
            _assert_plan_integrity(plan)
        end
        length(cache.plans)
    end
end

function Base.getproperty(plan::PTDPlan, name::Symbol)
    name in (:_agent, :agent) && throw(ArgumentError(
        "The deterministic DeterministicProblem inside PTDPlan is an opaque read-only snapshot."))
    return getfield(plan, name)
end

Base.propertynames(::PTDPlan, private::Bool=false) =
    private ? (:_agent, :path, :config, :fingerprint) :
    (:path, :config, :fingerprint)

_plan_agent(plan::PTDPlan) = getfield(plan, :_agent)

function _assert_plan_integrity(plan::PTDPlan)
    _plan_fingerprint(_plan_agent(plan), plan.path, plan.config) == plan.fingerprint ||
        throw(ErrorException("The read-only PTD plan snapshot was mutated."))
    return nothing
end

"""Solve and validate a deterministic PTD plan for an immutable nominal path."""
function build_ptd_plan(config::StochasticPTDConfig, path::NominalPath)
    first(path.dates) >= 1 || throw(ArgumentError(
        "A nominal plan must start at a positive date."))
    last(path.dates) == config.terminal.final_date || throw(ArgumentError(
        "The nominal path must end at the configured terminal date."))
    incomes = reshape(collect(path.incomes), 1, :)
    agent = DeterministicProblem(β=config.beta, γ=config.gamma,
        r=config.return_rate, xFloor=config.cash_floor, incomes=incomes)
    deterministic_ptd!(agent; threshold=2)
    _validate_solved_agent(agent)
    fingerprint = _plan_fingerprint(agent, path, config)
    return PTDPlan(agent, path, config, fingerprint, Val(:validated))
end

"""
Return the complete plan for a configuration/path request, building it once.

The solve is performed while holding the cache lock. This intentionally trades
parallel construction of identical tails for a strict one-miss/one-build audit
invariant. By default every hit rechecks the immutable plan's integrity
fingerprint. A policy-local cache owned by one contest worker may disable that
repeated SHA audit after construction; newly built plans are always checked.
"""
function cached_ptd_plan!(cache::PTDPlanCache,
    config::StochasticPTDConfig, path::NominalPath)

    key = plan_request_fingerprint(config, path)
    return lock(cache.lock) do
        if haskey(cache.plans, key)
            plan = cache.plans[key]
            cache.verify_integrity_on_hit && _assert_plan_integrity(plan)
            cache.hits += 1
            return plan
        end
        cache.misses += 1
        plan = build_ptd_plan(config, path)
        _assert_plan_integrity(plan)
        cache.plans[key] = plan
        cache.builds += 1
        return plan
    end
end

function _local_period(plan::PTDPlan, date::Integer)
    date_i = Int(date)
    first_date = first(plan.path.dates)
    last_date = last(plan.path.dates)
    first_date <= date_i <= last_date || throw(ArgumentError(
        "The requested date is outside this PTD plan."))
    return date_i - first_date + 1
end

function _plan_value_after_integrity_check(plan::PTDPlan, cash::Real,
    date::Integer)
    cash_f = Float64(cash)
    isfinite(cash_f) && cash_f >= 0.0 || throw(ArgumentError(
        "Cash supplied to a PTD plan must be finite and nonnegative."))
    agent = _plan_agent(plan)
    local_period = _local_period(plan, date)
    current_cash = cash_f
    result = 0.0
    for period in local_period:agent.tMax
        consumption = c_x(agent, current_cash; i=1, t=period)
        result += agent.β^(period - local_period) * u(agent, consumption)
        if period < agent.tMax
            assets = current_cash - consumption
            current_cash = xNext_a(agent, assets; i=1, t=period)
        end
    end
    return result
end

"""Pure lifetime-value query on a read-only PTD plan."""
function plan_value(plan::PTDPlan, cash::Real; date::Integer=first(plan.path.dates))
    _assert_plan_integrity(plan)
    return _plan_value_after_integrity_check(plan, cash, date)
end

"""Envelope marginal value `u'(c)` at a dated point on a PTD plan."""
function marginal_value(plan::PTDPlan, cash::Real;
    date::Integer=first(plan.path.dates))
    _assert_plan_integrity(plan)
    cash_f = Float64(cash)
    isfinite(cash_f) && cash_f > 0.0 || throw(ArgumentError(
        "Cash for a marginal-value query must be finite and positive."))
    agent = _plan_agent(plan)
    local_period = _local_period(plan, date)
    consumption = c_x(agent, cash_f; i=1, t=local_period)
    return mu(agent, consumption)
end

"""All segment boundaries, including zero and the unbounded upper endpoint."""
function segment_boundaries(plan::PTDPlan;
    date::Integer=first(plan.path.dates))
    _assert_plan_integrity(plan)
    local_period = _local_period(plan, date)
    policy = _plan_agent(plan).policy[1, local_period]
    return Tuple(vcat([segment.xStart for segment in policy], policy[end].xEnd))
end

"""Immutable result returned by a nonanticipative policy query.

`objective === nothing` denotes an action-only query, as used by the contest
evaluator when computing the objective would add unused timed work.
"""
struct ActionResult
    consumption::Float64
    saving::Float64
    objective::Union{Nothing,Float64}
    method::Symbol
    plan_fingerprint::String

    function ActionResult(consumption::Real, saving::Real,
        objective::Union{Nothing,Real},
        method::Symbol, plan_fingerprint::String)
        c = Float64(consumption)
        a = Float64(saving)
        v = isnothing(objective) ? nothing : Float64(objective)
        all(isfinite, (c, a)) && c >= 0.0 && a >= 0.0 || throw(ArgumentError(
            "An action must contain finite, nonnegative consumption and saving."))
        !isnothing(v) && isnan(v) && throw(ArgumentError(
            "An action objective cannot be NaN."))
        return new(c, a, v, method, plan_fingerprint)
    end
end

"""
Immutable evaluator record for one dated action on one realized path.

`observed` is the information available when `action` is chosen. `transition`
is the resulting true floor transition, or `nothing` at the terminal date.
`discounted_utility` is this row's contribution to realized lifetime utility,
not an expected value or a solver-owned continuation value.
"""
struct TrajectoryRow{S,T<:Union{Nothing,TransitionResult}}
    path::Int
    observed::ObservedState{S}
    action::ActionResult
    transition::T
    discounted_utility::Float64

    function TrajectoryRow(path::Integer, observed::ObservedState{S},
        action::ActionResult, transition::T,
        discounted_utility::Real) where
        {S,T<:Union{Nothing,TransitionResult}}
        path_i = Int(path)
        payoff = Float64(discounted_utility)
        path_i >= 1 || throw(ArgumentError(
            "A trajectory path index must be positive."))
        isnan(payoff) && throw(ArgumentError(
            "A trajectory discounted-utility contribution cannot be NaN."))
        tolerance = 64eps(max(observed.cash, 1.0))
        abs(action.saving - (observed.cash - action.consumption)) <= tolerance ||
            throw(ArgumentError(
                "A trajectory action must be feasible at its observed state."))
        if !isnothing(transition)
            all(isfinite, (transition.z, transition.transfer,
                transition.xnext)) || throw(ArgumentError(
                "A trajectory transition must contain finite cash values."))
            transition.transfer >= 0.0 && transition.xnext >= 0.0 ||
                throw(ArgumentError(
                    "A trajectory transition must have nonnegative transfer and cash."))
        end
        return new{S,T}(path_i, observed, action, transition, payoff)
    end
end

function _first_action_after_integrity_check(plan::PTDPlan,
    observed::ObservedState, method::Symbol;
    evaluate_objective::Bool=true)
    observed.date == first(plan.path.dates) || throw(ArgumentError(
        "The observed date must equal the first date of the PTD plan."))
    observed.exogenous == first(plan.path.states) || throw(ArgumentError(
        "The observed exogenous state must match the plan's conditioning state."))
    observed.cash >= plan.config.cash_floor || throw(ArgumentError(
        "Observed cash cannot lie below the configured floor."))
    agent = _plan_agent(plan)
    consumption = c_x(agent, observed.cash; i=1, t=1)
    objective = evaluate_objective ? _plan_value_after_integrity_check(
        plan, observed.cash, observed.date) : nothing
    return ActionResult(consumption, observed.cash - consumption,
        objective, method, plan.fingerprint)
end

"""Return the first action of a plan at its matching observed state."""
function first_action(plan::PTDPlan, observed::ObservedState;
    method::Symbol=:deterministic_ptd)
    _assert_plan_integrity(plan)
    return _first_action_after_integrity_check(plan, observed, method)
end

"""
    ptd_sceq(config, process, observed; cache=nothing,
             evaluate_objective=true)

PTD stochastic certainty-equivalent action. At every observed state this method
rebuilds the full remaining conditional-mean path, solves deterministic PTD, and
implements only its first action. Set `evaluate_objective=false` for an
action-only query when the deterministic lifetime value is not consumed.
"""
function ptd_sceq(config::StochasticPTDConfig,
    process::AbstractIncomeProcess, observed::ObservedState;
    cache::Union{Nothing,PTDPlanCache}=nothing,
    evaluate_objective::Bool=true)::ActionResult
    path = nominal_path(config, process, observed)
    plan = isnothing(cache) ? build_ptd_plan(config, path) :
        cached_ptd_plan!(cache, config, path)
    if !isnothing(cache) && !cache.verify_integrity_on_hit
        return _first_action_after_integrity_check(
            plan, observed, :ptd_sceq; evaluate_objective)
    end
    evaluate_objective && return first_action(
        plan, observed; method=:ptd_sceq)
    _assert_plan_integrity(plan)
    return _first_action_after_integrity_check(
        plan, observed, :ptd_sceq; evaluate_objective=false)
end
