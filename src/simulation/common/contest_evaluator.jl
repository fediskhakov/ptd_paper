"""One method's pathwise discounted-utility evaluation on hidden shocks."""
struct PolicyPathEvaluation
    method::Symbol
    utility::Vector{Float64}
    floor_hits::Vector{Int}
    total_transfers::Vector{Float64}
end

"""Common-shock pathwise results for every method in one outer contest."""
struct ContestPathEvaluation
    method_ids::Vector{Symbol}
    utility::Matrix{Float64}
    floor_hits::Matrix{Int}
    total_transfers::Matrix{Float64}
end

"""
Persistent same-state/action audit for batched policy evaluation.

Construct one `PolicyActionAudit()` per policy and outer contest, then pass it
to every batch through `action_audit`. The first valid evaluation binds the
audit to that exact callable policy, registered method, configuration, income
process, and initial-state type. Reusing it for another scope is rejected.

The audit is owned by one contest and one thread; separate concurrently evaluated
policies must own separate audits.
"""
mutable struct PolicyActionAudit
    bound::Bool
    policy::Any
    method::Symbol
    config_token::String
    process_token::String
    initial_state_type::Any
    initial_state_token::String
    state_actions::Dict{Any,Tuple{Float64,Float64}}
end

PolicyActionAudit() = PolicyActionAudit(false, nothing, :unbound, "", "",
    nothing, "", Dict{Any,Tuple{Float64,Float64}}())

"""Method matrix fixed for the paper's stochastic model classes."""
const IID_CONTEST_METHODS = (:ptd_sceq, :ptd_osa, :dcegm)

"""Return the paper methods admissible for an income-process type."""
allowed_stochastic_methods(::IIDLognormalProcess) = IID_CONTEST_METHODS
allowed_stochastic_methods(::LogAR1Process) =
    (:ptd_sceq, :ptd_osa, :g2egm)
allowed_stochastic_methods(::HSZProcess) =
    (:ptd_sceq, :ptd_osa, :g2egm)

function _crra_utility(consumption::Float64, gamma::Float64)
    consumption >= 0.0 || throw(ArgumentError(
        "CRRA evaluation requires nonnegative consumption."))
    if consumption == 0.0
        return gamma < 1.0 ? -1.0 / (1.0 - gamma) : -Inf
    end
    return gamma == 1.0 ? log(consumption) :
           (consumption^(1.0 - gamma) - 1.0) / (1.0 - gamma)
end

_state_matches(::IIDLognormalProcess, observed::ObservedState) =
    observed.exogenous isa IIDState
_state_matches(::LogAR1Process, observed::ObservedState) =
    observed.exogenous isa LogAR1State
_state_matches(::HSZProcess, observed::ObservedState) =
    observed.exogenous isa HSZState

function _validate_shock_array(process::AbstractIncomeProcess,
    innovations::AbstractArray, n_steps::Int)
    eltype(innovations) <: Real || throw(ArgumentError(
        "Hidden innovations must have a real element type."))
    dimension = innovation_dimension(process)
    if dimension == 1
        ndims(innovations) == 2 || throw(DimensionMismatch(
            "A scalar process requires a paths-by-dates innovation matrix."))
        size(innovations, 2) == n_steps || throw(DimensionMismatch(
            "The innovation matrix must have one column per nonterminal transition."))
    else
        ndims(innovations) == 3 || throw(DimensionMismatch(
            "A multivariate process requires a paths-by-dates-by-dimensions array."))
        size(innovations, 2) == n_steps || throw(DimensionMismatch(
            "The innovation array must have one date slice per nonterminal transition."))
        size(innovations, 3) == dimension || throw(DimensionMismatch(
            "The innovation array's final dimension does not match the process."))
    end
    size(innovations, 1) > 0 || throw(ArgumentError(
        "At least one hidden evaluation path is required."))
    all(isfinite, innovations) || throw(ArgumentError(
        "Every hidden innovation must be finite."))
    return nothing
end


_action_tolerance(cash::Float64) = 64eps(max(cash, 1.0))

function _validate_action(result, method::Symbol,
    config::StochasticPTDConfig, process::AbstractIncomeProcess,
    observed::ObservedState, terminal::Bool)
    result isa ActionResult || throw(ArgumentError(
        "A contest policy must return ActionResult."))
    method in allowed_stochastic_methods(process) || throw(ArgumentError(
        "Method $method is incompatible with $(typeof(process))."))
    result.method == method || throw(ArgumentError(
        "Policy method $(result.method) does not match registered method $method."))

    consumption = result.consumption
    0.0 <= consumption <= observed.cash || throw(ArgumentError(
        "A contest action must have consumption in [0, current cash]."))
    tolerance = _action_tolerance(observed.cash)
    abs(result.saving - (observed.cash - consumption)) <= tolerance ||
        throw(ArgumentError(
            "A contest action's reported saving must equal cash minus consumption."))
    if terminal
        abs(consumption - observed.cash) <= tolerance || throw(ArgumentError(
            "The terminal contest action must consume all remaining cash."))
        result.saving <= tolerance || throw(ArgumentError(
            "The terminal contest action must report zero saving."))
    end
    return consumption
end

function _assert_same_state_action!(seen::Dict, observed::ObservedState,
    result::ActionResult)
    action = (result.consumption, result.saving)
    if haskey(seen, observed)
        seen[observed] == action || throw(ArgumentError(
            "A contest policy returned different actions at the same observed state; " *
            "contest policies must be deterministic and nonanticipative."))
    else
        seen[observed] = action
    end
    return nothing
end

function _bind_policy_action_audit!(audit::PolicyActionAudit, action,
    method::Symbol, config::StochasticPTDConfig,
    process::AbstractIncomeProcess, initial::ObservedState)
    config_token = _config_token(config)
    process_token = _process_token(process)
    initial_state_type = typeof(initial)
    initial_state_token = join((string(initial.date),
        _float_token(initial.cash), _state_token(initial.exogenous)), '|')
    if !audit.bound
        audit.bound = true
        audit.policy = action
        audit.method = method
        audit.config_token = config_token
        audit.process_token = process_token
        audit.initial_state_type = initial_state_type
        audit.initial_state_token = initial_state_token
        return audit.state_actions
    end

    audit.policy === action || throw(ArgumentError(
        "A PolicyActionAudit cannot be reused for a different policy object."))
    audit.method == method || throw(ArgumentError(
        "A PolicyActionAudit cannot be reused for a different method."))
    audit.config_token == config_token || throw(ArgumentError(
        "A PolicyActionAudit cannot be reused for a different configuration."))
    audit.process_token == process_token || throw(ArgumentError(
        "A PolicyActionAudit cannot be reused for a different income process."))
    audit.initial_state_type === initial_state_type || throw(ArgumentError(
        "A PolicyActionAudit cannot be reused with an incompatible initial-state type."))
    audit.initial_state_token == initial_state_token || throw(ArgumentError(
        "A PolicyActionAudit cannot be reused for a different initial observed state."))
    return audit.state_actions
end

_hidden_innovation(innovations::AbstractMatrix, path::Int, step::Int, ::Val{1}) =
    Float64(innovations[path, step])

function _hidden_innovation(innovations::AbstractArray{<:Real,3},
    path::Int, step::Int, ::Val{D}) where {D}
    return ntuple(index -> Float64(innovations[path, step, index]), D)
end

"""
    evaluate_policy_paths(action, method, config, process, initial, innovations;
                          action_audit=nothing, trajectory_rows=nothing,
                          trajectory_path_offset=0)

Evaluate one already-constructed policy on a complete block of hidden shocks.
`action` is called with the current `ObservedState` and with nothing else: it
never receives an RNG, the innovation array, a path index, or future shocks.
Contest policies must be deterministic functions of that observed state; the
evaluator rejects different actions at an exactly repeated state. By default,
that check is local to this call. Supplying one `PolicyActionAudit()` as
`action_audit` persists the same-state/action map over successive path batches.
The returned utility vector contains one realized discounted lifetime payoff
per row; its mean is the within-contest expected-utility estimator. Supplying a
mutable `trajectory_rows` collection records immutable `TrajectoryRow`s for
diagnostics without changing the aggregate result. `trajectory_path_offset`
keeps path identifiers unique when explicitly tracing successive batches.
"""
function evaluate_policy_paths(action::F, method::Symbol,
    config::StochasticPTDConfig, process::AbstractIncomeProcess,
    initial::ObservedState, innovations::AbstractArray;
    action_audit::Union{Nothing,PolicyActionAudit}=nothing,
    trajectory_rows::Union{Nothing,AbstractVector}=nothing,
    trajectory_path_offset::Integer=0) where {F}
    _validate_observed(config, initial)
    _state_matches(process, initial) || throw(ArgumentError(
        "The initial exogenous state is incompatible with $(typeof(process))."))
    method in allowed_stochastic_methods(process) || throw(ArgumentError(
        "Method $method is incompatible with $(typeof(process))."))
    n_steps = config.terminal.final_date - initial.date
    _validate_shock_array(process, innovations, n_steps)
    n_paths = size(innovations, 1)
    path_offset = Int(trajectory_path_offset)
    path_offset >= 0 || throw(ArgumentError(
        "trajectory_path_offset must be nonnegative"))
    utility = zeros(Float64, n_paths)
    floor_hits = zeros(Int, n_paths)
    total_transfers = zeros(Float64, n_paths)
    dimension = Val(innovation_dimension(process))
    seen_actions = isnothing(action_audit) ?
        Dict{typeof(initial),Tuple{Float64,Float64}}() :
        _bind_policy_action_audit!(
            action_audit, action, method, config, process, initial)

    for path in 1:n_paths
        observed = initial
        discount = 1.0
        for step in 0:n_steps
            result = action(observed)
            consumption = _validate_action(result, method, config, process,
                observed, step == n_steps)
            _assert_same_state_action!(seen_actions, observed, result)
            discounted_utility =
                discount * _crra_utility(consumption, config.gamma)
            utility[path] += discounted_utility

            transition = if step == n_steps
                nothing
            else
                innovation = _hidden_innovation(
                    innovations, path, step + 1, dimension)
                true_transition(
                    config, process, observed, consumption, innovation)
            end
            if !isnothing(trajectory_rows)
                push!(trajectory_rows, TrajectoryRow(
                    path_offset + path, observed, result, transition,
                    discounted_utility))
            end
            isnothing(transition) && continue
            floor_hits[path] += transition.transfer > 0.0
            total_transfers[path] += transition.transfer
            observed = next_observed(observed, transition)
            discount *= config.beta
        end
    end

    return PolicyPathEvaluation(
        method, utility, floor_hits, total_transfers)
end

"""
    evaluate_contest_paths(actions, config, process, initial, innovations)

Run every named policy on the same evaluator-owned hidden innovation block.
`actions` is a named tuple such as
`(ptd_sceq=sceq_action, ptd_osa=oa_action, dcegm=dcegm_action)`.
Policies must be constructed before this function is called. This entry point is
the paper's nested welfare contest and therefore accepts IID income and exactly
the three registered IID competitors. AR(1) and HSZ path checks use
`evaluate_policy_paths` and are not part of the nested contest.
"""
function evaluate_contest_paths(actions::NamedTuple,
    config::StochasticPTDConfig, process::AbstractIncomeProcess,
    initial::ObservedState, innovations::AbstractArray)
    process isa IIDLognormalProcess || throw(ArgumentError(
        "The nested welfare contest is IID-only."))
    supplied = Tuple(keys(actions))
    length(supplied) == length(IID_CONTEST_METHODS) &&
        all(method -> method in supplied, IID_CONTEST_METHODS) ||
        throw(ArgumentError(
            "The IID contest requires exactly $(IID_CONTEST_METHODS)."))
    method_ids = collect(IID_CONTEST_METHODS)
    n_paths = size(innovations, 1)
    utilities = Matrix{Float64}(undef, n_paths, length(method_ids))
    floor_hits = Matrix{Int}(undef, n_paths, length(method_ids))
    transfers = Matrix{Float64}(undef, n_paths, length(method_ids))

    for (column, method) in enumerate(method_ids)
        evaluation = evaluate_policy_paths(getproperty(actions, method), method,
            config, process, initial, innovations)
        utilities[:, column] = evaluation.utility
        floor_hits[:, column] = evaluation.floor_hits
        transfers[:, column] = evaluation.total_transfers
    end

    return ContestPathEvaluation(
        method_ids, utilities, floor_hits, transfers)
end
