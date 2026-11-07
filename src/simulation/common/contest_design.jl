using Random

"""Closed parameter ranges used by the outer robustness design."""
struct ParameterBounds
    beta::Tuple{Float64,Float64}
    gamma::Tuple{Float64,Float64}
    initial_cash::Tuple{Float64,Float64}

    function ParameterBounds(beta, gamma, initial_cash)
        beta_bounds = (Float64(beta[1]), Float64(beta[2]))
        gamma_bounds = (Float64(gamma[1]), Float64(gamma[2]))
        cash_bounds = (Float64(initial_cash[1]), Float64(initial_cash[2]))
        0.0 < beta_bounds[1] < beta_bounds[2] <= 1.0 ||
            throw(ArgumentError("beta bounds must satisfy 0 < lower < upper <= 1"))
        0.0 < gamma_bounds[1] < gamma_bounds[2] ||
            throw(ArgumentError("gamma bounds must be strictly positive and ordered"))
        0.0 <= cash_bounds[1] < cash_bounds[2] ||
            throw(ArgumentError("initial-cash bounds must be nonnegative and ordered"))
        return new(beta_bounds, gamma_bounds, cash_bounds)
    end
end

ParameterBounds(; beta=(0.90, 0.995), gamma=(0.5, 5.0),
                initial_cash=(3.0, 50.0)) =
    ParameterBounds(beta, gamma, initial_cash)

"""
Configuration for the nested IID policy-welfare comparison.

`n_contests` is the outer robustness sample and `n_paths` is the inner
Monte Carlo sample used to estimate expected discounted utility in each contest.
`parameter_seed` controls only the outer parameter design. `evaluation_seed`
controls disjoint per-contest streams for the two variance panels and is never
stored in a `ContestSpec` supplied to a policy constructor.

The defaults mirror the locked 100-by-100 production fallback. Callers may
override them for reduced tests and diagnostics.
"""
Base.@kwdef struct ContestDesignConfig
    n_contests::Int = 100
    n_paths::Int = 100
    horizon::Int = 50
    floor::Float64 = 3.0
    bounds::ParameterBounds = ParameterBounds()
    parameter_seed::UInt64 = 0x5054445f50415241
    evaluation_seed::UInt64 = 0x5054445f4556414c
end

"""
One outer policy configuration.

Evaluation-stream seeds are deliberately absent: policy construction may use a
`ContestSpec` without receiving the seed of any hidden evaluation path.
"""
struct ContestSpec
    id::Int
    beta::Float64
    gamma::Float64
    initial_cash::Float64
end

function _validate(config::ContestDesignConfig)
    config.n_contests > 0 || throw(ArgumentError("n_contests must be positive"))
    config.n_paths > 0 || throw(ArgumentError("n_paths must be positive"))
    config.horizon >= 2 || throw(ArgumentError("horizon must be at least two"))
    isfinite(config.floor) && config.floor >= 0.0 ||
        throw(ArgumentError("floor must be finite and nonnegative"))
    config.bounds.initial_cash[1] >= config.floor || throw(ArgumentError(
        "initial cash is post-transfer and its lower bound cannot be below the floor"))
    config.parameter_seed != config.evaluation_seed || throw(ArgumentError(
        "the parameter and hidden-path designs require separate root seeds"))
    UInt128(config.n_contests) * UInt128(2) <= UInt128(typemax(UInt64)) + 1 ||
        throw(ArgumentError("n_contests is too large for disjoint panel streams"))
    return config
end

"""Randomized one-dimensional Latin-hypercube coordinates in `(0,1)`."""
function _stratified_coordinates(rng::AbstractRNG, n::Int)
    coordinates = ((0:n-1) .+ rand(rng, n)) ./ n
    return coordinates[randperm(rng, n)]
end

_rescale(u::Float64, bounds::Tuple{Float64,Float64}) =
    bounds[1] + u * (bounds[2] - bounds[1])

"""
    build_contest_design(config=ContestDesignConfig())

Construct a reproducible randomized Latin-hypercube design. The three parameter
coordinates are independently permuted. Hidden-path seeds are derived separately
by `panel_seed` and therefore do not appear in the returned policy specifications.
"""
function build_contest_design(config::ContestDesignConfig=ContestDesignConfig())
    _validate(config)
    n = config.n_contests
    rng = Xoshiro(config.parameter_seed)
    beta_u = _stratified_coordinates(rng, n)
    gamma_u = _stratified_coordinates(rng, n)
    cash_u = _stratified_coordinates(rng, n)

    return [ContestSpec(
        i,
        _rescale(beta_u[i], config.bounds.beta),
        _rescale(gamma_u[i], config.bounds.gamma),
        _rescale(cash_u[i], config.bounds.initial_cash),
    ) for i in 1:n]
end

"""SplitMix64's bijective output permutation (used only to derive RNG seeds)."""
function _splitmix64(value::UInt64)
    value = xor(value, value >> 30) * UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27) * UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

function _panel_offset(panel::Symbol)
    panel === :baseline && return UInt64(0)
    panel === :high_variance && return UInt64(1)
    throw(ArgumentError("panel must be :baseline or :high_variance"))
end

"""
    panel_seed(spec, config, panel)

Derive an evaluator-only seed for one contest-panel pair. Consecutive even/odd
stream identifiers and the bijective SplitMix64 permutation guarantee that all
`2*n_contests` seeds are distinct, including across the two panels.
"""
function panel_seed(spec::ContestSpec, config::ContestDesignConfig,
                    panel::Symbol)
    _validate(config)
    1 <= spec.id <= config.n_contests || throw(ArgumentError(
        "contest id is outside the configured design"))
    stream = UInt64(2) * UInt64(spec.id - 1) + _panel_offset(panel)
    return _splitmix64(config.evaluation_seed + stream)
end

"""
    predraw_innovations(spec, config, panel)

Pre-draw the complete latent-standard-normal shock block for one contest before
any policy is evaluated. Rows are paths and columns are innovations for dates
`2:T`. The returned matrix belongs to the evaluator and must never be supplied to
a policy constructor or current-action method.
"""
function predraw_innovations(spec::ContestSpec,
                             config::ContestDesignConfig,
                             panel::Symbol)
    _validate(config)
    1 <= spec.id <= config.n_contests ||
        throw(ArgumentError("contest id is outside the configured design"))
    rng = Xoshiro(panel_seed(spec, config, panel))
    return randn(rng, config.n_paths, config.horizon - 1)
end

"""A failed original or replacement draw, retained as a contest outcome."""
struct ContestFailure
    contest_id::Int
    exception::String
    message::String
end

"""Contest outcomes and their complete sampling and failure audit.

`design` and `results` retain the original slot order. Failed entries of
`results` are `ContestFailure` objects. `audit`, `provenance`, and `plain_design`
are plain records deliberately kept suitable for authenticated serialization.
"""
struct ContestDesignResult{P,A,D}
    design::Vector{ContestSpec}
    provenance::P
    results::Vector{Any}
    audit::A
    plain_design::D
end

"""Finite replacement-budget exhaustion, carrying every outcome and failure."""
struct ContestCompletionError{R<:ContestDesignResult} <: Exception
    result::R
end

function Base.showerror(io::IO, error::ContestCompletionError)
    audit = error.result.audit
    print(io, "Contest replacement budget exhausted with ",
        audit.failed_contests, " failed slots; the original and replacement ",
        "failures are retained in error.result.audit.")
end

# A tail certification failure invalidates the experiment itself, even when
# ordinary comparator failures are retained as outcomes.
_has_uncertified_tail(error) = error isa FusionTailError ||
    (error isa CompositeException && any(_has_uncertified_tail, error.exceptions)) ||
    (error isa TaskFailedException && any(item -> _has_uncertified_tail(item.exception),
        Base.current_exceptions(error.task)))

"""
    complete_contest_design(config; replacement_seed, attempt, scheduler, workers,
                            audit_schema, failure_policy=:retain,
                            max_replacement_rounds=100)

Evaluate the fixed Latin-hypercube design and retain every failed configuration
as a `ContestFailure`. The default performs no replacement: solver robustness
outcomes remain attached to the original parameter sample.

Explicit `failure_policy=:replace` enables the historical conditional design,
using independent deterministic replacement streams and at most the declared
number of replacement rounds. Exhaustion raises `ContestCompletionError` with
the complete partial result. All original failures remain in its audit.

`attempt(spec)` returns one complete in-memory contest result. An exception
other than `InterruptException` or an uncertified deterministic tail, or a literal
`false`, records failure. An uncertified tail aborts the experiment.
Successful slots are never rerun; hidden shocks remain attached to stable IDs.
"""
function complete_contest_design(config::ContestDesignConfig;
    replacement_seed::UInt64,
    attempt,
    scheduler,
    workers::Integer,
    audit_schema::AbstractString,
    failure_policy::Symbol=:retain,
    max_replacement_rounds::Integer=100)

    _validate(config)
    replacement_seed != config.parameter_seed || throw(ArgumentError(
        "initial and replacement parameter streams require distinct seeds"))
    replacement_seed != config.evaluation_seed || throw(ArgumentError(
        "replacement parameters and evaluator shocks require distinct seeds"))
    worker_count = Int(workers)
    worker_count > 0 || throw(ArgumentError("workers must be positive"))
    isempty(audit_schema) && throw(ArgumentError("audit_schema cannot be empty"))
    failure_policy in (:retain, :replace) || throw(ArgumentError(
        "failure_policy must be :retain or :replace"))
    max_replacement_rounds >= 0 || throw(ArgumentError(
        "max_replacement_rounds must be nonnegative"))

    initial = build_contest_design(config)
    design = copy(initial)
    results = Vector{Any}(undef, length(design))
    provenance = [(sample_source=:initial, sample_round=0,
        sample_draw_id=spec.id, sample_parameter_seed=config.parameter_seed)
        for spec in initial]
    pending_slots = collect(eachindex(design))
    rounds = NamedTuple[]
    round = 0
    failed_slots = Int[]

    while !isempty(pending_slots)
        outcomes = Vector{Any}(undef, length(pending_slots))
        failures = Vector{Any}(undef, length(pending_slots))
        fill!(outcomes, nothing)
        fill!(failures, nothing)
        scheduler(collect(enumerate(pending_slots)), worker_count) do item
            position, slot = item
            try
                outcome = attempt(design[slot])
                outcome === false && throw(ErrorException(
                    "contest completion callback returned false"))
                outcomes[position] = outcome
            catch error
                (error isa InterruptException || _has_uncertified_tail(error)) && rethrow()
                failures[position] = (
                    exception=string(nameof(typeof(error))),
                    message=sprint(showerror, error),
                )
            end
        end

        failed_positions = findall(!isnothing, failures)
        failed_slots = pending_slots[failed_positions]
        for (position, slot) in zip(eachindex(pending_slots), pending_slots)
            failure = failures[position]
            results[slot] = isnothing(failure) ? outcomes[position] :
                ContestFailure(design[slot].id, failure.exception, failure.message)
        end
        seed = round == 0 ? config.parameter_seed :
            provenance[first(pending_slots)].sample_parameter_seed
        failure_records = [(slot=failed_slots[index],
            contest_id=design[failed_slots[index]].id,
            failures[failed_positions[index]]...)
            for index in eachindex(failed_slots)]
        push!(rounds, (round, parameter_seed=seed,
            sampled_slots=Tuple(pending_slots),
            failed_slots=Tuple(failed_slots), failures=failure_records))
        isempty(failed_slots) && break
        failure_policy === :retain && break
        round >= max_replacement_rounds && break

        round == typemax(Int) && throw(OverflowError(
            "contest replacement round counter overflowed"))
        round += 1
        round_seed = _splitmix64(replacement_seed + UInt64(round - 1))
        replacement_config = ContestDesignConfig(
            n_contests=length(failed_slots),
            n_paths=config.n_paths,
            horizon=config.horizon,
            floor=config.floor,
            bounds=config.bounds,
            parameter_seed=round_seed,
            evaluation_seed=config.evaluation_seed,
        )
        candidates = build_contest_design(replacement_config)
        for (draw_id, slot) in enumerate(failed_slots)
            candidate = candidates[draw_id]
            design[slot] = ContestSpec(slot, candidate.beta, candidate.gamma,
                candidate.initial_cash)
            provenance[slot] = (sample_source=:replacement,
                sample_round=round, sample_draw_id=draw_id,
                sample_parameter_seed=round_seed)
        end
        pending_slots = failed_slots
    end

    plain_design = [(id=spec.id, beta=spec.beta, gamma=spec.gamma,
        initial_cash=spec.initial_cash, provenance[index]...)
        for (index, spec) in enumerate(design)]
    audit = (
        schema=String(audit_schema),
        rule=failure_policy === :retain ? :retain_original_configurations_and_failures :
            :replace_each_failed_configuration_one_for_one_until_complete,
        failure_policy,
        target_contests=config.n_contests,
        completed_contests=config.n_contests - length(failed_slots),
        failed_contests=length(failed_slots),
        failed_slots=Tuple(failed_slots),
        original_failed_contests=length(first(rounds).failed_slots),
        total_failed_attempts=sum(length(row.failed_slots) for row in rounds),
        conditional_on_success=!isempty(first(rounds).failed_slots),
        max_replacement_rounds=Int(max_replacement_rounds),
        initial_parameter_seed=config.parameter_seed,
        replacement_parameter_seed=replacement_seed,
        rounds,
        replacement_count=count(row ->
            row.sample_source === :replacement, provenance),
        final_design_fingerprint=stable_fingerprint(plain_design),
    )
    result = ContestDesignResult(design, provenance, results, audit, plain_design)
    failure_policy === :replace && !isempty(failed_slots) &&
        throw(ContestCompletionError(result))
    return result
end
