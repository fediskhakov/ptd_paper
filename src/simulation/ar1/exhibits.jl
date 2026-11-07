using Dates
using JLD2
using LinearAlgebra
using Random
using SHA
using Statistics
using DrWatson: datadir

import ..Ptd: AR1G2ContestGridConfiguration,
              _temporary_jld_path,
              _validate_data_output,
              LogAR1Process,
              LogAR1State,
              PTDOSAOptions,
              ObservedState,
              PolicyActionAudit,
              PolicyPathEvaluation,
              StochasticPTDConfig,
              _plain_ar1_g2_grid_configuration,
              ar1_contest_diagnostics,
              ar1_contest_quadrature_audit,
              build_ar1_contest_policy,
              candidate_ar1_g2_grid_configuration,
              evaluate_policy_paths,
              gauss_hermite_rule,
              reset_ar1_contest_diagnostics!,
              validate_ar1_contest_policy_alignment
using ..Ptd: paired_comparison
using ..IID: consumption_equivalent_bps
using ..G2EGM: AR1G2EGMOptions,
                   G2EGM_REFERENCE_COMMIT,
                   G2EGM_SOLVER_SCHEMA
using ..Ptd: PROJECT_ROOT, simulation_metadata, stable_fingerprint

export AR1_CONTEST_EXHIBIT_SCHEMA,
       AR1_CONTEST_EXHIBIT_AUTHENTICATION_SCHEMA,
       AR1_CONTEST_EXHIBIT_METHODS,
       AR1ContestExhibitConfig,
       production_ar1_contest_exhibit_config,
       validate_ar1_contest_exhibit_config,
       scientific_configuration,
       scientific_fingerprint,
       default_ar1_contest_exhibit_artifact_path,
       run_ar1_contest_exhibits,
       build_ar1_contest_exhibit_artifact,
       validate_ar1_contest_exhibit_artifact,
       save_ar1_contest_exhibit_artifact,
       load_ar1_contest_exhibit_artifact,
       warm_ar1_contest_exhibits!

const AR1_CONTEST_EXHIBIT_SCHEMA =
    "ptd-ar1-contest-exhibit-v2"
const AR1_CONTEST_EXHIBIT_AUTHENTICATION_SCHEMA =
    "ptd-ar1-contest-exhibit-authentication-v2"
const AR1_CONTEST_EXHIBIT_METHODS = (:ptd_sceq, :ptd_osa, :g2egm)
const AR1_CONTEST_EXHIBIT_LABELS = (
    ptd_sceq="PTD-SCEQ",
    ptd_osa="PTD-OSA",
    g2egm="G2EGM",
)
const AR1_CONTEST_EXHIBIT_SEED = UInt64(0x5054445f41523153)
const _EXHIBIT_UTC_FORMAT = dateformat"yyyy-mm-ddTHH:MM:SS.sss"

const DEFAULT_AR1_CONTEST_G2EGM_OPTIONS = let options = AR1G2EGMOptions()
    (bounding_box_padding=options.bounding_box_padding,
     minimum_barycentric_weight=options.minimum_barycentric_weight,
     hole_search_radius=options.hole_search_radius,
     value_atol=options.value_atol,
     value_rtol=options.value_rtol,
     determinant_tolerance=options.determinant_tolerance,
     reject_mixed_floor_regimes=options.reject_mixed_floor_regimes,
     enforce_monotone_saving=options.enforce_monotone_saving,
     monotonicity_atol=options.monotonicity_atol,
     monotonicity_rtol=options.monotonicity_rtol,
     direct_iterations=options.direct_iterations)
end

"""Frozen inputs for the paper's balanced scalar-AR(1) diagnostic."""
Base.@kwdef struct AR1ContestExhibitConfig
    beta::Float64 = 0.98
    gamma::Float64 = 1.0
    return_rate::Float64 = 0.10
    cash_floor::Float64 = 3.0
    horizon::Int = 50
    rho::Float64 = 0.9
    sigma::Float64 = 0.1
    unconditional_log_mean::Float64 = 0.0
    initial_income::Float64 = 1.0
    initial_cash_support::Vector{Float64} =
        collect(range(5.5, 26.5; length=5))
    n_paths::Int = 1_000
    trace_paths::Int = 100
    evaluation_seed::UInt64 = AR1_CONTEST_EXHIBIT_SEED
    confidence_level::Float64 = 0.95
    quadrature_nodes_per_dimension::Int = 5
    workers::Int = min(n_paths, Threads.nthreads())
    surface_date::Int = 1
    surface_cash_grid::Vector{Float64} =
        collect(range(3.0, 50.0; length=81))
    surface_log_income_grid::Vector{Float64} =
        collect(range(-0.6, 0.6; length=41))
    g2egm_grids::AR1G2ContestGridConfiguration =
        candidate_ar1_g2_grid_configuration()
    g2egm_options::NamedTuple = DEFAULT_AR1_CONTEST_G2EGM_OPTIONS
    ptd_osa_options::PTDOSAOptions = PTDOSAOptions()
end

"""Return the locked configuration for the paper's AR(1) exhibits."""
production_ar1_contest_exhibit_config() = AR1ContestExhibitConfig()

"""Validate an AR(1) exhibit configuration and return it unchanged."""
function validate_ar1_contest_exhibit_config(config::AR1ContestExhibitConfig)
    all(isfinite, (config.beta, config.gamma, config.return_rate,
        config.cash_floor, config.rho, config.sigma,
        config.unconditional_log_mean, config.initial_income,
        config.confidence_level)) && all(isfinite, config.initial_cash_support) ||
        throw(ArgumentError("contest-exhibit scalar inputs must be finite"))
    0.0 < config.beta <= 1.0 || throw(ArgumentError(
        "beta must lie in (0,1]"))
    config.gamma > 0.0 || throw(ArgumentError("gamma must be positive"))
    config.return_rate > -1.0 || throw(ArgumentError(
        "the gross return must be positive"))
    config.cash_floor >= 0.0 || throw(ArgumentError(
        "the cash floor must be nonnegative"))
    config.horizon >= 2 || throw(ArgumentError(
        "the horizon must contain at least two dates"))
    abs(config.rho) < 1.0 || throw(ArgumentError(
        "rho must have absolute value below one"))
    config.sigma > 0.0 || throw(ArgumentError(
        "the innovation standard deviation must be positive"))
    config.unconditional_log_mean == 0.0 || throw(ArgumentError(
        "the G2EGM adapter requires zero unconditional log mean"))
    config.initial_income > 0.0 || throw(ArgumentError(
        "initial income must be positive"))
    length(config.initial_cash_support) >= 1 &&
        all(diff(config.initial_cash_support) .> 0.0) || throw(ArgumentError(
        "initial cash support must be nonempty and strictly increasing"))
    first(config.initial_cash_support) >= config.cash_floor ||
        throw(ArgumentError("initial cash support cannot lie below the floor"))
    config.n_paths >= 2 || throw(ArgumentError(
        "at least two paths are required for paired uncertainty"))
    1 <= config.trace_paths <= config.n_paths || throw(ArgumentError(
        "trace_paths must lie in 1:n_paths"))
    config.n_paths % length(config.initial_cash_support) == 0 ||
        throw(ArgumentError(
            "n_paths must be divisible by the initial cash support size"))
    config.trace_paths % length(config.initial_cash_support) == 0 ||
        throw(ArgumentError(
            "trace_paths must be divisible by the initial cash support size"))
    0.0 < config.confidence_level < 1.0 || throw(ArgumentError(
        "confidence_level must lie in (0,1)"))
    config.quadrature_nodes_per_dimension == 5 || throw(ArgumentError(
        "the diagnostic requires the shared scalar Q5 rule"))
    1 <= config.workers <= config.n_paths || throw(ArgumentError(
        "workers must lie in 1:n_paths"))
    1 <= config.surface_date < config.horizon || throw(ArgumentError(
        "the policy surface must be nonterminal"))
    for (grid, name) in ((config.surface_cash_grid, "cash"),
                         (config.surface_log_income_grid, "log-income"))
        length(grid) >= 2 && all(isfinite, grid) &&
            all(diff(grid) .> 0.0) || throw(ArgumentError(
                "the $name surface grid must be finite and increasing"))
    end
    first(config.surface_cash_grid) >= config.cash_floor ||
        throw(ArgumentError("the policy surface begins below the cash floor"))
    first(config.surface_cash_grid) <= first(config.initial_cash_support) &&
        last(config.surface_cash_grid) >= last(config.initial_cash_support) ||
        throw(ArgumentError(
            "the policy surface must contain the initial cash support"))
    last(config.surface_cash_grid) <= last(config.g2egm_grids.asset_grid) ||
        throw(ArgumentError("the policy surface exceeds the G2EGM policy domain"))
    first(config.surface_log_income_grid) >=
        first(config.g2egm_grids.common_log_income_grid) &&
        last(config.surface_log_income_grid) <=
        last(config.g2egm_grids.common_log_income_grid) ||
        throw(ArgumentError(
            "the log-income surface exceeds the G2EGM policy domain"))
    return config
end

_paths_per_initial_cash(config::AR1ContestExhibitConfig) =
    div(config.n_paths, length(config.initial_cash_support))

_trace_paths_per_initial_cash(config::AR1ContestExhibitConfig) =
    div(config.trace_paths, length(config.initial_cash_support))

function _initial_cash_assignments(config::AR1ContestExhibitConfig)
    per_cash = _paths_per_initial_cash(config)
    return reduce(vcat,
        (fill(cash, per_cash) for cash in config.initial_cash_support);
        init=Float64[])
end

function _trace_path_ids(config::AR1ContestExhibitConfig)
    welfare_per_cash = _paths_per_initial_cash(config)
    trace_per_cash = _trace_paths_per_initial_cash(config)
    return reduce(vcat,
        (((index - 1) * welfare_per_cash + 1):
         ((index - 1) * welfare_per_cash + trace_per_cash)
         for index in eachindex(config.initial_cash_support)); init=Int[])
end

function _plain_ptd_osa_options(options::PTDOSAOptions)
    return (tie_atol=options.tie_atol,
        tie_rtol=options.tie_rtol,
        certification_atol=options.certification_atol,
        direct_iterations=options.direct_iterations)
end

function scientific_configuration(config::AR1ContestExhibitConfig)
    validate_ar1_contest_exhibit_config(config)
    return (;
        schema="ptd-ar1-contest-exhibit-scientific-v2",
        role=:balanced_initial_wealth_paired_diagnostic,
        broad_cross_calibration_evidence=false,
        methods=collect(AR1_CONTEST_EXHIBIT_METHODS),
        method_labels=AR1_CONTEST_EXHIBIT_LABELS,
        model=(;
            beta=config.beta,
            gamma=config.gamma,
            return_rate=config.return_rate,
            cash_floor=config.cash_floor,
            horizon=config.horizon,
            rho=config.rho,
            innovation_sigma=config.sigma,
            unconditional_log_mean=config.unconditional_log_mean,
            initial_income=config.initial_income,
            initial_cash_support=copy(config.initial_cash_support),
            initial_cash_weights=fill(
                inv(length(config.initial_cash_support)),
                length(config.initial_cash_support))),
        evaluation=(;
            n_paths=config.n_paths,
            paths_per_initial_cash=_paths_per_initial_cash(config),
            initial_cash_assignment=:balanced_contiguous_blocks,
            initial_cash_assignments_fingerprint=
                stable_fingerprint(_initial_cash_assignments(config)),
            seed=config.evaluation_seed,
            confidence_level=config.confidence_level,
            common_hidden_paths=true,
            innovations_revealed_after_action=true,
            workers=config.workers),
        quadrature=(;
            nodes_per_dimension=config.quadrature_nodes_per_dimension,
            ptd_sceq_uses_quadrature=false,
            ptd_osa_and_g2egm_share_exact_q5=true),
        g2egm=(;
            implementation=:pure_julia_problem_specific_g2egm,
            solver_schema=G2EGM_SOLVER_SCHEMA,
            upstream_commit=G2EGM_REFERENCE_COMMIT,
            grid=_plain_ar1_g2_grid_configuration(config.g2egm_grids),
            options=config.g2egm_options,
            solved_once_and_reused=true),
        ptd_osa_options=_plain_ptd_osa_options(config.ptd_osa_options),
        surface=(;
            date=config.surface_date,
            cash_grid=copy(config.surface_cash_grid),
            log_income_grid=copy(config.surface_log_income_grid),
            quantity=:consumption),
        traces=(;
            path_ids=_trace_path_ids(config),
            paths_per_initial_cash=_trace_paths_per_initial_cash(config),
            source=:balanced_first_paths_per_initial_cash,
            new_shocks=false,
            outside_reported_timing=true))
end

scientific_fingerprint(config::AR1ContestExhibitConfig) =
    stable_fingerprint(scientific_configuration(config))

function _is_publication_scientific_configuration(scientific)
    # Workers are execution settings, not a restriction on paper replication.
    # Preserve their original inclusion in the authenticated payload so saved
    # artifacts remain valid; only this publication-design comparison varies.
    evaluation = _exhibit_field(scientific, :evaluation, "scientific_configuration")
    workers = Int(_exhibit_field(evaluation, :workers, "evaluation"))
    paper = scientific_configuration(production_ar1_contest_exhibit_config())
    expected = merge(paper, (evaluation=merge(paper.evaluation, (; workers)),))
    return stable_fingerprint(scientific) == stable_fingerprint(expected)
end

_is_publication_config(config::AR1ContestExhibitConfig) =
    _is_publication_scientific_configuration(scientific_configuration(config))

"""Return the flat JLD2 path for the paper's AR(1) exhibit artifact."""
function default_ar1_contest_exhibit_artifact_path(
    config::AR1ContestExhibitConfig=production_ar1_contest_exhibit_config();
    root::AbstractString=datadir())
    validate_ar1_contest_exhibit_config(config)
    return joinpath(abspath(root), "ar1-contest.jld2")
end

function _path_partitions(n_paths::Int, workers::Int)
    count = min(n_paths, workers)
    base, extra = divrem(n_paths, count)
    ranges = UnitRange{Int}[]
    first_path = 1
    for worker in 1:count
        width = base + (worker <= extra)
        last_path = first_path + width - 1
        push!(ranges, first_path:last_path)
        first_path = last_path + 1
    end
    first(first(ranges)) == 1 && last(last(ranges)) == n_paths ||
        error("internal path partition failure")
    return ranges
end

_exhibit_utc_string(value::DateTime) = Dates.format(value, _EXHIBIT_UTC_FORMAT) * "Z"
_elapsed_seconds(started::UInt64) =
    Float64(UInt64(time_ns()) - started) / 1.0e9

function _float_bits(values)
    return Tuple(reinterpret(UInt64, Float64(value isa Tuple ? only(value) : value))
                 for value in values)
end

function _exhibit_sha256_file(path::AbstractString)
    return bytes2hex(sha256(read(path)))
end

function _exhibit_computational_input_tree()
    root = PROJECT_ROOT
    paths = String[joinpath(root, "Project.toml"),
        joinpath(root, "Manifest.toml")]
    for (directory, _, files) in walkdir(joinpath(root, "src"))
        for file in files
            endswith(file, ".jl") && push!(paths, joinpath(directory, file))
        end
    end
    runner = joinpath(root, "scripts", "4_ar1.jl")
    isfile(runner) && push!(paths, runner)
    sort!(unique!(paths))
    entries = [(path=relpath(path, root), sha256=_exhibit_sha256_file(path))
               for path in paths]
    return (schema="ptd-ar1-contest-exhibit-input-tree-v1",
        files=entries, fingerprint=stable_fingerprint(entries))
end

function _tiny_warmup_grid()
    return AR1G2ContestGridConfiguration(
        :ar1_contest_exhibit_warmup_v1,
        [0.0, 1.0, 4.0, 20.0],
        [-3.0, 0.0, 3.0],
        [3.0, 7.0, 15.0, 30.0, 45.0],
        [-3.0, 0.0, 3.0])
end

"""Compile the relevant solver paths without building the publication G2 grid."""
function warm_ar1_contest_exhibits!(config::AR1ContestExhibitConfig)
    validate_ar1_contest_exhibit_config(config)
    process = LogAR1Process(rho=config.rho, sigma=config.sigma,
        unconditional_log_mean=config.unconditional_log_mean)
    rule = gauss_hermite_rule(process, 5)
    full_model = StochasticPTDConfig(beta=config.beta, gamma=config.gamma,
        return_rate=config.return_rate, cash_floor=config.cash_floor,
        terminal_date=config.horizon)
    initial = ObservedState(1, first(config.initial_cash_support),
        LogAR1State(log(config.initial_income)))
    for method in (:ptd_sceq, :ptd_osa)
        policy = build_ar1_contest_policy(method, full_model, process;
            base_rule=rule, ptd_osa_options=config.ptd_osa_options,
            g2egm_grids=config.g2egm_grids,
            g2egm_options=config.g2egm_options)
        evaluate_policy_paths(policy, method, full_model, process, initial,
            zeros(1, config.horizon - 1);
            action_audit=PolicyActionAudit())
    end

    tiny_model = StochasticPTDConfig(beta=config.beta, gamma=config.gamma,
        return_rate=config.return_rate, cash_floor=config.cash_floor,
        terminal_date=3)
    tiny_grid = _tiny_warmup_grid()
    g2 = build_ar1_contest_policy(:g2egm, tiny_model, process;
        base_rule=rule, ptd_osa_options=config.ptd_osa_options,
        g2egm_grids=tiny_grid, g2egm_options=config.g2egm_options)
    # This discarded solve exists only to compile the G2 evaluator. Its tiny
    # policy domain is intentionally independent of the scientific initial
    # cash, which may lie anywhere on the publication surface.
    tiny_initial = ObservedState(1, 5.0,
        LogAR1State(log(config.initial_income)))
    evaluate_policy_paths(g2, :g2egm, tiny_model, process, tiny_initial,
        zeros(1, 2))
    return nothing
end

_method_named_tuple(values::AbstractDict) =
    (; (method => values[method]
        for method in AR1_CONTEST_EXHIBIT_METHODS)...)

function _exhibit_model_inputs(config::AR1ContestExhibitConfig)
    model = StochasticPTDConfig(beta=config.beta, gamma=config.gamma,
        return_rate=config.return_rate, cash_floor=config.cash_floor,
        terminal_date=config.horizon)
    process = LogAR1Process(rho=config.rho, sigma=config.sigma,
        unconditional_log_mean=config.unconditional_log_mean)
    rule = gauss_hermite_rule(process,
        config.quadrature_nodes_per_dimension)
    return (; model, process, rule)
end

_initial_state(config::AR1ContestExhibitConfig, cash::Real) =
    ObservedState(1, Float64(cash),
        LogAR1State(log(config.initial_income)))

function _constant_cash_subranges(range::UnitRange{Int}, assignments)
    isempty(range) && return UnitRange{Int}[]
    subranges = UnitRange{Int}[]
    lo = first(range)
    for index in (lo + 1):last(range)
        assignments[index] == assignments[index - 1] && continue
        push!(subranges, lo:(index - 1))
        lo = index
    end
    push!(subranges, lo:last(range))
    return subranges
end

function _build_policy(method::Symbol, inputs,
                       config::AR1ContestExhibitConfig)
    return build_ar1_contest_policy(method, inputs.model, inputs.process;
        base_rule=inputs.rule,
        ptd_osa_options=config.ptd_osa_options,
        g2egm_grids=config.g2egm_grids,
        g2egm_options=config.g2egm_options)
end

function _build_policy_workers(config::AR1ContestExhibitConfig, inputs,
                               partitions)
    policies = Dict{Symbol,Vector{Any}}()
    seconds = Dict{Symbol,Float64}()
    for method in AR1_CONTEST_EXHIBIT_METHODS
        started = UInt64(time_ns())
        method_policies = if method === :g2egm
            solved = _build_policy(method, inputs, config)
            fill(solved, length(partitions))
        else
            [_build_policy(method, inputs, config) for _ in partitions]
        end
        seconds[method] = _elapsed_seconds(started)

        # Output validation is outside the reported construction interval.
        if method === :g2egm
            validate_ar1_contest_policy_alignment(first(method_policies),
                method, inputs.model, inputs.process)
        else
            for policy in method_policies
                validate_ar1_contest_policy_alignment(policy, method,
                    inputs.model, inputs.process)
            end
        end
        policies[method] = method_policies
    end
    return policies, seconds
end

function _reset_method_diagnostics!(method::Symbol, policies)
    if method === :g2egm
        reset_ar1_contest_diagnostics!(first(policies))
    else
        for policy in policies
            reset_ar1_contest_diagnostics!(policy)
        end
    end
    return nothing
end

function _exhibit_validate_evaluation(evaluation, method::Symbol, paths::Int)
    evaluation isa PolicyPathEvaluation || throw(ArgumentError(
        "the path evaluator returned the wrong result type"))
    evaluation.method === method || throw(ArgumentError(
        "the path evaluator returned the wrong method id"))
    length(evaluation.utility) == paths &&
        length(evaluation.floor_hits) == paths &&
        length(evaluation.total_transfers) == paths ||
        throw(DimensionMismatch("a path partition has the wrong length"))
    all(isfinite, evaluation.utility) || throw(ArgumentError(
        "pathwise lifetime utility must be finite"))
    all(>=(0), evaluation.floor_hits) || throw(ArgumentError(
        "floor-hit counts must be nonnegative"))
    all(value -> isfinite(value) && value >= 0.0,
        evaluation.total_transfers) || throw(ArgumentError(
            "total transfers must be finite and nonnegative"))
    return evaluation
end

function _evaluate_partition(method::Symbol, policy, config, inputs,
                             innovations, range::UnitRange{Int},
                             initial_cash_assignments)
    first_thread = Threads.threadid()
    utility = Vector{Float64}(undef, length(range))
    floor_hits = Vector{Int}(undef, length(range))
    total_transfers = Vector{Float64}(undef, length(range))
    for subrange in _constant_cash_subranges(
        range, initial_cash_assignments)
        initial = _initial_state(
            config, initial_cash_assignments[first(subrange)])
        evaluation = evaluate_policy_paths(
            policy, method, inputs.model, inputs.process,
            initial, view(innovations, subrange, :);
            action_audit=PolicyActionAudit())
        _exhibit_validate_evaluation(evaluation, method, length(subrange))
        destination = (first(subrange) - first(range) + 1):(
            last(subrange) - first(range) + 1)
        utility[destination] = evaluation.utility
        floor_hits[destination] = evaluation.floor_hits
        total_transfers[destination] = evaluation.total_transfers
    end
    combined = PolicyPathEvaluation(
        method, utility, floor_hits, total_transfers)
    last_thread = Threads.threadid()
    return (;
        evaluation=_exhibit_validate_evaluation(combined, method, length(range)),
        thread_pair=(first_thread, last_thread))
end

function _evaluate_method(method::Symbol, policies, config, inputs,
                          innovations, partitions, initial_cash_assignments)
    _reset_method_diagnostics!(method, policies)
    outputs = Vector{Any}(undef, length(partitions))
    started = UInt64(time_ns())
    Threads.@threads :static for index in eachindex(partitions)
        outputs[index] = _evaluate_partition(method, policies[index],
            config, inputs, innovations, partitions[index],
            initial_cash_assignments)
    end
    seconds = _elapsed_seconds(started)
    results = PolicyPathEvaluation[
        output.evaluation for output in outputs]
    thread_pairs = NTuple{2,Int}[
        output.thread_pair for output in outputs]

    utility = Vector{Float64}(undef, config.n_paths)
    floor_hits = Vector{Int}(undef, config.n_paths)
    total_transfers = Vector{Float64}(undef, config.n_paths)
    for (range, evaluation) in zip(partitions, results)
        utility[range] = evaluation.utility
        floor_hits[range] = evaluation.floor_hits
        total_transfers[range] = evaluation.total_transfers
    end
    thread_ids = sort!(unique!(collect(Iterators.flatten(thread_pairs))))
    return (; utility, floor_hits, total_transfers, seconds,
        thread_ids, partition_ranges=copy(partitions))
end

function _quadrature_contract(policies)
    audits = Dict(method => ar1_contest_quadrature_audit(
        first(policies[method])) for method in AR1_CONTEST_EXHIBIT_METHODS)
    audits[:ptd_sceq].uses_quadrature === false || throw(ErrorException(
        "PTD-SCEQ unexpectedly reported numerical quadrature"))
    audits[:ptd_osa].uses_quadrature === true &&
        audits[:g2egm].uses_quadrature === true || throw(ErrorException(
            "PTD-OSA and G2EGM must both report Q5 quadrature"))
    Int(audits[:ptd_osa].nodes_per_dimension) == 5 &&
        Int(audits[:g2egm].nodes_per_dimension) == 5 ||
        throw(ErrorException("PTD-OSA and G2EGM did not both use Q5"))
    oa_node_bits = _float_bits(audits[:ptd_osa].base_nodes)
    oa_weight_bits = _float_bits(audits[:ptd_osa].base_weights)
    g2_node_bits = _float_bits(audits[:g2egm].base_nodes)
    g2_weight_bits = _float_bits(audits[:g2egm].base_weights)
    oa_node_bits == g2_node_bits && oa_weight_bits == g2_weight_bits ||
        throw(ErrorException(
            "PTD-OSA and G2EGM quadrature bits are not identical"))
    return (;
        ptd_sceq=audits[:ptd_sceq],
        ptd_osa=audits[:ptd_osa],
        g2egm=audits[:g2egm],
        exact_shared_q5=true,
        node_bits=oa_node_bits,
        weight_bits=oa_weight_bits)
end

function _collect_diagnostics(policies)
    return (;
        ptd_sceq=[ar1_contest_diagnostics(policy)
                  for policy in policies[:ptd_sceq]],
        ptd_osa=[ar1_contest_diagnostics(policy)
                for policy in policies[:ptd_osa]],
        g2egm=ar1_contest_diagnostics(first(policies[:g2egm])))
end

function _validate_solver_diagnostics(diagnostics)
    all(row -> row.certification_failures == 0 &&
               row.fallback_uses == 0 &&
               row.calls == row.certified_calls,
        diagnostics.ptd_osa) || throw(ErrorException(
            "PTD-OSA reported a certification failure or direct fallback"))
    g2 = diagnostics.g2egm.solver
    g2.query.domain_errors == 0 || throw(ErrorException(
        "G2EGM queried outside its declared domain"))
    g2.query.asset_ceiling_choices == 0 || throw(ErrorException(
        "G2EGM selected the numerical asset ceiling on a reported query"))
    g2.solve.policy_domain_asset_ceiling_winners == 0 ||
        throw(ErrorException(
            "G2EGM selected the numerical asset ceiling on its policy domain"))
    return diagnostics
end

function _surface_method(method::Symbol, policies, config)
    cash = config.surface_cash_grid
    logs = config.surface_log_income_grid
    consumption = Matrix{Float64}(undef, length(cash), length(logs))
    saving = similar(consumption)
    log_partitions = _path_partitions(length(logs),
        min(length(policies), length(logs)))
    thread_ids = Vector{NTuple{2,Int}}(undef, length(log_partitions))
    started = UInt64(time_ns())
    @sync for worker in eachindex(log_partitions)
        Threads.@spawn begin
            first_thread = Threads.threadid()
            policy = policies[worker]
            for log_index in log_partitions[worker]
                ell = logs[log_index]
                for cash_index in eachindex(cash)
                    x = cash[cash_index]
                    action = policy(ObservedState(config.surface_date, x,
                        LogAR1State(ell)))
                    consumption[cash_index, log_index] = action.consumption
                    saving[cash_index, log_index] = action.saving
                end
            end
            thread_ids[worker] = (first_thread, Threads.threadid())
        end
    end
    seconds = _elapsed_seconds(started)
    all(isfinite, consumption) && all(isfinite, saving) ||
        throw(ErrorException("a policy surface contains nonfinite values"))
    tolerance = 128eps(max(last(cash), 1.0))
    all(consumption .>= -tolerance) &&
        all(consumption .<= reshape(cash, :, 1) .+ tolerance) ||
        throw(ErrorException("a policy surface contains infeasible consumption"))
    minimum(diff(saving; dims=1)) >= -1.0e-7 || throw(ErrorException(
        "a policy surface violates monotone saving"))
    return (; consumption, saving, seconds,
        thread_ids=sort!(unique!(collect(Iterators.flatten(thread_ids)))))
end

function _sample_surfaces(policies, config)
    results = Dict(method => _surface_method(
        method, policies[method], config)
        for method in AR1_CONTEST_EXHIBIT_METHODS)
    return (;
        date=config.surface_date,
        cash_grid=copy(config.surface_cash_grid),
        log_income_grid=copy(config.surface_log_income_grid),
        consumption=_method_named_tuple(Dict(method =>
            results[method].consumption
            for method in AR1_CONTEST_EXHIBIT_METHODS)),
        saving=_method_named_tuple(Dict(method => results[method].saving
            for method in AR1_CONTEST_EXHIBIT_METHODS)),
        query_seconds=_method_named_tuple(Dict(method => results[method].seconds
            for method in AR1_CONTEST_EXHIBIT_METHODS)),
        thread_ids=_method_named_tuple(Dict(method => results[method].thread_ids
            for method in AR1_CONTEST_EXHIBIT_METHODS)),
        outside_reported_timing=true)
end

function _trace_group(method::Symbol, policy, config, inputs, innovations,
                      path_ids, per_cash::Int, group::Int)
    display_range = ((group - 1) * per_cash + 1):(group * per_cash)
    welfare_range = path_ids[display_range]
    initial = _initial_state(
        config, config.initial_cash_support[group])
    rows = Any[]
    evaluation = evaluate_policy_paths(policy, method,
        inputs.model, inputs.process, initial,
        view(innovations, welfare_range, :);
        action_audit=PolicyActionAudit(),
        trajectory_rows=rows,
        trajectory_path_offset=first(display_range) - 1)
    return (; evaluation, rows)
end

function _trace_method(method::Symbol, policies, config, inputs,
                       innovations)
    path_ids = _trace_path_ids(config)
    per_cash = _trace_paths_per_initial_cash(config)
    groups = eachindex(config.initial_cash_support)
    outputs = Vector{Any}(undef, length(groups))
    started = UInt64(time_ns())
    Threads.@threads :static for group in groups
        outputs[group] = _trace_group(method,
            policies[mod1(group, length(policies))], config, inputs,
            innovations, path_ids, per_cash, group)
    end
    seconds = _elapsed_seconds(started)
    rows = reduce(vcat, (output.rows for output in outputs); init=Any[])
    length(rows) == config.trace_paths * config.horizon ||
        throw(ErrorException("trace replay returned the wrong row count"))
    cash = fill(NaN, config.horizon, config.trace_paths)
    consumption = similar(cash)
    income = similar(cash)
    utility = fill(NaN, config.trace_paths)
    for group in groups
        display_range = ((group - 1) * per_cash + 1):(group * per_cash)
        evaluation = outputs[group].evaluation
        utility[display_range] = evaluation.utility
    end
    for row in rows
        path = Int(row.path)
        date = Int(row.observed.date)
        cash[date, path] = Float64(row.observed.cash)
        consumption[date, path] = Float64(row.action.consumption)
        income[date, path] = exp(Float64(row.observed.exogenous.log_income))
    end
    all(isfinite, cash) && all(isfinite, consumption) && all(isfinite, income) &&
        all(isfinite, utility) ||
        throw(ErrorException("trace replay left an unfilled or nonfinite cell"))
    return (; cash, consumption, income, utility, seconds)
end

function _sample_traces(policies, config, inputs, innovations,
                        expected_utilities)
    path_ids = _trace_path_ids(config)
    initial_cash_assignments = _initial_cash_assignments(config)
    results = Dict(method => _trace_method(method, policies[method], config,
        inputs, innovations)
        for method in AR1_CONTEST_EXHIBIT_METHODS)
    reference_income = results[:ptd_sceq].income
    all(method -> results[method].income == reference_income,
        AR1_CONTEST_EXHIBIT_METHODS) || throw(ErrorException(
            "methods did not replay identical AR(1) income paths"))
    for (column, method) in pairs(AR1_CONTEST_EXHIBIT_METHODS)
        results[method].utility ==
            expected_utilities[path_ids, column] ||
            throw(ErrorException(
                "trace replay utility does not match the welfare block for $method"))
    end
    return (;
        dates=collect(1:config.horizon),
        path_ids,
        initial_cash=initial_cash_assignments[path_ids],
        income=reference_income,
        cash=_method_named_tuple(Dict(method => results[method].cash
            for method in AR1_CONTEST_EXHIBIT_METHODS)),
        consumption=_method_named_tuple(Dict(method =>
            results[method].consumption
            for method in AR1_CONTEST_EXHIBIT_METHODS)),
        replayed_utility=_method_named_tuple(Dict(method =>
            results[method].utility
            for method in AR1_CONTEST_EXHIBIT_METHODS)),
        replay_seconds=_method_named_tuple(Dict(method => results[method].seconds
            for method in AR1_CONTEST_EXHIBIT_METHODS)),
        innovation_rows=Matrix{Float64}(view(innovations, path_ids, :)),
        outside_reported_timing=true,
        adds_welfare_observations=false)
end

function _welfare_statistics(config, utilities)
    expected = NamedTuple[]
    for (column, method) in pairs(AR1_CONTEST_EXHIBIT_METHODS)
        values = view(utilities, :, column)
        push!(expected, (; method,
            mean_utility=mean(values),
            monte_carlo_standard_error=
                std(values; corrected=true) / sqrt(length(values))))
    end
    comparisons = NamedTuple[]
    for a in 1:length(AR1_CONTEST_EXHIBIT_METHODS)-1,
        b in a+1:length(AR1_CONTEST_EXHIBIT_METHODS)
        method_a = AR1_CONTEST_EXHIBIT_METHODS[a]
        method_b = AR1_CONTEST_EXHIBIT_METHODS[b]
        result = paired_comparison(method_a, method_b,
            view(utilities, :, a), view(utilities, :, b);
            level=config.confidence_level)
        mean_a = expected[a].mean_utility
        mean_b = expected[b].mean_utility
        ce_point = consumption_equivalent_bps(mean_a, mean_b,
            config.beta, config.gamma, config.horizon)
        ce_lower = consumption_equivalent_bps(
            mean_b + result.ci_lower, mean_b,
            config.beta, config.gamma, config.horizon)
        ce_upper = consumption_equivalent_bps(
            mean_b + result.ci_upper, mean_b,
            config.beta, config.gamma, config.horizon)
        classification = result.ci_lower > 0.0 ? :method_a_preferred :
            result.ci_upper < 0.0 ? :method_b_preferred : :mc_unresolved
        push!(comparisons, (;
            method_a, method_b,
            n_paths=result.n_paths,
            confidence_level=result.confidence_level,
            mean_utility_difference=result.mean_difference,
            utility_standard_error=result.standard_error,
            utility_ci_lower=result.ci_lower,
            utility_ci_upper=result.ci_upper,
            ce_gap_bps=ce_point,
            ce_ci_lower_bps=ce_lower,
            ce_ci_upper_bps=ce_upper,
            classification))
    end
    return (; expected_utility=expected, comparisons)
end

function _result_payload(scientific_fingerprint_value, quadrature,
                         solver_diagnostics, welfare, timing, surface, traces)
    return (;
        scientific_fingerprint=scientific_fingerprint_value,
        quadrature,
        solver_diagnostics,
        welfare,
        timing,
        surface,
        traces)
end

function _artifact_authentication(scientific_fingerprint_value,
    result_fingerprint, timing_fingerprint, provenance, completed_utc)
    return stable_fingerprint((;
        schema=AR1_CONTEST_EXHIBIT_AUTHENTICATION_SCHEMA,
        artifact_schema=AR1_CONTEST_EXHIBIT_SCHEMA,
        scientific_fingerprint=scientific_fingerprint_value,
        result_fingerprint,
        timing_fingerprint,
        provenance_fingerprint=stable_fingerprint(provenance),
        completed_utc=String(completed_utc),
        input_tree_fingerprint=provenance.input_tree.fingerprint,
        source_commit=provenance.simulation.source.commit,
        solver_schema=String(_exhibit_field(provenance.upstream,
            :solver_schema, "provenance.upstream")),
        upstream_commit=String(_exhibit_field(provenance.upstream,
            :commit, "provenance.upstream"))))
end

"""Compute the complete contest-exhibit artifact without writing it."""
function build_ar1_contest_exhibit_artifact(
    config::AR1ContestExhibitConfig=
        production_ar1_contest_exhibit_config();
    warmup::Bool=true,
    require_publication::Bool=_is_publication_config(config),
    completed_at::DateTime=now(UTC))

    validate_ar1_contest_exhibit_config(config)
    publication = _is_publication_config(config)
    require_publication && !publication && throw(ArgumentError(
        "the requested configuration is not the frozen paper exhibit"))
    LinearAlgebra.BLAS.set_num_threads(1)
    LinearAlgebra.BLAS.get_num_threads() == 1 || throw(ErrorException(
        "the contest-exhibit experiment requires one BLAS thread"))
    if publication
        warmup || throw(ArgumentError(
            "the publication timing requires an explicit untimed warm-up"))
    end

    inputs = _exhibit_model_inputs(config)
    rng = Xoshiro(config.evaluation_seed)
    innovations = randn(rng, config.n_paths, config.horizon - 1)
    shock_fingerprint = stable_fingerprint(innovations)
    initial_cash_assignments = _initial_cash_assignments(config)
    initial_cash_assignments_fingerprint =
        stable_fingerprint(initial_cash_assignments)
    partitions = _path_partitions(config.n_paths, config.workers)
    warmup && warm_ar1_contest_exhibits!(config)

    policies, construction_seconds = _build_policy_workers(
        config, inputs, partitions)
    quadrature = _quadrature_contract(policies)
    evaluations = Dict{Symbol,Any}()
    for method in AR1_CONTEST_EXHIBIT_METHODS
        evaluations[method] = _evaluate_method(method, policies[method],
            config, inputs, innovations, partitions,
            initial_cash_assignments)
    end

    utilities = Matrix{Float64}(undef, config.n_paths,
        length(AR1_CONTEST_EXHIBIT_METHODS))
    floor_hits = Matrix{Int}(undef, size(utilities))
    total_transfers = Matrix{Float64}(undef, size(utilities))
    for (column, method) in pairs(AR1_CONTEST_EXHIBIT_METHODS)
        utilities[:, column] = evaluations[method].utility
        floor_hits[:, column] = evaluations[method].floor_hits
        total_transfers[:, column] = evaluations[method].total_transfers
    end
    statistics = _welfare_statistics(config, utilities)
    # These queries reuse the already constructed policies and are explicitly
    # outside construction/evaluation timing.
    traces = _sample_traces(
        policies, config, inputs, innovations, utilities)
    surface = _sample_surfaces(policies, config)
    final_diagnostics = _validate_solver_diagnostics(
        _collect_diagnostics(policies))

    evaluation_seconds = Dict(method => evaluations[method].seconds
        for method in AR1_CONTEST_EXHIBIT_METHODS)
    end_to_end_seconds = Dict(method =>
        construction_seconds[method] + evaluation_seconds[method]
        for method in AR1_CONTEST_EXHIBIT_METHODS)
    timing = (;
        convention=:one_construction_plus_one_common_path_block,
        construction_seconds=_method_named_tuple(construction_seconds),
        evaluation_seconds=_method_named_tuple(evaluation_seconds),
        end_to_end_seconds=_method_named_tuple(end_to_end_seconds),
        evaluation_boundary_thread_ids=_method_named_tuple(Dict(method =>
            evaluations[method].thread_ids
            for method in AR1_CONTEST_EXHIBIT_METHODS)),
        partition_ranges=copy(partitions),
        julia_threads=Threads.nthreads(),
        # Julia numbers interactive threads before default-pool threads. Store
        # both pools so an artifact can be validated on a different machine.
        thread_pools=(;
            default_threads=Threads.nthreads(:default),
            interactive_threads=Threads.nthreads(:interactive),
            default_thread_ids=Threads.threadpooltids(:default),
            interactive_thread_ids=Threads.threadpooltids(:interactive)),
        configured_workers=config.workers,
        blas_threads=LinearAlgebra.BLAS.get_num_threads(),
        timing_observations_per_method=1,
        compilation_warmup_outside=true,
        shock_generation_outside=true,
        validation_outside=true,
        io_outside=true,
        surface_queries_outside=true,
        trace_replay_outside=true,
        no_timing_dispersion_estimate=true)
    if publication && min(Threads.nthreads(), config.workers) > 1
        length(timing.evaluation_boundary_thread_ids.ptd_sceq) > 1 &&
            length(timing.evaluation_boundary_thread_ids.ptd_osa) > 1 ||
            throw(ErrorException(
                "the publication PTD evaluation did not use multiple Julia threads"))
    end

    welfare = (;
        methods=collect(AR1_CONTEST_EXHIBIT_METHODS),
        utility=utilities,
        floor_hits,
        total_transfers,
        expected_utility=statistics.expected_utility,
        comparisons=statistics.comparisons,
        innovations,
        shock_fingerprint,
        initial_cash_assignments,
        initial_cash_assignments_fingerprint,
        n_paths=config.n_paths,
        confidence_level=config.confidence_level)
    solver_diagnostics = (;
        scope=:welfare_surface_and_trace_queries,
        display_queries_outside_reported_timing=true,
        final=final_diagnostics)
    scientific = scientific_configuration(config)
    scientific_fp = stable_fingerprint(scientific)
    input_tree = _exhibit_computational_input_tree()
    provenance = (;
        simulation=simulation_metadata(scientific, PROJECT_ROOT;
            timestamp=completed_at),
        input_tree,
        upstream=(;
            repository="JeppeDruedahl/G2EGM",
            commit=G2EGM_REFERENCE_COMMIT,
            solver_schema=G2EGM_SOLVER_SCHEMA,
            runtime_dependencies=:pure_julia_no_matlab_mex_or_cpp))
    result_payload = _result_payload(scientific_fp, quadrature,
        solver_diagnostics, welfare, timing, surface, traces)
    result_fingerprint = stable_fingerprint(result_payload)
    timing_fingerprint = stable_fingerprint(timing)
    completed_utc = _exhibit_utc_string(completed_at)
    provenance_fingerprint = stable_fingerprint(provenance)
    authentication_fingerprint = _artifact_authentication(
        scientific_fp, result_fingerprint, timing_fingerprint, provenance,
        completed_utc)
    artifact = (;
        metadata=(;
            artifact_schema=AR1_CONTEST_EXHIBIT_SCHEMA,
            authentication_schema=
                AR1_CONTEST_EXHIBIT_AUTHENTICATION_SCHEMA,
            status=:complete,
            completed_utc,
            scientific_fingerprint=scientific_fp,
            result_fingerprint,
            timing_fingerprint,
            provenance_fingerprint,
            authentication_fingerprint),
        scientific_configuration=scientific,
        provenance,
        quadrature,
        solver_diagnostics,
        welfare,
        timing,
        surface,
        traces)
    return validate_ar1_contest_exhibit_artifact(artifact;
        expected_config=config, require_publication)
end

function _exhibit_field(container, name::Symbol, location::AbstractString)
    if container isa NamedTuple
        hasproperty(container, name) || throw(ArgumentError(
            "$location is missing field $name"))
        return getproperty(container, name)
    elseif container isa AbstractDict
        haskey(container, name) && return container[name]
        haskey(container, String(name)) && return container[String(name)]
    end
    throw(ArgumentError("$location is missing field $name"))
end

function _method_values(container, location::AbstractString)
    return Dict(method => _exhibit_field(container, method, location)
        for method in AR1_CONTEST_EXHIBIT_METHODS)
end

function _config_from_scientific(scientific)
    model = _exhibit_field(scientific, :model, "scientific_configuration")
    evaluation = _exhibit_field(scientific, :evaluation, "scientific_configuration")
    initial_cash_support = Float64.(collect(_exhibit_field(
        model, :initial_cash_support, "model")))
    initial_cash_weights = Float64.(collect(_exhibit_field(
        model, :initial_cash_weights, "model")))
    length(initial_cash_support) == length(initial_cash_weights) &&
        initial_cash_weights == fill(
            inv(length(initial_cash_support)), length(initial_cash_support)) ||
        throw(ArgumentError(
            "initial cash support does not have declared uniform weights"))
    config = (;
        beta=Float64(_exhibit_field(model, :beta, "model")),
        gamma=Float64(_exhibit_field(model, :gamma, "model")),
        horizon=Int(_exhibit_field(model, :horizon, "model")),
        cash_floor=Float64(_exhibit_field(model, :cash_floor, "model")),
        return_rate=Float64(_exhibit_field(model, :return_rate, "model")),
        rho=Float64(_exhibit_field(model, :rho, "model")),
        sigma=Float64(_exhibit_field(model, :innovation_sigma, "model")),
        unconditional_log_mean=Float64(_exhibit_field(
            model, :unconditional_log_mean, "model")),
        initial_income=Float64(_exhibit_field(model, :initial_income, "model")),
        initial_cash_support,
        n_paths=Int(_exhibit_field(evaluation, :n_paths, "evaluation")),
        evaluation_seed=UInt64(_exhibit_field(evaluation, :seed, "evaluation")),
        workers=Int(_exhibit_field(evaluation, :workers, "evaluation")),
        confidence_level=Float64(_exhibit_field(
            evaluation, :confidence_level, "evaluation")))
    Int(_exhibit_field(evaluation, :paths_per_initial_cash, "evaluation")) ==
        div(config.n_paths, length(initial_cash_support)) ||
        throw(ArgumentError(
            "paths per initial cash disagree with the balanced design"))
    Symbol(_exhibit_field(evaluation, :initial_cash_assignment, "evaluation")) ===
        :balanced_contiguous_blocks || throw(ArgumentError(
            "the initial cash assignment rule is invalid"))
    return config
end

"""
Compare recomputed summaries with their stored rows to floating-point precision.

Bounds checks and vectorized reduction order can change means/standard errors
by a few ulps. Subtracting nearby means and converting to basis points amplifies
that drift in CE gaps. Allow 64 ulps at a scale of at least one utility unit, or
10,000 basis points for CE fields (one unit of the unscaled consumption ratio).
Methods, counts, confidence levels and classifications remain exact. This is
only a numerical reconstruction check: the original stored payload, including
these summaries, is still authenticated by exact result/provenance hashes.
"""
function _welfare_summary_rows_reproduce(stored, recomputed)
    length(stored) == length(recomputed) || return false
    for (old, new) in zip(stored, recomputed)
        keys(old) == keys(new) || return false
        for name in keys(old)
            a, b = getproperty(old, name), getproperty(new, name)
            if a isa AbstractFloat && b isa AbstractFloat &&
                    name !== :confidence_level
                isfinite(a) && isfinite(b) || return false
                unit_scale = startswith(String(name), "ce_") ? 10_000.0 : 1.0
                scale = max(unit_scale, abs(Float64(a)), abs(Float64(b)))
                abs(a - b) <= 64eps(scale) || return false
            else
                isequal(a, b) || return false
            end
        end
    end
    return true
end

function _validate_welfare(welfare, scientific)
    config = _config_from_scientific(scientific)
    Int(_exhibit_field(welfare, :n_paths, "welfare")) == config.n_paths ||
        throw(ArgumentError(
            "the welfare path count disagrees with the configuration"))
    methods = Symbol.(collect(_exhibit_field(welfare, :methods, "welfare")))
    methods == collect(AR1_CONTEST_EXHIBIT_METHODS) ||
        throw(ArgumentError("welfare methods are missing or out of order"))
    utility = Matrix{Float64}(_exhibit_field(welfare, :utility, "welfare"))
    size(utility) == (config.n_paths,
        length(AR1_CONTEST_EXHIBIT_METHODS)) ||
        throw(DimensionMismatch("the welfare utility matrix has the wrong shape"))
    all(isfinite, utility) || throw(ArgumentError(
        "the welfare utility matrix contains nonfinite values"))
    floor_hits = Matrix{Int}(_exhibit_field(welfare, :floor_hits, "welfare"))
    transfers = Matrix{Float64}(_exhibit_field(
        welfare, :total_transfers, "welfare"))
    size(floor_hits) == size(utility) && size(transfers) == size(utility) ||
        throw(DimensionMismatch("welfare diagnostic matrices have the wrong shape"))
    all(floor_hits .>= 0) &&
        all(value -> isfinite(value) && value >= 0.0, transfers) ||
        throw(ArgumentError("welfare path diagnostics are invalid"))
    innovations = Matrix{Float64}(_exhibit_field(welfare, :innovations, "welfare"))
    size(innovations) == (config.n_paths, config.horizon - 1) &&
        all(isfinite, innovations) || throw(DimensionMismatch(
            "the innovation matrix has the wrong shape or values"))
    stable_fingerprint(innovations) == String(_exhibit_field(
        welfare, :shock_fingerprint, "welfare")) ||
        throw(ArgumentError("the welfare shock fingerprint is invalid"))
    expected_innovations = randn(Xoshiro(config.evaluation_seed),
        config.n_paths, config.horizon - 1)
    innovations == expected_innovations || throw(ArgumentError(
        "the welfare innovations do not reproduce from the declared seed"))
    initial_cash_assignments = Float64.(collect(_exhibit_field(
        welfare, :initial_cash_assignments, "welfare")))
    expected_assignments = _initial_cash_assignments(
        AR1ContestExhibitConfig(
            initial_cash_support=config.initial_cash_support,
            n_paths=config.n_paths,
            trace_paths=length(config.initial_cash_support), workers=1))
    initial_cash_assignments == expected_assignments || throw(ArgumentError(
        "the welfare initial-cash assignments are not balanced as declared"))
    stable_fingerprint(initial_cash_assignments) == String(_exhibit_field(
        welfare, :initial_cash_assignments_fingerprint, "welfare")) ||
        throw(ArgumentError(
            "the welfare initial-cash assignment fingerprint is invalid"))
    Float64(_exhibit_field(welfare, :confidence_level, "welfare")) ==
        config.confidence_level || throw(ArgumentError(
            "the welfare confidence level disagrees with the configuration"))

    reconstructed = _welfare_statistics(
        AR1ContestExhibitConfig(beta=config.beta, gamma=config.gamma,
            return_rate=config.return_rate, cash_floor=config.cash_floor,
            horizon=config.horizon, rho=config.rho, sigma=config.sigma,
            initial_income=config.initial_income,
            initial_cash_support=config.initial_cash_support,
            n_paths=config.n_paths, trace_paths=1,
            confidence_level=config.confidence_level,
            workers=1,
            surface_cash_grid=[config.cash_floor,
                last(config.initial_cash_support) + 1.0],
            surface_log_income_grid=[-0.1, 0.1]),
        utility)
    _welfare_summary_rows_reproduce(
        collect(_exhibit_field(welfare, :expected_utility, "welfare")),
        reconstructed.expected_utility) || throw(ArgumentError(
            "stored expected-utility rows do not reproduce from path utilities"))
    _welfare_summary_rows_reproduce(
        collect(_exhibit_field(welfare, :comparisons, "welfare")),
        reconstructed.comparisons) || throw(ArgumentError(
            "stored paired comparisons do not reproduce from path utilities"))
    return (; utility, innovations)
end

function _validate_surface(surface, scientific)
    declared = _exhibit_field(scientific, :surface, "scientific_configuration")
    Int(_exhibit_field(surface, :date, "surface")) ==
        Int(_exhibit_field(declared, :date, "surface config")) ||
        throw(ArgumentError(
            "the surface date disagrees with the configuration"))
    cash = Float64.(collect(_exhibit_field(surface, :cash_grid, "surface")))
    logs = Float64.(collect(_exhibit_field(surface, :log_income_grid, "surface")))
    cash == Float64.(collect(_exhibit_field(declared, :cash_grid, "surface config"))) &&
        logs == Float64.(collect(_exhibit_field(
            declared, :log_income_grid, "surface config"))) ||
        throw(ArgumentError("surface axes disagree with the configuration"))
    consumption = _method_values(_exhibit_field(
        surface, :consumption, "surface"), "surface.consumption")
    saving = _method_values(_exhibit_field(surface, :saving, "surface"),
        "surface.saving")
    tolerance = 128eps(max(last(cash), 1.0))
    for method in AR1_CONTEST_EXHIBIT_METHODS
        c = Matrix{Float64}(consumption[method])
        a = Matrix{Float64}(saving[method])
        size(c) == (length(cash), length(logs)) && size(a) == size(c) ||
            throw(DimensionMismatch("surface $method has the wrong shape"))
        all(isfinite, c) && all(isfinite, a) || throw(ArgumentError(
            "surface $method contains nonfinite values"))
        maximum(abs.(a .- (reshape(cash, :, 1) .- c))) <= tolerance ||
            throw(ArgumentError("surface saving is inconsistent for $method"))
        all(c .>= -tolerance) &&
            all(c .<= reshape(cash, :, 1) .+ tolerance) ||
            throw(ArgumentError("surface consumption is infeasible for $method"))
        minimum(diff(a; dims=1)) >= -1.0e-7 || throw(ArgumentError(
            "surface saving is not monotone for $method"))
    end
    _exhibit_field(surface, :outside_reported_timing, "surface") === true ||
        throw(ArgumentError("surface queries must be outside reported timing"))
    return nothing
end

function _validate_traces(traces, welfare, scientific)
    config = _config_from_scientific(scientific)
    path_ids = Int.(collect(_exhibit_field(traces, :path_ids, "traces")))
    dates = Int.(collect(_exhibit_field(traces, :dates, "traces")))
    declared_traces = _exhibit_field(
        scientific, :traces, "scientific_configuration")
    path_ids == Int.(collect(_exhibit_field(
        declared_traces, :path_ids, "trace config"))) ||
        throw(ArgumentError(
            "trace path ids disagree with the configured fixed paths"))
    dates == collect(1:config.horizon) || throw(ArgumentError(
        "trace dates must cover the complete horizon"))
    trace_paths = length(path_ids)
    income = Matrix{Float64}(_exhibit_field(traces, :income, "traces"))
    size(income) == (config.horizon, trace_paths) &&
        all(isfinite, income) && all(income .> 0.0) ||
        throw(DimensionMismatch("trace income has the wrong shape or values"))
    all(income[1, :] .== config.initial_income) || throw(ArgumentError(
        "trace income does not begin at initial income"))
    cash = _method_values(_exhibit_field(traces, :cash, "traces"), "traces.cash")
    consumption = _method_values(_exhibit_field(
        traces, :consumption, "traces"), "traces.consumption")
    trace_initial_cash = Float64.(collect(_exhibit_field(
        traces, :initial_cash, "traces")))
    welfare_initial_cash = Float64.(collect(_exhibit_field(
        welfare, :initial_cash_assignments, "welfare")))
    length(trace_initial_cash) == trace_paths &&
        trace_initial_cash == welfare_initial_cash[path_ids] ||
        throw(ArgumentError(
            "trace initial cash does not match the selected welfare paths"))
    for method in AR1_CONTEST_EXHIBIT_METHODS
        x = Matrix{Float64}(cash[method])
        c = Matrix{Float64}(consumption[method])
        size(x) == size(income) && size(c) == size(income) &&
            all(isfinite, x) && all(isfinite, c) || throw(DimensionMismatch(
                "trace arrays for $method have the wrong shape or values"))
        vec(x[1, :]) == trace_initial_cash && all(x .>= config.cash_floor) ||
            throw(ArgumentError("trace cash is invalid for $method"))
        all(c .>= 0.0) && all(c .<= x) || throw(ArgumentError(
            "trace consumption is infeasible for $method"))
    end
    innovation_rows = Matrix{Float64}(_exhibit_field(
        traces, :innovation_rows, "traces"))
    welfare_innovations = Matrix{Float64}(_exhibit_field(
        welfare, :innovations, "welfare"))
    innovation_rows == welfare_innovations[path_ids, :] ||
        throw(ArgumentError(
            "trace innovations are not the fixed first welfare paths"))
    expected_income = Matrix{Float64}(undef, config.horizon, trace_paths)
    expected_income[1, :] .= config.initial_income
    log_income = fill(log(config.initial_income), trace_paths)
    for date in 2:config.horizon
        log_income .= config.unconditional_log_mean .+
            config.rho .* (log_income .- config.unconditional_log_mean) .+
            config.sigma .* view(innovation_rows, :, date - 1)
        expected_income[date, :] .= exp.(log_income)
    end
    income == expected_income || throw(ArgumentError(
        "trace income does not reconstruct from the declared AR(1) process"))
    replayed = _method_values(_exhibit_field(
        traces, :replayed_utility, "traces"), "traces.replayed_utility")
    welfare_utility = Matrix{Float64}(_exhibit_field(welfare, :utility, "welfare"))
    for (column, method) in pairs(AR1_CONTEST_EXHIBIT_METHODS)
        Float64.(collect(replayed[method])) ==
            welfare_utility[path_ids, column] || throw(ArgumentError(
                "trace utility does not reproduce welfare for $method"))
    end
    _exhibit_field(traces, :outside_reported_timing, "traces") === true &&
        _exhibit_field(traces, :adds_welfare_observations, "traces") === false ||
        throw(ArgumentError("trace replay is misclassified as welfare evidence"))
    return nothing
end

"""
    _recorded_default_thread_ids(timing, julia_threads)

Validate stored pool counts and global thread IDs, independently of the loading
process's thread configuration. Artifacts predating pool metadata retain their
original `1:julia_threads` contract; their authenticated payload is unchanged.
"""
function _recorded_default_thread_ids(timing, julia_threads::Int)
    has_pools = timing isa AbstractDict ?
        (haskey(timing, :thread_pools) || haskey(timing, "thread_pools")) :
        hasproperty(timing, :thread_pools)
    has_pools || return 1:julia_threads

    pools = _exhibit_field(timing, :thread_pools, "timing")
    default_count = Int(_exhibit_field(
        pools, :default_threads, "timing.thread_pools"))
    interactive_count = Int(_exhibit_field(
        pools, :interactive_threads, "timing.thread_pools"))
    default_count == julia_threads && interactive_count >= 0 ||
        throw(ArgumentError("the recorded thread-pool counts are inconsistent"))
    default_ids = Int.(collect(_exhibit_field(
        pools, :default_thread_ids, "timing.thread_pools")))
    interactive_ids = Int.(collect(_exhibit_field(
        pools, :interactive_thread_ids, "timing.thread_pools")))
    expected_default_ids = (interactive_count + 1):(interactive_count + default_count)
    default_ids == collect(expected_default_ids) &&
        interactive_ids == collect(1:interactive_count) ||
        throw(ArgumentError("the recorded thread-pool IDs are inconsistent"))
    return expected_default_ids
end

function _validate_timing(timing, scientific; require_publication::Bool)
    config = _config_from_scientific(scientific)
    construction = _method_values(_exhibit_field(
        timing, :construction_seconds, "timing"), "construction_seconds")
    evaluation = _method_values(_exhibit_field(
        timing, :evaluation_seconds, "timing"), "evaluation_seconds")
    total = _method_values(_exhibit_field(
        timing, :end_to_end_seconds, "timing"), "end_to_end_seconds")
    for method in AR1_CONTEST_EXHIBIT_METHODS
        all(value -> value isa Real && isfinite(value) && value >= 0.0,
            (construction[method], evaluation[method], total[method])) ||
            throw(ArgumentError("timing values must be finite and nonnegative"))
        total[method] == construction[method] + evaluation[method] ||
            throw(ArgumentError("end-to-end timing is inconsistent for $method"))
    end
    all(flag -> _exhibit_field(timing, flag, "timing") === true,
        (:compilation_warmup_outside, :shock_generation_outside,
         :validation_outside, :io_outside, :surface_queries_outside,
         :trace_replay_outside, :no_timing_dispersion_estimate)) ||
        throw(ArgumentError("the timing exclusion contract is incomplete"))
    Int(_exhibit_field(timing, :timing_observations_per_method, "timing")) == 1 ||
        throw(ArgumentError("the diagnostic must report one timing observation"))
    julia_threads = Int(_exhibit_field(timing, :julia_threads, "timing"))
    configured_workers = Int(_exhibit_field(timing, :configured_workers, "timing"))
    julia_threads >= 1 || throw(ArgumentError(
        "the recorded Julia thread count must be positive"))
    default_thread_ids = _recorded_default_thread_ids(timing, julia_threads)
    configured_workers == config.workers || throw(ArgumentError(
        "the timing worker count disagrees with the configuration"))
    ranges = collect(_exhibit_field(timing, :partition_ranges, "timing"))
    length(ranges) == config.workers &&
        all(range -> range isa AbstractUnitRange{<:Integer}, ranges) ||
        throw(ArgumentError(
            "the timing partition count or representation is invalid"))
    normalized_ranges = [Int(first(range)):Int(last(range)) for range in ranges]
    normalized_ranges == _path_partitions(config.n_paths, config.workers) ||
        throw(ArgumentError(
            "the timing partitions do not cover the configured paths exactly"))
    thread_ids = _method_values(_exhibit_field(
        timing, :evaluation_boundary_thread_ids, "timing"),
        "evaluation_boundary_thread_ids")
    for method in AR1_CONTEST_EXHIBIT_METHODS
        ids = Int.(collect(thread_ids[method]))
        !isempty(ids) && all(id -> id in default_thread_ids, ids) ||
            throw(ArgumentError(
                "the recorded evaluation thread ids are invalid for $method"))
    end
    if require_publication
        Int(_exhibit_field(timing, :blas_threads, "timing")) == 1 ||
            throw(ArgumentError("the publication timing did not use BLAS=1"))
        if min(julia_threads, configured_workers) > 1
            length(thread_ids[:ptd_sceq]) > 1 && length(thread_ids[:ptd_osa]) > 1 ||
                throw(ArgumentError(
                    "the publication PTD evaluations did not use multiple threads"))
        end
    end
    return nothing
end

function _exhibit_validate_quadrature(quadrature, scientific)
    config = _config_from_scientific(scientific)
    declared = _exhibit_field(scientific, :quadrature,
        "scientific_configuration")
    Int(_exhibit_field(declared, :nodes_per_dimension, "quadrature config")) == 5 ||
        throw(ArgumentError("the scientific configuration is not scalar Q5"))
    _exhibit_field(declared, :ptd_sceq_uses_quadrature,
        "quadrature config") === false || throw(ArgumentError(
            "the scientific PTD-SCEQ quadrature contract is invalid"))
    _exhibit_field(declared, :ptd_osa_and_g2egm_share_exact_q5,
        "quadrature config") === true || throw(ArgumentError(
            "the scientific PTD-OSA/G2EGM quadrature contract is invalid"))

    ptd_sceq = _exhibit_field(quadrature, :ptd_sceq, "quadrature")
    ptd_osa = _exhibit_field(quadrature, :ptd_osa, "quadrature")
    g2 = _exhibit_field(quadrature, :g2egm, "quadrature")
    Symbol(_exhibit_field(ptd_sceq, :method,
        "quadrature.ptd_sceq")) === :ptd_sceq &&
        _exhibit_field(ptd_sceq, :uses_quadrature,
            "quadrature.ptd_sceq") === false || throw(ArgumentError(
                "PTD-SCEQ must not use numerical quadrature"))
    rule = gauss_hermite_rule(LogAR1Process(rho=config.rho,
        sigma=config.sigma,
        unconditional_log_mean=config.unconditional_log_mean), 5)
    expected_node_bits = _float_bits(rule.nodes)
    expected_weight_bits = _float_bits(rule.weights)
    for (method, audit) in ((:ptd_osa, ptd_osa), (:g2egm, g2))
        Symbol(_exhibit_field(audit, :method, "quadrature.$method")) === method &&
            _exhibit_field(audit, :uses_quadrature,
                "quadrature.$method") === true || throw(ArgumentError(
                    "$method has an invalid quadrature method contract"))
        Int(_exhibit_field(audit, :nodes_per_dimension,
            "quadrature.$method")) == 5 || throw(ArgumentError(
                "$method did not use scalar Q5"))
        _float_bits(_exhibit_field(audit, :base_nodes,
            "quadrature.$method")) == expected_node_bits ||
            throw(ArgumentError("$method Q5 nodes are not exact"))
        _float_bits(_exhibit_field(audit, :base_weights,
            "quadrature.$method")) == expected_weight_bits ||
            throw(ArgumentError("$method Q5 weights are not exact"))
    end
    _exhibit_field(quadrature, :exact_shared_q5, "quadrature") === true ||
        throw(ArgumentError("artifact does not record exact shared Q5"))
    Tuple(_exhibit_field(quadrature, :node_bits, "quadrature")) ==
        expected_node_bits || throw(ArgumentError(
            "the top-level Q5 node bits are invalid"))
    Tuple(_exhibit_field(quadrature, :weight_bits, "quadrature")) ==
        expected_weight_bits || throw(ArgumentError(
            "the top-level Q5 weight bits are invalid"))
    return nothing
end

"""Authenticate a saved or in-memory contest-exhibit artifact."""
function validate_ar1_contest_exhibit_artifact(artifact;
    expected_config::Union{Nothing,AR1ContestExhibitConfig}=nothing,
    require_publication::Bool=true)

    metadata = _exhibit_field(artifact, :metadata, "artifact")
    String(_exhibit_field(metadata, :artifact_schema, "metadata")) ==
        AR1_CONTEST_EXHIBIT_SCHEMA || throw(ArgumentError(
            "contest-exhibit artifact schema mismatch"))
    String(_exhibit_field(metadata, :authentication_schema, "metadata")) ==
        AR1_CONTEST_EXHIBIT_AUTHENTICATION_SCHEMA || throw(ArgumentError(
            "contest-exhibit authentication schema mismatch"))
    _exhibit_field(metadata, :status, "metadata") === :complete ||
        throw(ArgumentError("contest-exhibit artifact is incomplete"))
    completed = String(_exhibit_field(metadata, :completed_utc, "metadata"))
    !isempty(completed) && endswith(completed, 'Z') || throw(ArgumentError(
        "contest-exhibit completion timestamp is invalid"))
    scientific = _exhibit_field(artifact, :scientific_configuration, "artifact")
    scientific_fp = stable_fingerprint(scientific)
    scientific_fp == String(_exhibit_field(
        metadata, :scientific_fingerprint, "metadata")) ||
        throw(ArgumentError("scientific fingerprint mismatch"))
    if !isnothing(expected_config)
        scientific_fp == scientific_fingerprint(expected_config) ||
            throw(ArgumentError(
                "artifact scientific configuration does not match the request"))
    end
    if require_publication
        _is_publication_scientific_configuration(scientific) ||
            throw(ArgumentError("artifact is not a recognized paper exhibit"))
    end

    welfare = _exhibit_field(artifact, :welfare, "artifact")
    _validate_welfare(welfare, scientific)
    surface = _exhibit_field(artifact, :surface, "artifact")
    traces = _exhibit_field(artifact, :traces, "artifact")
    _validate_surface(surface, scientific)
    _validate_traces(traces, welfare, scientific)
    timing = _exhibit_field(artifact, :timing, "artifact")
    _validate_timing(timing, scientific; require_publication)
    quadrature = _exhibit_field(artifact, :quadrature, "artifact")
    _exhibit_validate_quadrature(quadrature, scientific)
    solver_diagnostics = _exhibit_field(
        artifact, :solver_diagnostics, "artifact")
    _exhibit_field(solver_diagnostics, :scope, "solver_diagnostics") ===
        :welfare_surface_and_trace_queries || throw(ArgumentError(
            "solver diagnostic scope is invalid"))
    _exhibit_field(solver_diagnostics, :display_queries_outside_reported_timing,
        "solver_diagnostics") === true || throw(ArgumentError(
            "display-query diagnostics are misclassified as timed work"))
    _validate_solver_diagnostics(_exhibit_field(
        solver_diagnostics, :final, "solver_diagnostics"))

    provenance = _exhibit_field(artifact, :provenance, "artifact")
    upstream = _exhibit_field(provenance, :upstream, "provenance")
    declared_solver = _exhibit_field(scientific, :g2egm, "scientific_configuration")
    String(_exhibit_field(upstream, :solver_schema, "provenance.upstream")) ==
        String(_exhibit_field(declared_solver, :solver_schema, "scientific_configuration.g2egm")) ||
        throw(ArgumentError("recorded solver schema disagrees with scientific configuration"))
    String(_exhibit_field(upstream, :commit, "provenance.upstream")) ==
        String(_exhibit_field(declared_solver, :upstream_commit, "scientific_configuration.g2egm")) ||
        throw(ArgumentError("recorded upstream commit disagrees with scientific configuration"))
    provenance_fingerprint = stable_fingerprint(provenance)
    provenance_fingerprint == String(_exhibit_field(
        metadata, :provenance_fingerprint, "metadata")) ||
        throw(ArgumentError("contest-exhibit provenance fingerprint mismatch"))
    input_tree = _exhibit_field(provenance, :input_tree, "provenance")
    stable_fingerprint(_exhibit_field(input_tree, :files, "input_tree")) ==
        String(_exhibit_field(input_tree, :fingerprint, "input_tree")) ||
        throw(ArgumentError("computational input-tree fingerprint mismatch"))
    result_payload = _result_payload(scientific_fp, quadrature,
        solver_diagnostics, welfare, timing, surface, traces)
    result_fingerprint = stable_fingerprint(result_payload)
    result_fingerprint == String(_exhibit_field(
        metadata, :result_fingerprint, "metadata")) ||
        throw(ArgumentError("contest-exhibit result fingerprint mismatch"))
    timing_fingerprint = stable_fingerprint(timing)
    timing_fingerprint == String(_exhibit_field(
        metadata, :timing_fingerprint, "metadata")) ||
        throw(ArgumentError("contest-exhibit timing fingerprint mismatch"))
    authentication = _artifact_authentication(
        scientific_fp, result_fingerprint, timing_fingerprint, provenance,
        completed)
    authentication == String(_exhibit_field(
        metadata, :authentication_fingerprint, "metadata")) ||
        throw(ArgumentError("contest-exhibit authentication mismatch"))
    return artifact
end

function _write_artifact(path::AbstractString, artifact; compress::Bool)
    JLD2.jldopen(path, "w"; compress) do file
        file["artifact"] = artifact
    end
    return path
end

"""Atomically publish an authenticated compact artifact and read it back."""
function save_ar1_contest_exhibit_artifact(path::AbstractString, artifact;
    compress::Bool=true, require_publication::Bool=true)
    validate_ar1_contest_exhibit_artifact(artifact; require_publication)
    destination = _validate_data_output(path)
    mkpath(dirname(destination))
    ispath(destination) && throw(ArgumentError(
        "refusing to overwrite an existing contest-exhibit artifact"))
    temporary = _temporary_jld_path(destination)
    try
        _write_artifact(temporary, artifact; compress)
        loaded = load_ar1_contest_exhibit_artifact(temporary;
            require_publication)
        ispath(destination) && throw(ArgumentError(
            "another process published the contest-exhibit artifact"))
        Base.Filesystem.rename(temporary, destination)
        return (; path=destination, artifact=loaded)
    finally
        ispath(temporary) && rm(temporary)
    end
end

"""Read and authenticate a compact artifact without invoking a solver."""
function load_ar1_contest_exhibit_artifact(path::AbstractString;
    expected_config::Union{Nothing,AR1ContestExhibitConfig}=nothing,
    require_publication::Bool=true,
    require_live_inputs::Bool=require_publication)
    isfile(path) || throw(ArgumentError(
        "contest-exhibit artifact does not exist: $path"))
    artifact = JLD2.jldopen(path, "r") do file
        haskey(file, "artifact") || throw(ArgumentError(
            "JLD2 file is missing the artifact payload"))
        file["artifact"]
    end
    validated = validate_ar1_contest_exhibit_artifact(artifact;
        expected_config, require_publication)
    if require_live_inputs
        provenance = _exhibit_field(validated, :provenance, "artifact")
        input_tree = _exhibit_field(provenance, :input_tree, "provenance")
        stored = String(_exhibit_field(input_tree, :fingerprint, "input_tree"))
        live = _exhibit_computational_input_tree().fingerprint
        stored == live || throw(ArgumentError(
            "artifact computational inputs differ from the live project tree"))
    end
    return validated
end

function _quarantine_artifact(path::AbstractString)
    stamp = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSSsss")
    directory = mktempdir(; prefix="ptd-invalid-artifact-" * stamp * "-",
        cleanup=false)
    destination = joinpath(directory, basename(path))
    mv(path, destination)
    return destination
end

function _exhibit_with_run_lock(action::F, output_path::AbstractString) where {F}
    lock_path = _runtime_lock_path(output_path, :ar1_exhibit)
    mkpath(dirname(lock_path))
    try
        mkdir(lock_path)
    catch error
        isdir(lock_path) || rethrow(error)
        throw(ArgumentError(
            "contest-exhibit lock already exists at $lock_path; inspect the owner record and remove the directory only after confirming no run is active"))
    end
    try
        open(joinpath(lock_path, "owner.txt"), "w") do io
            println(io, "pid=", getpid())
            println(io, "hostname=", Base.gethostname())
            println(io, "destination=", abspath(output_path))
            println(io, "started_utc=", _exhibit_utc_string(now(UTC)))
            println(io, "julia_threads=", Threads.nthreads())
        end
        return action()
    finally
        ispath(lock_path) && rm(lock_path; recursive=true)
    end
end

"""
Run or resume the paper's AR(1) contest exhibit. A valid existing artifact is
returned without solving; an invalid file is quarantined before recomputation.
"""
function run_ar1_contest_exhibits(
    config::AR1ContestExhibitConfig=
        production_ar1_contest_exhibit_config();
    output_path::AbstractString=
        default_ar1_contest_exhibit_artifact_path(config),
    warmup::Bool=true,
    require_publication::Bool=_is_publication_config(config),
    compress::Bool=true,
    lock_run::Bool=true)

    require_publication && !lock_run && throw(ArgumentError(
        "the paper exhibit requires its process-level lock"))
    action = function ()
        if isfile(output_path)
            try
                artifact = load_ar1_contest_exhibit_artifact(output_path;
                    expected_config=config, require_publication)
                return (; status=:loaded, path=abspath(output_path), artifact,
                    quarantined=nothing)
            catch error
                quarantined = _quarantine_artifact(output_path)
                @warn "Quarantined invalid contest-exhibit artifact" path=output_path quarantined=quarantined exception=(error, catch_backtrace())
                artifact = build_ar1_contest_exhibit_artifact(config;
                    warmup, require_publication)
                saved = save_ar1_contest_exhibit_artifact(output_path, artifact;
                    compress, require_publication)
                return (; status=:recomputed, path=saved.path,
                    artifact=saved.artifact, quarantined)
            end
        end
        artifact = build_ar1_contest_exhibit_artifact(config;
            warmup, require_publication)
        saved = save_ar1_contest_exhibit_artifact(output_path, artifact;
            compress, require_publication)
        return (; status=:computed, path=saved.path,
            artifact=saved.artifact, quarantined=nothing)
    end
    return lock_run ? _exhibit_with_run_lock(action, output_path) : action()
end
