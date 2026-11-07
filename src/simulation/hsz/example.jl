import ..Ptd: _validate_data_output, _temporary_jld_path
using Dates
using JLD2
using LinearAlgebra
using Random
using SHA
using Statistics
using DrWatson: datadir

import ..Ptd: G2EGM, HSZProcess, HSZState, PTDOSAOptions, ObservedState, PROJECT_ROOT,
              StochasticPTDConfig, build_hsz_contest_policy,
              gauss_hermite_rule, hsz_contest_quadrature_audit,
              income, next_observed, true_transition,
              validate_hsz_contest_policy_alignment

export HSZ_EXAMPLE_SCHEMA, HSZ_EXAMPLE_METHODS, HSZExampleConfig,
       production_hsz_example_config, default_hsz_example_artifact_path,
       warm_hsz_example!, run_hsz_example, load_hsz_example

const HSZ_EXAMPLE_SCHEMA = "ptd-hsz-example-v3"
const HSZ_EXAMPLE_METHODS = (:ptd_sceq, :ptd_osa, :g2egm)
const HSZ_EXAMPLE_SEED = UInt64(0x5054445f48535a31)

"""Scientific and numerical configuration for the paper's HSZ exercise."""
Base.@kwdef struct HSZExampleConfig
    beta::Float64 = 0.98
    gamma::Float64 = 1.0
    return_rate::Float64 = 0.10
    cash_floor::Float64 = 3.0
    horizon::Int = 50
    rho_earnings::Float64 = 0.9
    sigma_earnings::Float64 = 0.1
    rho_medical::Float64 = 0.9
    sigma_medical::Float64 = 0.1
    unconditional_log_earnings::Float64 = log(2.0)
    unconditional_log_medical::Float64 = log(1.5)
    initial_cash_support::Vector{Float64} =
        collect(range(5.5, 26.5; length=5))
    n_paths::Int = 100
    paths_per_initial_cash::Int = 20
    quadrature_nodes_per_dimension::Int = 5
    seed::UInt64 = HSZ_EXAMPLE_SEED
    asset_maximum::Float64 = 300.0
    asset_points::Int = 401
    common_cash_points::Int = 401
    post_earnings_points::Int = 17
    post_medical_points::Int = 17
    common_earnings_points::Int = 17
    common_medical_points::Int = 17
    state_radius::Float64 = 3.2
end

"""Return the locked configuration for the paper's HSZ exercise."""
production_hsz_example_config() = HSZExampleConfig()

function _validate(config::HSZExampleConfig)
    config.gamma == 1.0 || throw(ArgumentError(
        "the fixed HSZ example requires log utility"))
    config.rho_earnings == config.rho_medical == 0.9 || throw(ArgumentError(
        "both fixed HSZ persistence parameters must equal 0.9"))
    config.sigma_earnings == config.sigma_medical == 0.1 ||
        throw(ArgumentError(
            "both fixed HSZ innovation standard deviations must equal 0.1"))
    config.quadrature_nodes_per_dimension == 5 || throw(ArgumentError(
        "the HSZ example requires 5x5 quadrature"))
    config.n_paths == length(config.initial_cash_support) *
        config.paths_per_initial_cash || throw(ArgumentError(
            "the HSZ path allocation is not balanced"))
    config.horizon >= 2 && config.n_paths >= 1 || throw(ArgumentError(
        "the HSZ horizon/path counts are invalid"))
    return config
end

function _scientific_configuration(config::HSZExampleConfig)
    return (; schema=HSZ_EXAMPLE_SCHEMA, utility=:log,
        beta=config.beta, return_rate=config.return_rate,
        cash_floor=config.cash_floor, horizon=config.horizon,
        process=(rho_earnings=config.rho_earnings,
            sigma_earnings=config.sigma_earnings,
            rho_medical=config.rho_medical,
            sigma_medical=config.sigma_medical,
            unconditional_log_earnings=config.unconditional_log_earnings,
            unconditional_log_medical=config.unconditional_log_medical,
            net_income=:earnings_minus_medical_expense),
        quadrature=(nodes_per_dimension=5, dimensions=2, total_nodes=25,
            ptd_osa_and_g2egm_share_rule=true,
            ptd_sceq_uses_quadrature=false),
        paths=(count=config.n_paths,
            initial_cash_support=copy(config.initial_cash_support),
            paths_per_initial_cash=config.paths_per_initial_cash,
            common_shocks=true, seed=config.seed),
        g2egm=(geometry=:six_tetrahedra_per_tensor_cell,
            pure_julia=true, upstream_commit=G2EGM.G2EGM_REFERENCE_COMMIT,
            asset_maximum=config.asset_maximum,
            asset_points=config.asset_points,
            common_cash_points=config.common_cash_points,
            post_earnings_points=config.post_earnings_points,
            post_medical_points=config.post_medical_points,
            common_earnings_points=config.common_earnings_points,
            common_medical_points=config.common_medical_points,
            postdecision_total_points=config.asset_points *
                config.post_earnings_points * config.post_medical_points,
            common_total_points=config.common_cash_points *
                config.common_earnings_points * config.common_medical_points,
            state_radius=config.state_radius))
end

function _scientific_fingerprint(config::HSZExampleConfig)
    return bytes2hex(SHA.sha256(repr(_scientific_configuration(config))))
end

"""Return the HSZ example artifact path."""
function default_hsz_example_artifact_path(
    config::HSZExampleConfig=production_hsz_example_config();
    root::AbstractString=joinpath(PROJECT_ROOT, "data"))
    _validate(config)
    return joinpath(abspath(root), "hsz-example-medical-1p5.jld2")
end

function _model(config)
    ptd_config = StochasticPTDConfig(beta=config.beta, gamma=config.gamma,
        return_rate=config.return_rate, cash_floor=config.cash_floor,
        terminal_date=config.horizon)
    process = HSZProcess(rho_earnings=config.rho_earnings,
        sigma_earnings=config.sigma_earnings,
        rho_medical=config.rho_medical,
        sigma_medical=config.sigma_medical,
        unconditional_log_earnings=config.unconditional_log_earnings,
        unconditional_log_medical=config.unconditional_log_medical)
    return ptd_config, process
end

function _g2_grid(config, process)
    return G2EGM.hsz_g2egm_grid(process;
        asset_maximum=config.asset_maximum,
        asset_points=config.asset_points,
        common_cash_points=config.common_cash_points,
        post_earnings_points=config.post_earnings_points,
        post_medical_points=config.post_medical_points,
        common_earnings_points=config.common_earnings_points,
        common_medical_points=config.common_medical_points,
        earnings_radius=config.state_radius,
        medical_radius=config.state_radius,
        cash_floor=config.cash_floor,
        gross_return=1.0 + config.return_rate)
end

function _initial_cash(config)
    return reduce(vcat, (fill(cash, config.paths_per_initial_cash)
        for cash in config.initial_cash_support); init=Float64[])
end

function _common_exogenous_paths(config, process, innovations)
    earnings = Matrix{Float64}(undef, config.horizon, config.n_paths)
    medical = similar(earnings)
    net_income = similar(earnings)
    earnings[1, :] .= config.unconditional_log_earnings
    medical[1, :] .= config.unconditional_log_medical
    @inbounds for path in 1:config.n_paths
        net_income[1, path] = exp(earnings[1, path]) - exp(medical[1, path])
        for date in 1:config.horizon-1
            earnings[date + 1, path] = config.unconditional_log_earnings +
                config.rho_earnings *
                (earnings[date, path] - config.unconditional_log_earnings) +
                config.sigma_earnings * innovations[path, date, 1]
            medical[date + 1, path] = config.unconditional_log_medical +
                config.rho_medical *
                (medical[date, path] - config.unconditional_log_medical) +
                config.sigma_medical * innovations[path, date, 2]
            net_income[date + 1, path] = exp(earnings[date + 1, path]) -
                exp(medical[date + 1, path])
        end
    end
    return (; log_earnings=earnings, log_medical=medical, net_income)
end

function _build_worker_policies(method, config, process, rule, grid)
    count = Threads.nthreads()
    if method === :g2egm
        policy = build_hsz_contest_policy(:g2egm, config, process;
            base_rule=rule, g2egm_grid=grid)
        return fill(policy, count)
    end
    return [build_hsz_contest_policy(method, config, process;
        base_rule=rule, ptd_osa_options=PTDOSAOptions()) for _ in 1:count]
end

function _evaluate_method(method, policies, config, process,
    innovations, initial_cash)
    horizon, n_paths = config.terminal.final_date, length(initial_cash)
    cash = Matrix{Float64}(undef, horizon, n_paths)
    consumption = similar(cash)
    utility = zeros(Float64, n_paths)
    floor_hits = zeros(Int, n_paths)
    transfers = zeros(Float64, n_paths)
    initial_state = HSZState(process.unconditional_log_earnings,
        process.unconditional_log_medical)
    # Logical workers own their policy and disjoint path columns. Global thread
    # IDs need not start at one within the default pool (interactive threads
    # come first), and a task may migrate without changing this ownership.
    Threads.@threads for worker in eachindex(policies)
        policy = policies[worker]
        for path in worker:length(policies):n_paths
            observed = ObservedState(1, initial_cash[path], initial_state)
            discount = 1.0
            for date in 1:horizon
                action = policy(observed)
                action.method == method || error("HSZ policy method mismatch")
                cash[date, path] = observed.cash
                consumption[date, path] = action.consumption
                utility[path] += discount * log(action.consumption)
                date == horizon && continue
                transition = true_transition(config, process, observed,
                    action.consumption,
                    (innovations[path, date, 1], innovations[path, date, 2]))
                floor_hits[path] += transition.transfer > 0.0
                transfers[path] += transition.transfer
                observed = next_observed(observed, transition)
                discount *= config.beta
            end
        end
    end
    return (; cash, consumption, utility, floor_hits,
            total_transfers=transfers)
end

"""Compile every HSZ construction and replay path outside reported timers."""
function warm_hsz_example!(config::HSZExampleConfig=
    production_hsz_example_config())
    _validate(config)
    full_model, process = _model(config)
    rule = gauss_hermite_rule(process, config.quadrature_nodes_per_dimension)
    full_innovations = zeros(Float64, 1, config.horizon - 1, 2)
    full_initial_cash = [first(config.initial_cash_support)]
    for method in (:ptd_sceq, :ptd_osa)
        policies = _build_worker_policies(
            method, full_model, process, rule, nothing)
        _evaluate_method(method, policies, full_model, process,
            full_innovations, full_initial_cash)
    end

    tiny_model = StochasticPTDConfig(beta=config.beta, gamma=config.gamma,
        return_rate=config.return_rate, cash_floor=config.cash_floor,
        terminal_date=3)
    warm_grid = G2EGM.hsz_g2egm_grid(process;
        asset_maximum=30.0, asset_points=9, common_cash_points=9,
        post_earnings_points=5, post_medical_points=5,
        common_earnings_points=5, common_medical_points=5,
        earnings_radius=config.state_radius,
        medical_radius=config.state_radius,
        cash_floor=config.cash_floor,
        gross_return=1.0 + config.return_rate)
    policies = _build_worker_policies(
        :g2egm, tiny_model, process, rule, warm_grid)
    _evaluate_method(:g2egm, policies, tiny_model, process,
        zeros(Float64, 1, 2, 2), full_initial_cash)
    return nothing
end

"""Solve all three policies and evaluate the fixed common-shock path replay."""
function run_hsz_example(config::HSZExampleConfig=production_hsz_example_config();
    output_path::AbstractString=default_hsz_example_artifact_path(config))
    _validate_data_output(output_path)
    _validate(config)
    warm_hsz_example!(config)
    ptd_config, process = _model(config)
    rule = gauss_hermite_rule(process, config.quadrature_nodes_per_dimension)
    length(rule.weights) == 25 || error("internal HSZ Q25 failure")
    grid = _g2_grid(config, process)
    initial_cash = _initial_cash(config)
    innovations = randn(Xoshiro(config.seed), config.n_paths,
        config.horizon - 1, 2)
    exogenous = _common_exogenous_paths(config, process, innovations)

    construction = Dict{Symbol,Float64}()
    evaluation = Dict{Symbol,Float64}()
    policies = Dict{Symbol,Any}()
    results = Dict{Symbol,Any}()
    for method in HSZ_EXAMPLE_METHODS
        started = time_ns()
        method_policies = _build_worker_policies(
            method, ptd_config, process, rule, grid)
        construction[method] = (time_ns() - started) / 1.0e9
        policies[method] = method_policies
        validate_hsz_contest_policy_alignment(first(method_policies),
            method, ptd_config, process)
        started = time_ns()
        results[method] = _evaluate_method(method, method_policies,
            ptd_config, process, innovations, initial_cash)
        evaluation[method] = (time_ns() - started) / 1.0e9
    end
    g2_policy = first(policies[:g2egm])
    artifact = (; schema=HSZ_EXAMPLE_SCHEMA,
        created_at=string(Dates.now(Dates.UTC)),
        solver_schema=G2EGM.HSZ_G2EGM_SOLVER_SCHEMA,
        source_files=_utility_source_files(),
        scientific_configuration=_scientific_configuration(config),
        scientific_fingerprint=_scientific_fingerprint(config),
        julia=(version=string(VERSION), threads=Threads.nthreads(),
            blas_threads=BLAS.get_num_threads()),
        quadrature=(ptd_sceq=hsz_contest_quadrature_audit(
                first(policies[:ptd_sceq])),
            ptd_osa=hsz_contest_quadrature_audit(first(policies[:ptd_osa])),
            g2egm=G2EGM.quadrature_audit(g2_policy.solution)),
        timing=(construction_seconds=(;
                (method => construction[method]
                 for method in HSZ_EXAMPLE_METHODS)...),
            evaluation_seconds=(;
                (method => evaluation[method]
                 for method in HSZ_EXAMPLE_METHODS)...),
            end_to_end_seconds=(;
                (method => construction[method] + evaluation[method]
                 for method in HSZ_EXAMPLE_METHODS)...),
            compilation_warmup_outside=true),
        initial_cash, innovations, exogenous,
        paths=(; (method => results[method]
                  for method in HSZ_EXAMPLE_METHODS)...),
        g2egm=(solution_fingerprint=g2_policy.policy_fingerprint,
            diagnostics=G2EGM.diagnostics(g2_policy.solution),
            grid=(asset_points=length(grid.asset),
                common_cash_points=length(grid.cash),
                post_earnings_points=length(grid.post_log_earnings),
                post_medical_points=length(grid.post_log_medical),
                common_earnings_points=length(grid.log_earnings),
                common_medical_points=length(grid.log_medical),
                postdecision_total_points=length(grid.asset) *
                    length(grid.post_log_earnings) *
                    length(grid.post_log_medical),
                common_total_points=length(grid.cash) *
                    length(grid.log_earnings) * length(grid.log_medical))))
    path = abspath(output_path)
    mkpath(dirname(path))
    jldsave(path; artifact)
    return (; path, artifact)
end

"""Load and validate a completed HSZ paper artifact."""
function load_hsz_example(path::AbstractString)
    payload = JLD2.load(abspath(path))
    haskey(payload, "artifact") || throw(ArgumentError(
        "HSZ example artifact is missing its payload"))
    artifact = payload["artifact"]
    artifact.schema == HSZ_EXAMPLE_SCHEMA || throw(ArgumentError(
        "HSZ example artifact schema mismatch"))
    return artifact
end
