# Tests for src/g2egm/ar1_model.jl and src/g2egm/ar1_solver.jl (file 15).
module AR1G2EGMSolverTests

using Test
import Ptd

const G2 = Ptd.G2EGM
const BASE_RHO = 0.9
const BASE_SIGMA = 0.1
const INITIAL_LOG_INCOME = 0.0

function _utility(gamma::Float64, consumption::Float64)
    consumption > 0.0 || return -Inf
    return gamma == 1.0 ? log(consumption) :
           (consumption^(1.0 - gamma) - 1.0) / (1.0 - gamma)
end

function _two_period_objective(config, process, rule, cash, log_income, saving)
    0.0 <= saving <= cash || return -Inf
    current = _utility(config.gamma, cash - saving)
    isfinite(current) || return current
    gross = 1.0 + config.return_rate
    mean = process.unconditional_log_mean
    continuation = 0.0
    @inbounds for q in eachindex(rule.weights)
        next_log_income = mean + process.rho * (log_income - mean) +
                          process.log_std * only(rule.nodes[q])
        next_cash = max(
            gross * saving + exp(next_log_income), config.cash_floor)
        continuation += rule.weights[q] * _utility(config.gamma, next_cash)
    end
    return current + config.beta * continuation
end

function _golden_maximum(objective, lower::Float64, upper::Float64;
    iterations::Int=120)
    upper >= lower || throw(ArgumentError("golden-section bounds are reversed"))
    upper == lower && return (saving=lower, value=objective(lower))
    ratio = (sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)
    for _ in 1:iterations
        if left_value < right_value
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
        else
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        end
    end
    candidates = (lower, left, right, upper)
    values = objective.(candidates)
    index = argmax(values)
    return (saving=candidates[index], value=values[index])
end

"""Independent dense screen plus regime-by-regime golden Bellman search."""
function _dense_golden_reference(config, process, rule, cash, log_income;
    dense_points::Int=10_001, asset_maximum::Real=cash)
    gross = 1.0 + config.return_rate
    mean = process.unconditional_log_mean
    upper = min(cash, Float64(asset_maximum))
    0.0 < upper <= cash || throw(ArgumentError(
        "the direct-reference asset bound must lie in (0,cash]"))
    objective = saving -> _two_period_objective(
        config, process, rule, cash, log_income, saving)

    # The exact finite-Q5 transfer kinks split the action interval into smooth,
    # concave pieces. This calculation is independent of the G2EGM routines.
    kinks = Float64[]
    for node in rule.nodes
        next_log_income = mean + process.rho * (log_income - mean) +
                          process.log_std * only(node)
        saving = (config.cash_floor - exp(next_log_income)) / gross
        0.0 < saving < upper && push!(kinks, saving)
    end
    sort!(unique!(kinks))
    boundaries = [0.0; kinks; upper]

    dense_savings = range(0.0, upper; length=dense_points)
    dense_values = objective.(dense_savings)
    dense_index = argmax(dense_values)
    candidate_savings = Float64[dense_savings[dense_index]]
    candidate_values = Float64[dense_values[dense_index]]
    append!(candidate_savings, boundaries)
    append!(candidate_values, objective.(boundaries))
    for interval in 1:length(boundaries)-1
        refined = _golden_maximum(
            objective, boundaries[interval], boundaries[interval + 1])
        push!(candidate_savings, refined.saving)
        push!(candidate_values, refined.value)
    end
    best_value = maximum(candidate_values)
    tied = findall(value -> isapprox(value, best_value;
        atol=2e-13, rtol=2e-13), candidate_values)
    # Match the solver's deterministic welfare-tie rule: save more.
    best_index = tied[argmax(candidate_savings[tied])]
    return (saving=candidate_savings[best_index], value=candidate_values[best_index],
            dense_saving=Float64(dense_savings[dense_index]),
            dense_value=dense_values[dense_index], kinks)
end

function _baseline_inputs(; terminal_date::Int=2)
    config = Ptd.StochasticPTDConfig(
        beta=0.96,
        gamma=1.0,
        return_rate=0.10,
        cash_floor=3.0,
        terminal_date=terminal_date,
    )
    process = Ptd.LogAR1Process(
        rho=BASE_RHO,
        sigma=BASE_SIGMA,
        unconditional_log_mean=0.0,
    )
    rule = Ptd.gauss_hermite_rule(process, 5)
    grid = G2.ar1_g2egm_grid(
        asset_maximum=20.0,
        asset_points=201,
        common_cash_points=241,
        post_log_income_points=31,
        common_log_income_points=25,
        cash_floor=config.cash_floor,
        gross_return=1.0 + config.return_rate,
        log_income_bounds=(-3.0, 3.0),
    )
    return (; config, process, rule, grid)
end

@testset "AR(1) pure-Julia G2EGM solver" begin
    baseline = _baseline_inputs()

    @testset "declared grids, process support, and fixed Q5 rule" begin
        (; config, process, rule, grid) = baseline
        @test BASE_RHO == 0.9
        @test BASE_SIGMA == 0.1
        @test exp(INITIAL_LOG_INCOME) == 1.0
        @test first(grid.asset) == 0.0
        @test first(grid.cash) == config.cash_floor
        @test extrema(grid.post_log_income) == (-3.0, 3.0)
        @test extrema(grid.log_income) == (-3.0, 3.0)
        @test last(grid.cash) ≈
            (1.0 + config.return_rate) * last(grid.asset) +
            exp(last(grid.post_log_income))
        @test all(diff(grid.asset) .> 0.0)
        @test all(diff(grid.cash) .> 0.0)
        @test all(diff(grid.post_log_income) .> 0.0)
        @test all(diff(grid.log_income) .> 0.0)

        @test rule.nodes_per_dimension == 5
        @test length(rule.nodes) == length(rule.weights) == 5
        @test all(>(0.0), rule.weights)
        @test sum(rule.weights) ≈ 1.0 atol=8eps(Float64)
        repeated_rule = Ptd.gauss_hermite_rule(process, 5)
        @test repeated_rule.nodes == rule.nodes
        @test repeated_rule.weights == rule.weights

        next_support = [process.unconditional_log_mean +
            process.rho * (ell - process.unconditional_log_mean) +
            process.log_std * only(node)
            for ell in grid.post_log_income, node in rule.nodes]
        lower_margin = minimum(next_support) - first(grid.log_income)
        upper_margin = last(grid.log_income) - maximum(next_support)
        expected_margin = 3.0 * (1.0 - process.rho) -
                          process.log_std * maximum(only.(rule.nodes))
        @test lower_margin ≈ expected_margin atol=2e-15
        @test upper_margin ≈ expected_margin atol=2e-15
        @test lower_margin > 0.0
        @test upper_margin > 0.0

        @test_throws ArgumentError G2.AR1G2EGMGrid(
            [0.1, 1.0], [-3.0, 3.0], [3.0, 4.0], [-3.0, 3.0])
        @test_throws ArgumentError G2.AR1G2EGMGrid(
            [0.0, 1.0], [-3.0, 3.0], [3.0, 4.0], [-2.0, 3.0])
        narrow_grid = G2.AR1G2EGMGrid(
            grid.asset, range(-0.5, 0.5; length=11), grid.cash,
            range(-0.5, 0.5; length=9))
        @test_throws ArgumentError G2.build_ar1_solution(
            config, process; base_rule=rule, grid=narrow_grid)
        @test_throws ArgumentError G2.build_ar1_solution(
            config, process;
            base_rule=Ptd.gauss_hermite_rule(process, 3), grid=grid)
    end

    terminal_inputs = _baseline_inputs(terminal_date=1)
    terminal_solution = G2.build_ar1_solution(
        terminal_inputs.config, terminal_inputs.process;
        base_rule=terminal_inputs.rule, grid=terminal_inputs.grid)

    @testset "one-period terminal problem" begin
        (; config, grid) = terminal_inputs
        @test length(terminal_solution.policy) == 1
        @test length(terminal_solution.value) == 1
        @test terminal_solution.policy[1] == repeat(
            reshape(grid.cash, :, 1), 1, length(grid.log_income))
        expected = [_utility(config.gamma, cash)
                    for cash in grid.cash, _ in grid.log_income]
        @test terminal_solution.value[1] ≈ expected atol=0.0 rtol=0.0
        for (cash, ell) in ((3.0, -3.0), (7.25, 0.0),
                            (last(grid.cash), 3.0))
            @test G2.consumption(
                terminal_solution, cash, ell; date=1) == cash
            @test G2.value_at(
                terminal_solution, cash, ell; date=1) == log(cash)
            observed = Ptd.ObservedState(1, cash, Ptd.LogAR1State(ell))
            @test G2.action(terminal_solution, observed) == cash
        end
        solve_audit = G2.diagnostics(terminal_solution).solve
        @test solve_audit.quadrature_expectations == 0
        @test solve_audit.continuation_queries == 0
        @test solve_audit.triangles_considered == 0
    end

    baseline_solution = G2.build_ar1_solution(
        baseline.config, baseline.process;
        base_rule=baseline.rule, grid=baseline.grid)

    @testset "two-period independent dense-plus-golden Bellman check" begin
        (; config, process, rule, grid) = baseline
        # These are exact common-grid states and include both the borrowing
        # constraint and smooth interior regimes.
        cash_indices = (128, 148, 178)
        log_income_indices = (11, 13, 15)
        for j in log_income_indices, i in cash_indices
            cash = grid.cash[i]
            ell = grid.log_income[j]
            reference = _dense_golden_reference(
                config, process, rule, cash, ell)
            consumption = G2.consumption(
                baseline_solution, cash, ell; date=1)
            saving = cash - consumption
            achieved = _two_period_objective(
                config, process, rule, cash, ell, saving)

            @test reference.value >= reference.dense_value - 2e-13
            @test saving ≈ reference.saving atol=1e-3 rtol=0.0
            @test reference.value - achieved <= 1e-7
            @test isapprox(G2.value_at(
                    baseline_solution, cash, ell; date=1), reference.value;
                atol=1e-4, rtol=0.0)
        end
    end

    @testset "upper asset constraint controls continuation padding" begin
        config = Ptd.StochasticPTDConfig(
            beta=0.995, gamma=0.5, return_rate=0.10,
            cash_floor=3.0, terminal_date=2)
        process = Ptd.LogAR1Process(
            rho=BASE_RHO, sigma=BASE_SIGMA,
            unconditional_log_mean=0.0)
        rule = Ptd.gauss_hermite_rule(process, 5)
        grid = G2.ar1_g2egm_grid(
            asset_maximum=20.0,
            asset_points=81,
            common_cash_points=101,
            post_log_income_points=15,
            common_log_income_points=11,
            cash_floor=config.cash_floor,
            gross_return=1.0 + config.return_rate,
            log_income_bounds=(-3.0, 3.0),
        )
        solution = G2.build_ar1_solution(
            config, process; base_rule=rule, grid=grid)
        cash_index = lastindex(grid.cash)
        income_index = (length(grid.log_income) + 1) ÷ 2
        cash = grid.cash[cash_index]
        log_income = grid.log_income[income_index]
        asset_maximum = last(grid.asset)
        @test cash > asset_maximum

        reference = _dense_golden_reference(
            config, process, rule, cash, log_income;
            asset_maximum)
        consumption = solution.policy[1][cash_index, income_index]
        saving = cash - consumption
        achieved = _two_period_objective(
            config, process, rule, cash, log_income, saving)
        stored_value = solution.value[1][cash_index, income_index]
        direct_diagnostics = G2.G2EGMSolveDiagnostics()
        direct = G2._direct_global_action(
            config, process, rule, grid, solution.options,
            solution.value[2], solution.policy[2], cash, log_income,
            direct_diagnostics)

        @test reference.saving == asset_maximum
        @test direct.saving == asset_maximum
        @test saving ≈ direct.saving atol=64eps(cash) rtol=0.0
        @test stored_value ≈ direct.value atol=8eps(abs(direct.value)) rtol=0.0
        @test reference.value - achieved <= 1e-10

        audit = G2.diagnostics(solution).solve
        @test audit.asset_ceiling_candidates > 0
        @test audit.asset_ceiling_winners > 0
        @test audit.policy_domain_asset_ceiling_winners == 0
        @test audit.continuation_padding_asset_ceiling_winners > 0
        @test audit.asset_ceiling_winners ==
            audit.policy_domain_asset_ceiling_winners +
            audit.continuation_padding_asset_ceiling_winners
    end

    @testset "feasibility, borrowing constraint, and exact floor kinks" begin
        (; config, process, rule, grid) = baseline
        @test !G2._prefer_g2egm_candidate(
            -Inf, 0.0, 1.0, 1.0, baseline_solution.options)
        @test G2._prefer_g2egm_candidate(
            1.0, 1.0, -Inf, 0.0, baseline_solution.options)
        policy = baseline_solution.policy[1]
        cash = repeat(reshape(grid.cash, :, 1), 1, length(grid.log_income))
        saving = cash .- policy
        lower_tolerance = 64eps.(max.(cash, 1.0))
        @test all(policy .> 0.0)
        @test all(policy .<= cash)
        @test all(saving .>= -lower_tolerance)
        @test all(saving .<= last(grid.asset) + 64eps(last(grid.asset)))
        @test all(@view(policy[1, :]) .== config.cash_floor)

        ell = grid.log_income[21]
        independent_kinks = Float64[]
        for node in rule.nodes
            next_log = process.unconditional_log_mean +
                process.rho * (ell - process.unconditional_log_mean) +
                process.log_std * only(node)
            saving = (config.cash_floor - exp(next_log)) /
                     (1.0 + config.return_rate)
            saving > 0.0 && push!(independent_kinks, saving)
        end
        sort!(unique!(independent_kinks))
        @test length(independent_kinks) == 2
        for saving in independent_kinks
            residuals = [abs((1.0 + config.return_rate) * saving +
                exp(process.unconditional_log_mean +
                    process.rho * (ell - process.unconditional_log_mean) +
                    process.log_std * only(node)) - config.cash_floor)
                for node in rule.nodes]
            @test minimum(residuals) <= 8eps(config.cash_floor)
        end
        # Kinks are evaluated exactly even when the reconstructed triangle or a
        # direct monotonicity repair ultimately wins the Bellman comparison.
        cash = grid.cash[55]
        selected_saving = cash - G2.consumption(
            baseline_solution, cash, ell; date=1)
        selected_value = _two_period_objective(
            config, process, rule, cash, ell, selected_saving)
        kink_values = [_two_period_objective(
            config, process, rule, cash, ell, saving)
            for saving in independent_kinks]
        @test selected_value >= maximum(kink_values) - 1e-9

        solve_audit = G2.diagnostics(baseline_solution).solve
        @test solve_audit.constraint_winners > 0
        @test solve_audit.triangle_winners > 0
        @test solve_audit.floor_kink_candidates > 0
        @test solve_audit.final_q5_candidate_revaluations > 0
        @test solve_audit.triangles_mixed_floor_regime > 0
        @test solve_audit.monotonicity_violations_remaining == 0

        policy_last = searchsortedlast(grid.cash, last(grid.asset))
        for j in eachindex(grid.log_income)
            audited_saving = grid.cash[1:policy_last] .-
                policy[1:policy_last, j]
            tolerance = baseline_solution.options.monotonicity_atol .+
                baseline_solution.options.monotonicity_rtol .* max.(
                    abs.(audited_saving[1:end-1]),
                    abs.(audited_saving[2:end]),
                    grid.cash[2:policy_last], 1.0)
            @test all(diff(audited_saving) .>= -tolerance)
        end
    end

    @testset "zero-innovation state invariance" begin
        config = Ptd.StochasticPTDConfig(
            beta=0.96, gamma=2.0, return_rate=0.10,
            cash_floor=3.0, terminal_date=2)
        # rho=0 makes the next conditional law independent of today's ell;
        # sigma=0 then collapses all five quadrature branches to that same law.
        process = Ptd.LogAR1Process(
            rho=0.0, sigma=0.0, unconditional_log_mean=0.0)
        rule = Ptd.gauss_hermite_rule(process, 5)
        grid = G2.ar1_g2egm_grid(
            asset_maximum=10.0,
            asset_points=81,
            common_cash_points=101,
            post_log_income_points=15,
            common_log_income_points=11,
            cash_floor=config.cash_floor,
            gross_return=1.0 + config.return_rate,
            log_income_bounds=(-3.0, 3.0),
        )
        solution = G2.build_ar1_solution(
            config, process; base_rule=rule, grid=grid)
        middle = cld(length(grid.log_income), 2)
        repeated_policy = repeat(
            solution.policy[1][:, middle:middle], 1, length(grid.log_income))
        repeated_value = repeat(
            solution.value[1][:, middle:middle], 1, length(grid.log_income))
        @test isapprox(solution.policy[1], repeated_policy;
            atol=8eps(last(grid.cash)), rtol=0.0)
        @test isapprox(solution.value[1], repeated_value;
            atol=8eps(maximum(abs, solution.value[1])), rtol=0.0)
        for ell in grid.log_income
            @test G2.consumption(
                solution, grid.cash[60], ell; date=1) ≈
                solution.policy[1][60, middle] atol=8eps(grid.cash[60]) rtol=0.0
        end
    end

    @testset "direct repair finds a peak hidden between asset nodes" begin
        config = Ptd.StochasticPTDConfig(beta=0.96, gamma=1.0,
            return_rate=0.0, cash_floor=1.0, terminal_date=2)
        process = Ptd.LogAR1Process(rho=0.0, sigma=0.0,
            unconditional_log_mean=0.0)
        rule = Ptd.gauss_hermite_rule(process, 5)
        grid = G2.AR1G2EGMGrid(collect(0.0:1.0:5.0), [-1.0, 1.0],
            [1.0, 2.0, 2.39, 2.4, 2.41, 3.0, 4.0, 5.0, 8.0], [-1.0, 1.0])
        # A triangular child-value peak at z=2.4 maps to saving a=1.4.
        # Every coarse asset node misses it. The old discrete-local-maximum
        # screen would only search [0,1], where the objective is decreasing.
        bump = cash -> 4.0 * max(1.0 - abs(cash - 2.4) / 0.01, 0.0)
        child_value = [bump(cash) for cash in grid.cash, _ in grid.log_income]
        child_policy = repeat(reshape(grid.cash, :, 1), 1, length(grid.log_income))
        options = G2.AR1G2EGMOptions()
        cash = 6.0
        independent = saving -> log(cash - saving) + config.beta * bump(saving + 1.0)
        screened_values = independent.(grid.asset)
        @test argmax(screened_values) == 1
        old_refinement = _golden_maximum(independent, 0.0, 1.0)
        direct = G2._direct_global_action(config, process, rule, grid, options,
            child_value, child_policy, cash, 0.0, G2.G2EGMSolveDiagnostics())
        @test direct isa G2.DirectActionResult
        @test direct.saving ≈ 1.4 atol=2e-15 rtol=0.0
        @test direct.value ≈ independent(1.4) atol=3e-14 rtol=0.0
        @test direct.value > old_refinement.value + 3.0
        @test 0.0 <= direct.saving <= min(cash, last(grid.asset))

        # Deliberately introduce a decreasing saving pair into a feasible
        # reconstructed policy. Direct repairs recover the same peak at both
        # states and may propagate to adjacent nodes, but solve each at most once.
        policy = copy(child_policy)
        for j in axes(policy, 2), i in axes(policy, 1)
            policy[i, j] = G2._direct_global_action(config, process, rule, grid,
                options, child_value, child_policy, grid.cash[i], grid.log_income[j],
                G2.G2EGMSolveDiagnostics()).consumption
        end
        policy[2, :] .= grid.cash[2] - 1.5
        policy[3, :] .= grid.cash[3] - 0.3
        repaired_value = zeros(size(policy))
        sources = zeros(UInt8, size(policy))
        audit = G2.G2EGMSolveDiagnostics()
        G2._repair_monotone_saving!(policy, repaired_value, sources,
            config, process, rule, grid, options, child_value, child_policy, audit)
        last_audited = searchsortedlast(grid.cash, last(grid.asset))
        for j in axes(policy, 2)
            saving = grid.cash[1:last_audited] - policy[1:last_audited, j]
            @test all(diff(saving) .>= -1e-12)
            @test saving[2] ≈ 1.4 atol=2e-15
            @test saving[3] ≈ 1.4 atol=2e-15
        end
        @test 4 <= audit.monotonicity_repairs <= last_audited * length(grid.log_income)
        @test audit.monotonicity_violations_remaining == 0
    end

    @testset "flat utility repair uses analytic saving, not welfare near-ties" begin
        config = Ptd.StochasticPTDConfig(beta=0.96, gamma=3.5,
            return_rate=0.0, cash_floor=1.0, terminal_date=2)
        process = Ptd.LogAR1Process(rho=0.0, sigma=0.0,
            unconditional_log_mean=0.0)
        rule = Ptd.gauss_hermite_rule(process, 5)
        grid = G2.AR1G2EGMGrid([0.0, 5_000.0, 10_000.0, 50_000.0],
            [-1.0, 1.0], [1.0, 100_000.0], [-1.0, 1.0])
        child_value = [1_000.0 1_000.0; 1_000.0 + 2e-9 1_000.0 + 2e-9]
        child_policy = repeat(reshape(grid.cash, :, 1), 1, 2)
        # Deliberately large triangle-screen tolerances must not relax direct
        # maximization or the unchanged saving-monotonicity certificate.
        options = G2.AR1G2EGMOptions(value_atol=1.0, value_rtol=1.0)
        optimal_consumption = setprecision(BigFloat, 256) do
            slope = (BigFloat(child_value[2, 1]) - BigFloat(child_value[1, 1])) /
                    (BigFloat(grid.cash[2]) - BigFloat(grid.cash[1]))
            # Retain the actual quadrature normalization, including roundoff.
            slope *= sum(BigFloat.(rule.weights))
            Float64((BigFloat(config.beta) * slope)^(-1 / BigFloat(config.gamma)))
        end
        cash_states = [20_000.0, 20_000.001, 20_000.002]
        repaired = [G2._direct_global_action(config, process, rule, grid, options,
            child_value, child_policy, cash, 0.0, G2.G2EGMSolveDiagnostics())
            for cash in cash_states]
        for (cash, action) in zip(cash_states, repaired)
            @test action.consumption ≈ optimal_consumption atol=1e-8 rtol=0.0
            @test action.saving ≈ cash - optimal_consumption atol=1e-8 rtol=0.0
        end
        @test all(diff(getfield.(repaired, :saving)) .> 0.0)
        @test_throws ArgumentError G2._direct_global_action(config, process, rule,
            grid, options, child_value, child_policy, 0.0, 0.0, G2.G2EGMSolveDiagnostics())
        @test_throws DomainError G2._direct_global_action(config,
            Ptd.LogAR1Process(rho=0.9, sigma=10.0, unconditional_log_mean=0.0),
            rule, grid, options, child_value, child_policy, 20_000.0, 0.0,
            G2.G2EGMSolveDiagnostics())
    end

    @testset "queries fail closed outside the solved domain" begin
        (; grid) = baseline
        policy_cash_upper = grid.cash[
            searchsortedlast(grid.cash, last(grid.asset))]
        first_padding_cash = grid.cash[
            searchsortedlast(grid.cash, last(grid.asset)) + 1]
        G2.reset_query_diagnostics!(baseline_solution)
        @test_throws DomainError G2.consumption(
            baseline_solution, first(grid.cash) - 1e-6, 0.0; date=1)
        @test_throws DomainError G2.consumption(
            baseline_solution, first_padding_cash, 0.0; date=1)
        @test_throws DomainError G2.value_at(
            baseline_solution, grid.cash[100], first(grid.log_income) - 1e-6;
            date=1)
        @test_throws DomainError G2.value_at(
            baseline_solution, grid.cash[100], last(grid.log_income) + 1e-6;
            date=1)
        @test_throws ArgumentError G2.consumption(
            baseline_solution, NaN, 0.0; date=1)
        @test_throws ArgumentError G2.value_at(
            baseline_solution, 4.0, Inf; date=1)
        @test_throws ArgumentError G2.consumption(
            baseline_solution, 4.0, 0.0; date=0)
        @test_throws ArgumentError G2.value_at(
            baseline_solution, 4.0, 0.0; date=3)

        # The analytic terminal rule remains part of the declared solve and
        # must obey the same fail-closed state-domain contract.
        @test_throws DomainError G2.consumption(
            terminal_solution, first(terminal_inputs.grid.cash) - 1e-6,
            0.0; date=1)
        @test_throws DomainError G2.value_at(
            terminal_solution, 4.0,
            last(terminal_inputs.grid.log_income) + 1e-6; date=1)

        query_audit = G2.diagnostics(baseline_solution).query
        @test query_audit.domain_errors == 4
        @test query_audit.consumption_queries == 0
        @test query_audit.value_queries == 0

        G2.reset_query_diagnostics!(baseline_solution)
        G2.consumption(baseline_solution,
            first(grid.cash), first(grid.log_income); date=1)
        G2.value_at(baseline_solution,
            policy_cash_upper, last(grid.log_income); date=1)
        query_audit = G2.diagnostics(baseline_solution).query
        @test query_audit.consumption_queries == 1
        @test query_audit.value_queries == 1
        @test query_audit.lower_cash_contacts == 1
        @test query_audit.upper_cash_contacts == 1
        @test query_audit.lower_log_income_contacts == 1
        @test query_audit.upper_log_income_contacts == 1
        @test query_audit.domain_errors == 0
        G2.reset_query_diagnostics!(baseline_solution)
        reset_audit = G2.diagnostics(baseline_solution).query
        @test reset_audit.consumption_queries == 0
        @test reset_audit.value_queries == 0
        @test isinf(reset_audit.minimum_cash_margin)
        @test isinf(reset_audit.minimum_log_income_margin)
    end

    @testset "fingerprints, alignment, diagnostics, and provenance" begin
        (; config, process, rule, grid) = baseline
        fingerprint = G2.solution_fingerprint(baseline_solution)
        @test occursin(r"^[0-9a-f]{64}$", fingerprint)
        @test G2.validate_alignment(
            baseline_solution, config, process, rule, grid) === nothing

        repeated_terminal = G2.build_ar1_solution(
            terminal_inputs.config, terminal_inputs.process;
            base_rule=terminal_inputs.rule, grid=terminal_inputs.grid)
        @test G2.solution_fingerprint(repeated_terminal) ==
            G2.solution_fingerprint(terminal_solution)

        wrong_config = Ptd.StochasticPTDConfig(
            beta=0.95, gamma=config.gamma,
            return_rate=config.return_rate, cash_floor=config.cash_floor,
            terminal_date=config.terminal.final_date)
        @test_throws ArgumentError G2.validate_alignment(
            baseline_solution, wrong_config, process, rule, grid)

        changed_asset = copy(grid.asset)
        changed_asset[2] = (changed_asset[1] + changed_asset[3]) / 2.0
        changed_grid = G2.AR1G2EGMGrid(
            changed_asset, grid.post_log_income, grid.cash, grid.log_income)
        @test_throws ArgumentError G2.validate_alignment(
            baseline_solution, config, process, rule, changed_grid)

        changed_options = G2.AR1G2EGMOptions(hole_search_radius=1)
        @test_throws ArgumentError G2.validate_alignment(
            baseline_solution, config, process, rule, grid;
            options=changed_options)

        tampered = G2.AR1G2EGMSolution(
            baseline_solution.config,
            baseline_solution.process,
            baseline_solution.base_rule,
            baseline_solution.grid,
            baseline_solution.options,
            baseline_solution.policy,
            baseline_solution.value,
            baseline_solution.solve_diagnostics,
            baseline_solution.query_diagnostics,
            "not-a-valid-fingerprint",
        )
        @test_throws ArgumentError G2.validate_alignment(
            tampered, config, process, rule, grid)

        changed_policy = deepcopy(baseline_solution.policy)
        changed_policy[1][2, 2] = nextfloat(changed_policy[1][2, 2])
        tampered_arrays = G2.AR1G2EGMSolution(
            baseline_solution.config,
            baseline_solution.process,
            baseline_solution.base_rule,
            baseline_solution.grid,
            baseline_solution.options,
            changed_policy,
            baseline_solution.value,
            baseline_solution.solve_diagnostics,
            baseline_solution.query_diagnostics,
            baseline_solution.fingerprint,
        )
        @test_throws ArgumentError G2.validate_alignment(
            tampered_arrays, config, process, rule, grid)

        changed_value = deepcopy(baseline_solution.value)
        changed_value[1][2, 2] = nextfloat(changed_value[1][2, 2])
        tampered_values = G2.AR1G2EGMSolution(
            baseline_solution.config,
            baseline_solution.process,
            baseline_solution.base_rule,
            baseline_solution.grid,
            baseline_solution.options,
            baseline_solution.policy,
            changed_value,
            baseline_solution.solve_diagnostics,
            baseline_solution.query_diagnostics,
            baseline_solution.fingerprint,
        )
        @test_throws ArgumentError G2.validate_alignment(
            tampered_values, config, process, rule, grid)

        solve_audit = G2.diagnostics(baseline_solution).solve
        periods = config.terminal.final_date - 1
        common_nodes = length(grid.cash) * length(grid.log_income)
        @test solve_audit.quadrature_expectations > 0
        @test solve_audit.continuation_queries ==
            5 * solve_audit.quadrature_expectations
        @test solve_audit.triangles_considered > 0
        @test solve_audit.triangles_accepted <= solve_audit.triangles_considered
        @test solve_audit.triangle_winners +
              solve_audit.floor_kink_winners +
              solve_audit.direct_repair_winners +
              solve_audit.constraint_winners == periods * common_nodes
        @test solve_audit.minimum_solve_cash_margin >= 0.0
        @test solve_audit.minimum_solve_log_income_margin > 0.0

        quadrature = G2.quadrature_audit(baseline_solution)
        @test quadrature.family == :positive_gauss_hermite
        @test quadrature.nodes_per_dimension == 5
        @test quadrature.standard_normal_nodes == Tuple(only.(rule.nodes))
        @test quadrature.normalized_weights == Tuple(rule.weights)
        @test quadrature.weight_sum ≈ 1.0 atol=8eps(Float64)
        @test G2.G2EGM_SOLVER_SCHEMA ==
            "ar1-g2egm-exact-cell-repair-v4"
        @test G2.G2EGM_REFERENCE_COMMIT ==
            "30aa550d8319582f9a17e08192dc8f1361726c9b"
        @test occursin(r"^[0-9a-f]{40}$", G2.G2EGM_REFERENCE_COMMIT)
    end
end

end # module AR1G2EGMSolverTests
