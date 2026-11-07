# Tests for src/ptd/ptd_osa.jl and source-order file 5.
using Test
using Random

module OAPTDUnderTest
include(joinpath(@__DIR__, "..", "src", "ptd", "ptd_osa.jl"))
end

const OA = OAPTDUnderTest

function dense_ptd_osa_check(config, observed, tails; points=4001)
    savings = range(0.0, observed.cash; length=points)
    values = [OA._ptd_osa_objective(config, observed, tails, Float64(saving))
        for saving in savings]
    maximum_value = maximum(values)
    # The policy tie rule chooses more saving among equal grid values.
    index = findlast(==(maximum_value), values)
    return Float64(savings[index]), maximum_value
end

@testset verbose = true "One-step-ahead PTD" begin
    quick = OA.PTDOSAOptions(
        certification_atol=2e-7, direct_iterations=16)

    @testset "Brent root search" begin
        calls = Ref(0)
        gap = saving -> begin
            calls[] += 1
            return inv(1.0 - saving) - 3.0
        end
        lower, upper = 0.0, 1.0
        root = OA._brent_ptd_osa_root(
            gap, lower, upper, gap(lower), gap(upper))
        @test root ≈ 2 / 3 rtol=OA.PTD_OSA_ROOT_XRTOL
        @test calls[] < 30
        @test_throws ArgumentError OA._brent_ptd_osa_root(
            identity, 1.0, 2.0, 1.0, 2.0)
    end

    @testset "terminal and last-decision objectives" begin
        config = OA.StochasticPTDConfig(beta=0.97, gamma=2.0,
            return_rate=0.04, cash_floor=1.0, terminal_date=3)
        process = OA.IIDLognormalProcess(sigma=0.3)
        rule = OA.gauss_hermite_rule(process, 7)

        terminal = OA.ObservedState(3, 4.5, OA.IIDState())
        terminal_result = OA.ptd_osa_with_diagnostics(
            config, process, terminal, rule; options=quick)
        @test terminal_result.action.method == :ptd_osa
        @test terminal_result.action.consumption == 4.5
        @test terminal_result.action.saving == 0.0
        @test terminal_result.action.objective ==
            (4.5^(1 - config.gamma) - 1) / (1 - config.gamma)
        @test terminal_result.diagnostics.certified
        @test terminal_result.diagnostics.tail_start_date == 3

        observed = OA.ObservedState(2, 5.0, OA.IIDState())
        result = OA.ptd_osa_with_diagnostics(
            config, process, observed, rule; options=quick)
        saving = result.action.saving
        consumption = observed.cash - saving
        exact_continuation = OA.quadrature_expectation(rule) do node
            income_next = exp(process.log_mean + process.log_std * node[1])
            next_cash = max((1 + config.return_rate) * saving + income_next,
                config.cash_floor)
            (next_cash^(1 - config.gamma) - 1) / (1 - config.gamma)
        end
        exact_objective = (consumption^(1 - config.gamma) - 1) /
            (1 - config.gamma) + config.beta * exact_continuation
        @test result.action.objective ≈ exact_objective rtol=2e-13 atol=2e-13

        tails = OA._build_ptd_osa_tails(config, process, observed, rule)
        dense_saving, dense_value = dense_ptd_osa_check(
            config, observed, tails; points=10_001)
        @test result.action.objective >= dense_value - 2e-10
        @test abs(result.action.saving - dense_saving) <= observed.cash / 10_000 + 2e-8
        @test result.diagnostics.certification_gap <= quick.certification_atol
    end

    @testset "tail date t+1 uses conditional-mean income from t+2" begin
        config = OA.StochasticPTDConfig(beta=0.96, gamma=1.5,
            return_rate=0.02, cash_floor=1.0, terminal_date=4)
        process = OA.LogAR1Process(rho=0.8, sigma=0.2,
            unconditional_log_mean=0.1)
        observed = OA.ObservedState(1, 6.0, OA.LogAR1State(0.7))
        rule = OA.gauss_hermite_rule(process, 3)
        tails = OA._build_ptd_osa_tails(config, process, observed, rule)
        zero_index = argmin(abs(node[1]) for node in rule.nodes)
        @test abs(rule.nodes[zero_index][1]) <= 16eps(Float64)
        tail = tails[zero_index]
        next_log = 0.1 + 0.8 * (0.7 - 0.1)
        t_plus_2_mean_log = 0.1 + 0.8 * (next_log - 0.1)

        @test tail.plan.path.dates == (2, 3, 4)
        @test tail.plan.path.states[1].log_income ≈ next_log
        # Column one is the already observed t+1 income and is bookkeeping for
        # the tail. Column two is the first future CE income, at t+2.
        @test tail.plan.path.incomes[1] ≈ exp(next_log)
        @test tail.plan.path.incomes[2] ≈
            exp(t_plus_2_mean_log + process.log_std^2 / 2)
        @test getfield(tail.plan, :_agent).incomes[1, 2] ==
            tail.plan.path.incomes[2]

        result = OA.ptd_osa_with_diagnostics(
            config, process, observed, rule; options=quick)
        @test result.diagnostics.tail_start_date == 2
    end

    @testset "floor derivative exclusion and complete structural search" begin
        config = OA.StochasticPTDConfig(beta=0.98, gamma=2.0,
            return_rate=0.1, cash_floor=3.0, terminal_date=3)
        process = OA.IIDLognormalProcess(sigma=0.4)
        observed = OA.ObservedState(1, 5.0, OA.IIDState())
        rule = OA.gauss_hermite_rule(process, 7)
        tails = OA._build_ptd_osa_tails(config, process, observed, rule)
        boundaries = OA._ptd_osa_structural_boundaries(config, observed, tails)

        first_midpoint = (boundaries[1] + boundaries[2]) / 2
        branches = OA._ptd_osa_cell_branches(config, tails, first_midpoint)
        excluded = count(isnothing, branches)
        @test excluded > 0
        gap = OA._ptd_osa_euler_gap(
            config, observed, tails, branches, first_midpoint)
        manual_rhs = 0.0
        gross_return = 1 + config.return_rate
        for (tail, segment) in zip(tails, branches)
            isnothing(segment) && continue
            resources = gross_return * first_midpoint + tail.next_income
            next_consumption = segment.b0 + segment.b1 * resources
            manual_rhs += tail.weight * next_consumption^(-config.gamma)
        end
        manual_gap = (observed.cash - first_midpoint)^(-config.gamma) -
            config.beta * gross_return * manual_rhs
        @test gap ≈ manual_gap rtol=2e-14 atol=2e-14

        # Every node floor switch that lies in the feasible saving domain is an
        # explicit structural candidate.
        for tail in tails
            mapped = (config.cash_floor - tail.next_income) / gross_return
            if 0.0 < mapped < observed.cash
                @test mapped in boundaries
            end
        end

        result = OA.ptd_osa_with_diagnostics(
            config, process, observed, rule; options=quick)
        @test result.diagnostics.floor_derivative_exclusions >= excluded
        @test result.diagnostics.structural_boundaries == Tuple(boundaries)
        @test result.diagnostics.candidate_count ==
            length(unique(vcat(boundaries,
                collect(result.diagnostics.interior_roots))))

        # On each fixed-signature cell, every sign-changing Euler bracket has
        # an enumerated root in that cell.
        roots = collect(result.diagnostics.interior_roots)
        for cell in 1:length(boundaries)-1
            lower, upper = boundaries[cell], boundaries[cell+1]
            midpoint = lower + (upper - lower) / 2
            cell_branches = OA._ptd_osa_cell_branches(config, tails, midpoint)
            cell_gap = saving -> OA._ptd_osa_euler_gap(
                config, observed, tails, cell_branches, saving)
            if OA._opposite_sign(cell_gap(lower), cell_gap(upper))
                @test any(root -> lower < root < upper, roots)
            end
        end
        @test result.diagnostics.certified
        @test !result.diagnostics.fallback_used

        tie_options = OA.PTDOSAOptions(tie_atol=1e-6, tie_rtol=0.0,
            certification_atol=1e-8, direct_iterations=8)
        selected = OA._best_ptd_osa_pairs(
            [(0.0, 1.0), (1.0, 1.0 - 0.75e-6),
                (2.0, 1.0 - 1.5e-6)], tie_options)
        @test selected[1] == 1.0
        @test OA._best_ptd_osa_pairs(
            [(0.0, 1.0), (2.0, 1.0)], tie_options)[1] == 2.0

        # A failed diagnostic must not let an approximate direct search replace
        # a known better structural candidate.  If values tie, save more.
        worse_direct = OA._resolve_ptd_osa_certification(
            1.0, 10.0, 2.0, 9.0, false, tie_options)
        @test worse_direct.saving == 1.0
        @test worse_direct.value == 10.0
        @test !worse_direct.certified
        @test worse_direct.fallback_used
        tied_direct = OA._resolve_ptd_osa_certification(
            1.0, 10.0, 2.0, 10.0 - 0.5e-6, false, tie_options)
        @test tied_direct.saving == 2.0
        @test tied_direct.value == 10.0 - 0.5e-6
    end

    @testset "global agreement for IID, AR1, and HSZ" begin
        cases = (
            (
                OA.StochasticPTDConfig(beta=0.96, gamma=1.4,
                    return_rate=0.03, cash_floor=1.0, terminal_date=3),
                OA.IIDLognormalProcess(sigma=0.25),
                OA.ObservedState(1, 6.0, OA.IIDState()),
                5,
            ),
            (
                OA.StochasticPTDConfig(beta=0.96, gamma=1.4,
                    return_rate=0.03, cash_floor=1.0, terminal_date=3),
                OA.LogAR1Process(rho=0.8, sigma=0.15,
                    unconditional_log_mean=0.0),
                OA.ObservedState(1, 6.0, OA.LogAR1State(log(1.5))),
                5,
            ),
            (
                OA.StochasticPTDConfig(beta=0.96, gamma=1.4,
                    return_rate=0.03, cash_floor=1.0, terminal_date=3),
                OA.HSZProcess(rho_earnings=0.8, sigma_earnings=0.1,
                    rho_medical=0.7, sigma_medical=0.1,
                    unconditional_log_earnings=log(2.5),
                    unconditional_log_medical=log(0.4)),
                OA.ObservedState(1, 6.0,
                    OA.HSZState(log(3.0), log(0.5))),
                5,
            ),
        )

        for (config, process, observed, nodes) in cases
            rule = OA.gauss_hermite_rule(process, nodes)
            result = OA.ptd_osa_with_diagnostics(
                config, process, observed, rule; options=quick)
            tails = OA._build_ptd_osa_tails(config, process, observed, rule)
            dense_saving, dense_value = dense_ptd_osa_check(
                config, observed, tails; points=3001)
            @test result.action.method == :ptd_osa
            @test result.action.objective >= dense_value - 3e-8
            @test abs(result.action.saving - dense_saving) <=
                observed.cash / 3000 + 5e-6
            @test result.diagnostics.certification_gap <=
                quick.certification_atol
        end
    end

    @testset "deterministic collapse and nonanticipation" begin
        config = OA.StochasticPTDConfig(beta=0.97, gamma=1.5,
            return_rate=0.03, cash_floor=3.0, terminal_date=5)
        deterministic_process = OA.IIDLognormalProcess(sigma=0.0)
        observed = OA.ObservedState(1, 9.0, OA.IIDState())
        rule = OA.gauss_hermite_rule(deterministic_process, 7)

        first_action = OA.ptd_osa(
            config, deterministic_process, observed, rule; options=quick)
        rand(MersenneTwister(22), 20_000)
        second_action = OA.ptd_osa(
            config, deterministic_process, observed, rule; options=quick)
        sceq_action = OA.ptd_sceq(config, deterministic_process, observed)
        @test first_action.method == :ptd_osa
        @test first_action.consumption == second_action.consumption
        @test first_action.objective == second_action.objective
        @test first_action.plan_fingerprint == second_action.plan_fingerprint
        @test first_action.consumption ≈ sceq_action.consumption rtol=2e-13 atol=2e-13
        @test first_action.objective ≈ sceq_action.objective rtol=2e-13 atol=2e-13
        @test !applicable(OA.ptd_osa, config, deterministic_process,
            observed, rule, MersenneTwister(1))
    end

    @testset "five-node reported rule and higher-node IID sensitivity" begin
        config = OA.StochasticPTDConfig(beta=0.96, gamma=2.0,
            return_rate=0.02, cash_floor=1.0, terminal_date=3)
        process = OA.IIDLognormalProcess(sigma=0.1)
        observed = OA.ObservedState(1, 8.0, OA.IIDState())
        rule5 = OA.gauss_hermite_rule(process, 5)
        rule31 = OA.gauss_hermite_rule(process, 31)
        action5 = OA.ptd_osa(config, process, observed, rule5; options=quick)
        action31 = OA.ptd_osa(config, process, observed, rule31; options=quick)
        @test action5.consumption ≈ action31.consumption rtol=2e-7 atol=2e-7
        @test action5.objective ≈ action31.objective rtol=2e-9 atol=2e-9
    end
end
