# Tests for src/ptd/stochastic_model.jl and src/ptd/ptd_sceq.jl.
using Test
using Random

module StochasticCoreUnderTest
include(joinpath(@__DIR__, "..", "src", "ptd", "stochastic_model.jl"))
include(joinpath(@__DIR__, "..", "src", "ptd", "ptd_sceq.jl"))
end

const SC = StochasticCoreUnderTest

@testset verbose = true "Nonanticipative stochastic PTD core" begin
    @testset "immutable contracts and validation" begin
        iid = SC.IIDLognormalProcess(sigma=0.5)
        ar1 = SC.LogAR1Process(rho=0.8, sigma=0.2)
        hsz = SC.HSZProcess(rho_earnings=0.9, sigma_earnings=0.1,
            rho_medical=0.8, sigma_medical=0.2)
        observed = SC.ObservedState(1, 3.0, SC.IIDState())
        config = SC.StochasticPTDConfig(beta=0.98, gamma=2.0,
            return_rate=0.1, cash_floor=3.0, terminal_date=50)
        action = SC.ActionResult(2.0, 1.0, nothing, :ptd_sceq, "row-test")
        transition = SC.TransitionResult(2.1, 0.9, 3.0, SC.IIDState())
        row = SC.TrajectoryRow(1, observed, action, transition, -0.5)
        terminal_row = SC.TrajectoryRow(1, observed, action, nothing, -Inf)

        @test all(isimmutable, (iid, ar1, hsz, observed, config,
            config.terminal, action, row, terminal_row))
        @test row.transition === transition
        @test isnothing(terminal_row.transition)
        @test_throws ArgumentError SC.TrajectoryRow(
            0, observed, action, transition, 0.0)
        @test_throws ArgumentError SC.TrajectoryRow(
            1, observed,
            SC.ActionResult(1.0, 1.0, nothing, :ptd_sceq, "infeasible"),
            transition, 0.0)
        @test_throws ArgumentError SC.TrajectoryRow(
            1, observed, action, transition, NaN)
        @test iid.log_mean == -0.5^2 / 2
        @test exp(iid.log_mean + iid.log_std^2 / 2) ≈ 1.0
        @test SC.innovation_dimension(iid) == 1
        @test SC.innovation_dimension(ar1) == 1
        @test SC.innovation_dimension(hsz) == 2

        @test_throws ArgumentError SC.IIDLognormalProcess(sigma=-0.1)
        @test_throws ArgumentError SC.LogAR1Process(rho=1.0, sigma=0.1)
        @test_throws ArgumentError SC.HSZProcess(
            rho_earnings=0.9, sigma_earnings=-0.1,
            rho_medical=0.8, sigma_medical=0.1)
        @test_throws ArgumentError SC.ObservedState(0, 3.0, SC.IIDState())
        @test_throws ArgumentError SC.StochasticPTDConfig(beta=0.0, gamma=2.0,
            return_rate=0.1, cash_floor=3.0, terminal_date=50)
    end

    @testset "positive normalized Gauss-Hermite rules" begin
        scalar = SC.gauss_hermite_rule(7; dimension=1)
        @test isimmutable(scalar)
        @test length(scalar.nodes) == 7
        @test all(>(0.0), scalar.weights)
        @test sum(scalar.weights) ≈ 1.0 atol=2e-15
        @test SC.quadrature_expectation(node -> node[1], scalar) ≈ 0.0 atol=2e-15
        @test SC.quadrature_expectation(scalar, node -> node[1]^2) ≈ 1.0 atol=2e-14
        @test SC.quadrature_expectation(node -> node[1]^4, scalar) ≈ 3.0 atol=2e-13

        tensor = SC.gauss_hermite_rule(5; dimension=2)
        @test length(tensor.nodes) == 25
        @test all(>(0.0), tensor.weights)
        @test sum(tensor.weights) ≈ 1.0 atol=2e-15
        @test SC.quadrature_expectation(node -> node[1]^2, tensor) ≈ 1.0 atol=2e-14
        @test SC.quadrature_expectation(node -> node[2]^2, tensor) ≈ 1.0 atol=2e-14
        @test SC.quadrature_expectation(node -> node[1] * node[2], tensor) ≈ 0.0 atol=2e-15
        hsz_process = SC.HSZProcess(
            rho_earnings=0.9, sigma_earnings=0.1,
            rho_medical=0.8, sigma_medical=0.2)
        @test SC.gauss_hermite_rule(hsz_process, 3) isa SC.GaussHermiteRule{2,9}
        @test SC.gauss_hermite_rule(hsz_process, 5) isa SC.GaussHermiteRule{2,25}

        mean_one = SC.IIDLognormalProcess(sigma=0.5)
        iid_rule = SC.gauss_hermite_rule(mean_one, 15)
        integrated_mean = SC.quadrature_expectation(iid_rule) do node
            exp(mean_one.log_mean + mean_one.log_std * node[1])
        end
        @test integrated_mean ≈ 1.0 rtol=2e-14

        # The reported five-node rule uses standard-normal nodes exactly once
        # in the mean-one lognormal transform. These moment checks catch a
        # missing sqrt(2), a second sigma scaling, or an omitted -sigma^2/2.
        for sigma in (0.1, 0.5)
            process = SC.IIDLognormalProcess(sigma=sigma)
            rule5 = SC.gauss_hermite_rule(process, 5)
            incomes = [exp(process.log_mean + process.log_std * node[1])
                       for node in rule5.nodes]
            integrated_income = sum(rule5.weights .* incomes)
            integrated_variance = sum(rule5.weights .*
                (incomes .- integrated_income) .^ 2)
            integrated_log_income = sum(rule5.weights .* log.(incomes))
            @test integrated_income ≈ 1.0 atol=4e-8 rtol=0.0
            @test integrated_variance ≈ exp(sigma^2) - 1.0 rtol=2e-4
            @test integrated_log_income ≈ -sigma^2 / 2 atol=2e-14
        end
        @test_throws ArgumentError SC.gauss_hermite_rule(3; dimension=3)
    end

    @testset "centralized true floor transition" begin
        config = SC.StochasticPTDConfig(beta=0.98, gamma=1.0,
            return_rate=0.1, cash_floor=3.0, terminal_date=4)
        iid = SC.IIDLognormalProcess(sigma=0.0)
        observed = SC.ObservedState(1, 3.0, SC.IIDState())
        result = SC.true_transition(config, iid, observed, 3.0, -19.0)
        @test result.z == 1.0
        @test result.transfer == 2.0
        @test result.xnext == 3.0
        @test result.snext == SC.IIDState()
        @test SC.next_observed(observed, result) ==
            SC.ObservedState(2, 3.0, SC.IIDState())

        no_floor = SC.true_transition(config, iid,
            SC.ObservedState(1, 10.0, SC.IIDState()), 4.0, 12.0)
        @test no_floor.z ≈ 7.6
        @test no_floor.transfer == 0.0
        @test no_floor.xnext ≈ 7.6
        @test_throws ArgumentError SC.true_transition(
            config, iid, observed, 3.1, 0.0)
        @test_throws ArgumentError SC.true_transition(config, iid,
            SC.ObservedState(4, 3.0, SC.IIDState()), 3.0, 0.0)
    end

    @testset "conditional-mean nominal paths preserve persistent state" begin
        config = SC.StochasticPTDConfig(beta=0.96, gamma=2.0,
            return_rate=0.02, cash_floor=1.0, terminal_date=5)

        iid = SC.IIDLognormalProcess(sigma=0.5)
        iid_path = SC.nominal_path(config, iid,
            SC.ObservedState(2, 4.0, SC.IIDState()))
        @test iid_path.dates == (2, 3, 4, 5)
        @test all(isapprox.(iid_path.incomes, (1.0, 1.0, 1.0, 1.0)))
        @test iid_path.rule == :conditional_mean_income
        @test length(iid_path.fingerprint) == 64

        ar1 = SC.LogAR1Process(rho=0.8, sigma=0.2,
            unconditional_log_mean=0.1)
        ar1_observed = SC.ObservedState(2, 4.0, SC.LogAR1State(0.7))
        ar1_path = SC.nominal_path(config, ar1, ar1_observed)
        mean_log_h1 = 0.1 + 0.8 * (0.7 - 0.1)
        mean_log_h2 = 0.1 + 0.8^2 * (0.7 - 0.1)
        @test ar1_path.states[1].log_income == 0.7
        @test ar1_path.states[2].log_income ≈ mean_log_h1
        @test ar1_path.incomes[2] ≈ exp(mean_log_h1 + 0.2^2 / 2)
        @test ar1_path.incomes[3] ≈
            exp(mean_log_h2 + 0.2^2 * (1 + 0.8^2) / 2)
        @test ar1_path.incomes[2] != exp(0.1 + 0.2^2 / (2 * (1 - 0.8^2)))

        ar1_transition = SC.true_transition(config, ar1, ar1_observed, 2.0, 0.0)
        @test ar1_transition.snext.log_income ≈ mean_log_h1
        @test ar1_transition.z ≈ 1.02 * 2.0 + exp(mean_log_h1)

        hsz = SC.HSZProcess(rho_earnings=0.9, sigma_earnings=0.1,
            rho_medical=0.8, sigma_medical=0.2,
            unconditional_log_earnings=log(2.0),
            unconditional_log_medical=log(0.5))
        hsz_observed = SC.ObservedState(
            2, 4.0, SC.HSZState(log(4.0), log(1.0)))
        hsz_path = SC.nominal_path(config, hsz, hsz_observed)
        mean_e_h1 = log(2.0) + 0.9 * (log(4.0) - log(2.0))
        mean_m_h1 = log(0.5) + 0.8 * (log(1.0) - log(0.5))
        @test hsz_path.states[2].log_earnings ≈ mean_e_h1
        @test hsz_path.states[2].log_medical ≈ mean_m_h1
        @test hsz_path.incomes[2] ≈
            exp(mean_e_h1 + 0.1^2 / 2) - exp(mean_m_h1 + 0.2^2 / 2)

        hsz_transition = SC.true_transition(
            config, hsz, hsz_observed, 2.0, (0.0, 0.0))
        @test hsz_transition.snext == SC.HSZState(mean_e_h1, mean_m_h1)
        @test hsz_transition.z ≈
            1.02 * 2.0 + exp(mean_e_h1) - exp(mean_m_h1)
    end

    @testset "validated read-only PTD plan and calendar indexing" begin
        config = SC.StochasticPTDConfig(beta=0.98, gamma=2.0,
            return_rate=0.1, cash_floor=3.0, terminal_date=4)
        process = SC.IIDLognormalProcess(sigma=0.0)
        observed = SC.ObservedState(2, 8.0, SC.IIDState())
        path = SC.nominal_path(config, process, observed)
        plan = SC.build_ptd_plan(config, path)
        action = SC.first_action(plan, observed)

        @test isimmutable(plan)
        @test propertynames(plan) == (:path, :config, :fingerprint)
        @test_throws ArgumentError plan.agent
        @test path.dates == (2, 3, 4)
        @test isfinite(action.objective)
        @test action.method == :deterministic_ptd
        @test 0.0 < action.consumption <= observed.cash
        @test action.saving == observed.cash - action.consumption
        boundaries = SC.segment_boundaries(plan)
        @test first(boundaries) == 0.0
        @test isinf(last(boundaries))
        @test issorted(collect(boundaries))

        direct = SC.DeterministicProblem(β=config.beta, γ=config.gamma,
            r=config.return_rate, xFloor=config.cash_floor,
            incomes=ones(1, 3))
        SC.deterministic_ptd!(direct)
        @test action.consumption ≈ SC.c_x(direct, 8.0; i=1, t=1)
        @test SC.plan_value(plan, 8.0) ≈ SC.v_x(direct, 8.0; i=1, t=1)
        @test SC.marginal_value(plan, 8.0) ≈
            SC.mu(direct, action.consumption)

        # A CE continuation attached at calendar t+2 starts with that date in
        # local column one; its first transition income is calendar t+3 in
        # local column two. This guards the former off-by-one ambiguity.
        tail_observed = SC.ObservedState(3, 6.0, SC.IIDState())
        tail_path = SC.nominal_path(
            SC.StochasticPTDConfig(beta=0.98, gamma=2.0,
                return_rate=0.1, cash_floor=3.0, terminal_date=6),
            process, tail_observed)
        tail_plan = SC.build_ptd_plan(
            SC.StochasticPTDConfig(beta=0.98, gamma=2.0,
                return_rate=0.1, cash_floor=3.0, terminal_date=6),
            tail_path)
        @test tail_path.dates == (3, 4, 5, 6)
        @test getfield(tail_plan, :_agent).incomes[1, 2] == tail_path.incomes[2]

        # Although Julia permits deliberate `getfield` escape hatches, every
        # public query detects mutation of the opaque policy snapshot.
        corrupt_plan = SC.build_ptd_plan(config, path)
        wrapped = getfield(corrupt_plan, :_agent)
        segment = wrapped.policy[1, 1][1]
        wrapped.policy[1, 1][1] = SC.PolicySegment(xStart=segment.xStart,
            xEnd=segment.xEnd, b0=segment.b0 + 0.01, b1=segment.b1)
        @test_throws ErrorException SC.plan_value(corrupt_plan, 8.0)
    end

    @testset "PTD-SCEQ nonanticipation, terminal rule, and deterministic collapse" begin
        config = SC.StochasticPTDConfig(beta=0.97, gamma=1.5,
            return_rate=0.03, cash_floor=3.0, terminal_date=5)
        process = SC.IIDLognormalProcess(sigma=0.0)
        observed = SC.ObservedState(1, 9.0, SC.IIDState())

        first_result = SC.ptd_sceq(config, process, observed)
        rand(MersenneTwister(11), 10_000)
        second_result = SC.ptd_sceq(config, process, observed)
        @test first_result.consumption == second_result.consumption
        @test first_result.objective == second_result.objective
        @test first_result.plan_fingerprint == second_result.plan_fingerprint
        @test first_result.method == :ptd_sceq
        action_only = SC.ptd_sceq(
            config, process, observed; evaluate_objective=false)
        @test action_only.consumption == first_result.consumption
        @test action_only.saving == first_result.saving
        @test action_only.plan_fingerprint == first_result.plan_fingerprint
        @test isnothing(action_only.objective)
        @test !applicable(SC.ptd_sceq, config, process, observed,
            MersenneTwister(1))

        direct = SC.DeterministicProblem(β=config.beta, γ=config.gamma,
            r=config.return_rate, xFloor=config.cash_floor,
            incomes=ones(1, config.terminal.final_date))
        SC.deterministic_ptd!(direct)
        @test first_result.consumption ≈ SC.c_x(direct, 9.0; i=1, t=1)

        low_shock = SC.true_transition(
            config, process, observed, first_result.consumption, -100.0)
        high_shock = SC.true_transition(
            config, process, observed, first_result.consumption, 100.0)
        @test low_shock == high_shock

        terminal_observed = SC.ObservedState(5, 7.0, SC.IIDState())
        terminal_action = SC.ptd_sceq(config, process, terminal_observed)
        @test terminal_action.consumption == 7.0
        @test terminal_action.saving == 0.0
        @test terminal_action.objective == SC.u(
            getfield(SC.build_ptd_plan(
                config, SC.nominal_path(config, process, terminal_observed)),
                :_agent), 7.0)
        @test_throws ArgumentError SC.true_transition(
            config, process, terminal_observed, 7.0, 0.0)

        persistent_config = SC.StochasticPTDConfig(beta=0.97, gamma=1.5,
            return_rate=0.03, cash_floor=1.0, terminal_date=3)
        ar1 = SC.LogAR1Process(rho=0.8, sigma=0.1)
        ar1_action = SC.ptd_sceq(persistent_config, ar1,
            SC.ObservedState(1, 6.0, SC.LogAR1State(log(2.0))))
        @test ar1_action.method == :ptd_sceq
        @test 0.0 < ar1_action.consumption <= 6.0

        hsz = SC.HSZProcess(rho_earnings=0.9, sigma_earnings=0.1,
            rho_medical=0.8, sigma_medical=0.1,
            unconditional_log_earnings=log(3.0),
            unconditional_log_medical=log(0.5))
        hsz_action = SC.ptd_sceq(persistent_config, hsz,
            SC.ObservedState(1, 6.0, SC.HSZState(log(4.0), log(0.5))))
        @test hsz_action.method == :ptd_sceq
        @test 0.0 < hsz_action.consumption <= 6.0
    end
end
