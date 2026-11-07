# Stochastic DCEGM model tests (src/11-src/13).
using Distributions
using Random

module EgmGaussHermiteFixture
include(joinpath(@__DIR__, "..", "src", "ptd", "stochastic_model.jl"))
end

const StochasticEgmUnderTest = Ptd.DcEgm
const EgmGH = EgmGaussHermiteFixture

@testset verbose = true "Stochastic pure-Julia DCEGM" begin
    @testset "supported model matrix and validation" begin
        a = [0.0, 1.0]

        @test_throws ArgumentError StochasticEgmUnderTest.build_agent(
            model="iid", β=0.98, γ=1.0, r=0.0, xFloor=0.0, tMax=2,
            a=a, d=Bernoulli(0.5))
        @test_throws ArgumentError StochasticEgmUnderTest.build_agent(
            model="ar1", β=0.98, γ=1.0, r=0.0, xFloor=1.0, tMax=2,
            a=a, ρ=NaN, d=Normal(), y=[0.5, 1.5])
        @test_throws ArgumentError StochasticEgmUnderTest.build_agent(
            model="ar1", β=0.98, γ=1.0, r=0.0, xFloor=1.0, tMax=2,
            a=a, ρ=0.9, d=Normal(), y=[1.0, Inf])
        @test_throws ArgumentError StochasticEgmUnderTest.build_agent(
            model="ar1", β=0.98, γ=1.0, r=0.0, xFloor=1.0, tMax=2,
            a=a, ρ=0.9, d=Bernoulli(0.5), y=[0.5, 1.5])

        @test_throws ArgumentError StochasticEgmUnderTest.build_agent(
            model="hsz", β=0.98, γ=1.0, r=0.0, xFloor=1.0, tMax=2,
            a=a)
        @test !isdefined(StochasticEgmUnderTest, :AgentHSZ)
        @test !isdefined(StochasticEgmUnderTest, :HSZModel)
        @test !isdefined(StochasticEgmUnderTest, :simulation_hsz)
    end

    @testset "Gauss-Hermite field adapter and moments" begin
        base = EgmGH.gauss_hermite_rule(9)
        μ, σ = 0.1, 0.25
        rule = StochasticEgmUnderTest.distribution_quadrature(
            LogNormal(μ, σ), base)

        @test length(rule.nodes) == length(base.nodes)
        @test Tuple(reinterpret(UInt64, weight) for weight in rule.weights) ==
              Tuple(reinterpret(UInt64, weight) for weight in base.weights)
        @test all(>(0.0), rule.weights)
        @test sum(rule.weights) ≈ 1.0 atol=4eps(Float64)
        @test sum(rule.weights .* rule.nodes) ≈ exp(μ + σ^2 / 2) rtol=2e-13
        @test sum(rule.weights .* rule.nodes .^ 2) ≈ exp(2μ + 2σ^2) rtol=2e-12

        @test_throws ArgumentError StochasticEgmUnderTest.EGMQuadratureRule(
            [(0.0, 1.0)], [1.0])
        @test_throws ArgumentError StochasticEgmUnderTest.EGMQuadratureRule(
            [0.0, 1.0], [1.0, -1.0])
        overflow_safe = StochasticEgmUnderTest.EGMQuadratureRule(
            [0.0, 1.0], [floatmax(Float64), floatmax(Float64)])
        @test overflow_safe.weights == (0.5, 0.5)
        @test_throws ArgumentError StochasticEgmUnderTest.EGMQuadratureRule(
            [0.0, 1.0], [nextfloat(0.0), floatmax(Float64)])
        @test_throws MethodError setindex!(overflow_safe.weights, -1.0, 1)

        # A higher-order sensitivity rule uses 31 nodes. Its extreme normal
        # nodes are beyond the range where `cdf` followed by `quantile` is a
        # finite Float64 transform, so Normal and LogNormal must use their exact
        # affine/log-affine maps.
        refined_base = EgmGH.gauss_hermite_rule(31)
        refined_normal = StochasticEgmUnderTest.distribution_quadrature(
            Normal(0.1, 0.25), refined_base)
        refined_lognormal = StochasticEgmUnderTest.distribution_quadrature(
            LogNormal(0.1, 0.25), refined_base)
        @test all(isfinite, refined_normal.nodes)
        @test all(isfinite, refined_lognormal.nodes)
        @test sum(refined_normal.weights .* refined_normal.nodes) ≈ 0.1 atol=2e-15
        @test sum(refined_normal.weights .* refined_normal.nodes .^ 2) ≈
              0.1^2 + 0.25^2 rtol=2e-14
        @test sum(refined_lognormal.weights .* refined_lognormal.nodes) ≈
              exp(0.1 + 0.25^2 / 2) rtol=2e-14
    end

    @testset "shared five-node reported quadrature is used by IID and scalar AR(1)" begin
        gh = EgmGH.gauss_hermite_rule(5)
        iid = StochasticEgmUnderTest.build_agent(
            model="iid", β=0.98, γ=1.0, r=0.02, xFloor=1.0, tMax=3,
            a=collect(range(0.0, 8.0; length=33)), d=LogNormal(0.0, 0.2))
        iid_solution = StochasticEgmUnderTest.solve_egm!(iid; quadrature=gh)
        @test iid_solution.solve_diagnostics.quadrature_expectations > 0
        @test iid_solution.solve_diagnostics.quadgk_fallbacks == 0
        @test iid_solution.solve_diagnostics.continuation_queries > 0
        @test iid_solution.solve_diagnostics.quadrature_continuation_mass > 0.0
        @test all(isfinite, StochasticEgmUnderTest.action(
            iid_solution, x; t=1) for x in (1.0, 2.0, 5.0))

        refined_iid = StochasticEgmUnderTest.build_agent(
            model="iid", β=0.98, γ=1.0, r=0.02, xFloor=1.0, tMax=2,
            a=collect(range(0.0, 8.0; length=17)), d=LogNormal(0.0, 0.2))
        refined_solution = StochasticEgmUnderTest.solve_egm!(refined_iid;
            quadrature=EgmGH.gauss_hermite_rule(31))
        @test refined_solution.solve_diagnostics.quadrature_expectations > 0
        @test refined_solution.solve_diagnostics.quadgk_fallbacks == 0
        @test isfinite(StochasticEgmUnderTest.action(
            refined_solution, 2.0; t=1))

        ar1 = StochasticEgmUnderTest.build_agent(
            model="ar1", β=0.98, γ=1.0, r=0.02, xFloor=1.0, tMax=3,
            a=collect(range(0.0, 8.0; length=33)), ρ=0.85,
            d=Normal(0.0, 0.15), y=exp.(collect(range(-0.4, 0.4; length=5))))
        ar1_solution = StochasticEgmUnderTest.solve_egm!(ar1; quadrature=gh)
        @test ar1_solution.solve_diagnostics.quadrature_expectations > 0
        @test ar1_solution.solve_diagnostics.quadgk_fallbacks == 0
        @test ar1_solution.solve_diagnostics.common_grid_projections == 2
        @test ar1_solution.solve_diagnostics.continuation_queries > 0
        @test ar1_solution.solve_diagnostics.continuation_income_clamps > 0
        @test ar1_solution.solve_diagnostics.max_continuation_log_income_clamp > 0.0
        @test 0.0 < ar1_solution.solve_diagnostics.quadrature_income_clamp_mass <=
                    ar1_solution.solve_diagnostics.quadrature_continuation_mass
        @test all(length(grid) >= 2 for grid in values(ar1_solution.state_grids))
        @test all(isfinite, StochasticEgmUnderTest.action(
            ar1_solution, 2.0, y; t=1) for y in ar1.y)

        # Direct transformed rules remain available for evaluator-controlled
        # non-Gaussian node sets.
        transformed = StochasticEgmUnderTest.distribution_quadrature(ar1.d, gh)
        @test StochasticEgmUnderTest.rhs(ar1, 1.0, 1.0; t=2,
            quadrature=transformed) > 0.0

        # A deliberately narrow asset domain must be diagnosed during backward
        # solution, before any public action query is made.
        narrow_iid = StochasticEgmUnderTest.build_agent(
            model="iid", β=0.96, γ=1.5, r=0.02, xFloor=1.0, tMax=3,
            a=collect(range(0.0, 1.0; length=9)), d=LogNormal(0.0, 0.5))
        narrow_solution = StochasticEgmUnderTest.solve_egm!(narrow_iid;
            quadrature=gh)
        @test narrow_solution.solve_diagnostics.continuation_wealth_extrapolations > 0
        @test narrow_solution.solve_diagnostics.max_continuation_wealth_extrapolation > 0.0
        @test narrow_solution.solve_diagnostics.min_continuation_upper_wealth_margin < 0.0
    end

    @testset "finite-node floor regimes never share one Euler polyline" begin
        agent = StochasticEgmUnderTest.build_agent(
            model="iid", β=0.98, γ=1.0, r=0.1, xFloor=3.0, tMax=2,
            a=collect(range(0.0, 100.0; length=400)),
            d=LogNormal(-0.1^2 / 2, 0.1))
        rule = StochasticEgmUnderTest.distribution_quadrature(
            agent.d, EgmGH.gauss_hermite_rule(5))
        kink_assets = StochasticEgmUnderTest._iid_structural_kink_assets(
            agent, rule)
        expected = sort!(unique!(Float64[
            (agent.bf.xFloor - income) / (1 + agent.bf.r)
            for income in rule.nodes
            if 0.0 < (agent.bf.xFloor - income) / (1 + agent.bf.r) <=
                     agent.bf.a[end]
        ]))
        @test kink_assets == expected

        threshold = first(kink_assets)
        left_signature = StochasticEgmUnderTest._iid_continuation_signature(
            agent, threshold - 1e-6, rule, Float64[])
        right_signature = StochasticEgmUnderTest._iid_continuation_signature(
            agent, threshold + 1e-6, rule, Float64[])
        @test left_signature != right_signature

        # These two valid regimes mimic the large endogenous-state drop when a
        # finite quadrature atom becomes nonbinding. Without regime labels the
        # middle edge would be treated as an economic polyline segment.
        x = [252.0, 260.0, 13.5, 14.0]
        c = [250.0, 257.0, 12.0, 12.2]
        v = [6.5, 6.6, 3.5, 3.6]
        valid = trues(4)
        regimes = [:one_atom, :one_atom, :two_atoms, :two_atoms]
        unsplit = StochasticEgmUnderTest._interior_candidates(
            x, c, v, valid)
        split = StochasticEgmUnderTest._interior_candidates(
            x, c, v, valid; regimes)
        @test length(unsplit) == 1
        @test length(split) == 2
        @test split[1].x == x[1:2]
        @test split[2].x == x[3:4]
        @test_throws DimensionMismatch StochasticEgmUnderTest._interior_candidates(
            x, c, v, valid; regimes=regimes[1:3])

        choices = [(0.0, 0.0), (1.0, 0.15), (2.5, 0.45), (5.0, 0.9)]
        domain, _ = StochasticEgmUnderTest._state_domain(agent)
        minimum_state = StochasticEgmUnderTest._corner_lower_bound(agent, domain)
        fixed_envelope = StochasticEgmUnderTest._fixed_saving_envelope(
            agent, choices, minimum_state, 20.0)
        fixed_saving = fixed_envelope.x .- fixed_envelope.c
        @test all(saving -> any(choice -> saving == choice[1], choices),
            fixed_saving)
        @test all(diff(fixed_saving) .>= 0.0)
        @test all(eachindex(fixed_envelope.x)) do index
            state = fixed_envelope.x[index]
            expected_value = maximum(
                StochasticEgmUnderTest._fixed_saving_value(
                    agent, state, choice...)
                for choice in choices)
            isapprox(fixed_envelope.v[index], expected_value;
                rtol=2e-12, atol=2e-12)
        end
    end

    @testset "fixed-saving hull feasibility and save-more ties" begin
        function envelope_saving(envelope, state)
            index = clamp(searchsortedlast(envelope.x, state),
                1, length(envelope.x) - 1)
            lower, upper = envelope.x[index], envelope.x[index+1]
            weight = (state - lower) / (upper - lower)
            consumption = (1 - weight) * envelope.c[index] +
                weight * envelope.c[index+1]
            return state - consumption
        end

        function brute_saving(agent, choices, state)
            values = [StochasticEgmUnderTest._fixed_saving_value(
                agent, state, choice...) for choice in choices]
            best = maximum(values)
            tied = [index for index in eachindex(choices)
                    if values[index] == best ||
                       isapprox(values[index], best;
                           rtol=StochasticEgmUnderTest.ENVELOPE_VALUE_RTOL,
                           atol=StochasticEgmUnderTest.ENVELOPE_VALUE_ATOL)]
            return maximum(choices[index][1] for index in tied)
        end

        # Every saving choice is feasible immediately to the right of x=a.
        # The continuation intercept is calibrated so that the new choice wins
        # at a+1e-6 for each boundary value of risk aversion used by the contest.
        for gamma in (0.5, 1.0, 5.0)
            agent = StochasticEgmUnderTest.build_agent(
                model="iid", β=0.98, γ=gamma, r=0.1, xFloor=0.0,
                tMax=2, a=[0.0, 0.5, 1.0, 2.0], d=Uniform(0.5, 1.5))
            state = 1.0 + 1e-6
            incumbent = (0.0, 0.0)
            base_new = (1.0, 0.0)
            incumbent_value = StochasticEgmUnderTest._fixed_saving_value(
                agent, state, incumbent...)
            base_new_value = StochasticEgmUnderTest._fixed_saving_value(
                agent, state, base_new...)
            # A relative margin avoids losing the intended strict improvement
            # to cancellation when gamma=5 makes near-zero utility about -1e23.
            margin = 1e-6 * max(1.0, abs(base_new_value))
            advantage = (incumbent_value - base_new_value + margin) /
                agent.bf.β
            choices = [incumbent, (1.0, advantage)]
            domain, _ = StochasticEgmUnderTest._state_domain(agent)
            minimum_state = StochasticEgmUnderTest._corner_lower_bound(
                agent, domain)
            envelope = StochasticEgmUnderTest._fixed_saving_envelope(
                agent, choices, minimum_state, 4.0)

            @test envelope_saving(envelope, 1.0) ≈ 0.0 atol=4eps(1.0)
            @test envelope_saving(envelope, state) ≈ 1.0 atol=2e-10
            @test StochasticEgmUnderTest._fixed_saving_right_wins(
                agent, state, choices...)
        end

        # Exercise unsorted inputs, duplicate saving, dominated candidates, and
        # stack pops by comparing the hull policy with an exhaustive choice set.
        agent = StochasticEgmUnderTest.build_agent(
            model="iid", β=0.98, γ=1.0, r=0.1, xFloor=0.0,
            tMax=2, a=[0.0, 0.5, 1.0, 2.0], d=Uniform(0.5, 1.5))
        choices = [(3.0, 4.0), (0.0, 0.0), (1.0, 1.0),
                   (2.0, -2.0), (1.0, 0.5), (4.0, 5.0),
                   (6.0, 0.5)]
        domain, _ = StochasticEgmUnderTest._state_domain(agent)
        minimum_state = StochasticEgmUnderTest._corner_lower_bound(agent, domain)
        envelope = StochasticEgmUnderTest._fixed_saving_envelope(
            agent, choices, minimum_state, 20.0)
        states = sort!(unique!(vcat(
            collect(range(minimum_state, 20.0; length=401)),
            first.(choices), nextfloat.(first.(choices)))))
        filter!(state -> minimum_state <= state <= 20.0, states)
        @test all(states) do state
            isapprox(envelope_saving(envelope, state),
                brute_saving(agent, choices, state); atol=2e-9, rtol=0.0)
        end

        # A slightly lower value within the common tolerance is an explicit tie,
        # so the higher-saving action must win even at the upper endpoint.
        endpoint = 10.0
        left = (0.0, 0.0)
        delta = 0.25e-12
        right_continuation = (
            StochasticEgmUnderTest._fixed_saving_value(
                agent, endpoint, left...) -
            StochasticEgmUnderTest._fixed_saving_value(
                agent, endpoint, 1.0, 0.0) - delta) / agent.bf.β
        right = (1.0, right_continuation)
        @test StochasticEgmUnderTest._fixed_saving_right_wins(
            agent, endpoint, left, right)
        endpoint_envelope = StochasticEgmUnderTest._fixed_saving_envelope(
            agent, [left, right], minimum_state, endpoint)
        @test envelope_saving(endpoint_envelope, endpoint) ≈ 1.0 atol=2e-10

        @test_throws ArgumentError StochasticEgmUnderTest._fixed_saving_envelope(
            agent, [left], 2.0, 1.0)
    end

    @testset "adaptive IID integration uses the bounded Bellman envelope" begin
        agent = StochasticEgmUnderTest.build_agent(
            model="iid", β=0.96, γ=1.5, r=0.02, xFloor=1.0,
            tMax=3, a=collect(range(0.0, 4.0; length=31)),
            d=Uniform(0.5, 1.5))
        solution = StochasticEgmUnderTest.solve_egm!(agent)
        _, state_cap = StochasticEgmUnderTest._state_domain(agent)
        @test solution.solve_diagnostics.quadgk_fallbacks > 0
        @test solution.solve_diagnostics.quadrature_expectations == 0
        @test last(solution.state_grids[1]) == state_cap
        @test all(0.0 <= StochasticEgmUnderTest.action(
            solution, wealth; t=1) <= wealth for wealth in (1.0, 2.0, 8.0))
    end

    @testset "long-horizon IID Bellman-envelope correction" begin
        base_rule = EgmGH.gauss_hermite_rule(5)
        distribution = LogNormal(-0.5^2 / 2, 0.5)
        agent = StochasticEgmUnderTest.build_agent(
            model="iid", β=0.98, γ=1.0, r=0.1, xFloor=3.0, tMax=50,
            a=collect(range(0.0, 100.0; length=400)), d=distribution)
        rule = StochasticEgmUnderTest.distribution_quadrature(
            distribution, base_rule)
        solution = StochasticEgmUnderTest.solve_egm!(agent; quadrature=rule)

        @test last(solution.state_grids[1]) == 400.0
        @test solution.solve_diagnostics.continuation_wealth_extrapolations == 0
        discontinuities = StochasticEgmUnderTest._iid_policy_discontinuities(
            agent, 2)
        @test !isempty(discontinuities)
        child_grid = solution.state_grids[2]
        child_consumption = agent.bf.policy[2](child_grid)
        boundary_indices = searchsortedfirst.(Ref(child_grid), discontinuities)
        @test all(>(1), boundary_indices)
        @test all(zip(discontinuities, boundary_indices)) do (boundary, index)
            child_grid[index] - child_grid[index-1] <=
                8eps(max(abs(boundary), 1.0))
        end
        @test all(boundary_indices) do index
            !isapprox(child_consumption[index-1], child_consumption[index];
                rtol=1e-10, atol=1e-12)
        end

        function bellman_objective(wealth, consumption)
            saving = wealth - consumption
            continuation = sum(weight * StochasticEgmUnderTest.v_x(agent,
                    max((1 + agent.bf.r) * saving + income,
                        agent.bf.xFloor); t=2)
                for (income, weight) in zip(rule.nodes, rule.weights))
            return log(consumption) + agent.bf.β * continuation
        end

        # Before the global fixed-saving envelope, the x=50 residual was about
        # 0.23 utility units. A 400-point corrected solve keeps representative
        # long-horizon residuals below five thousandths under its own V_{t+1}.
        for wealth in (13.0, 20.0, 50.0)
            solved_consumption = StochasticEgmUnderTest.action(
                solution, wealth; t=1)
            consumption_grid = range(wealth / 20_000, wealth; length=20_001)
            direct_maximum = maximum(
                bellman_objective(wealth, consumption)
                for consumption in consumption_grid)
            solved_value = bellman_objective(wealth, solved_consumption)
            @test direct_maximum - solved_value <= 5e-3
        end
    end

    @testset "AR(1) simulation process is the solved process" begin
        d = Normal(0.0, 0.2)
        ar1 = StochasticEgmUnderTest.build_agent(
            model="ar1", β=0.98, γ=1.0, r=0.0, xFloor=1.0, tMax=2,
            a=[0.0, 1.0], ρ=0.9, d=d, y=[0.5, 1.5])

        derived = StochasticEgmUnderTest.AR1Model(ar1, [1.0, 2.0], [0.8, 1.2])
        legacy = StochasticEgmUnderTest.AR1Model(
            ar1, [1.0, 2.0], [0.8, 1.2], ar1.ρ, deepcopy(ar1.d))
        @test derived.ρ == legacy.ρ == ar1.ρ
        @test isequal(derived.d, legacy.d) && isequal(legacy.d, ar1.d)
        _, paths = StochasticEgmUnderTest.init_simulation(derived, seed=9)
        @test paths['y'][:, 1] ≈ [0.8, 1.2]
        @test all(isfinite, paths['y'])
        @test_throws ArgumentError StochasticEgmUnderTest.AR1Model(
            ar1, [1.0], [1.0], 0.8, ar1.d)
        @test_throws ArgumentError StochasticEgmUnderTest.AR1Model(
            ar1, [1.0], [1.0], ar1.ρ, Normal(1.0, 0.2))
        @test_throws ArgumentError StochasticEgmUnderTest.AR1Model(
            ar1, [1.0, 2.0], [1.0])
        @test_throws ArgumentError StochasticEgmUnderTest.AR1Model(
            ar1, [NaN], [1.0])
        @test_throws ArgumentError StochasticEgmUnderTest.AR1Model(
            ar1, [1.0], [0.0])
    end

    @testset "external paths are deterministic and RNG-free" begin
        agent = StochasticEgmUnderTest.build_agent(
            model="iid", β=0.98, γ=1.0, r=0.01, xFloor=1.0, tMax=3,
            a=collect(range(0.0, 6.0; length=25)), d=LogNormal(0.0, 0.2))
        model = StochasticEgmUnderTest.IIDModel(agent, [1.0, 2.0])
        solution = StochasticEgmUnderTest.solve_egm!(agent;
            quadrature=EgmGH.gauss_hermite_rule(5))
        paths = [1.0 1.1 0.9; 1.2 0.8 1.3]
        original = copy(paths)

        expected_rng = copy(Random.default_rng())
        first_result = StochasticEgmUnderTest.evaluate_paths(model, paths, solution)
        second_result = StochasticEgmUnderTest.evaluate_paths(model, paths, solution)
        @test rand() == rand(expected_rng)
        @test paths == original
        @test first_result['x'] == second_result['x']
        @test first_result['c'] == second_result['c']
        @test first_result['y'] == original
        @test first_result['y'] !== paths

        ar1 = StochasticEgmUnderTest.build_agent(
            model="ar1", β=0.98, γ=1.0, r=0.01, xFloor=1.0, tMax=2,
            a=collect(range(0.0, 4.0; length=17)), ρ=0.8,
            d=Normal(0.0, 0.1), y=[0.7, 1.0, 1.4])
        ar1_model = StochasticEgmUnderTest.AR1Model(ar1, [1.0], [1.0])
        ar1_solution = StochasticEgmUnderTest.solve_egm!(ar1;
            quadrature=EgmGH.gauss_hermite_rule(5))
        @test_throws ArgumentError StochasticEgmUnderTest.evaluate_paths(
            ar1_model, [1.1 1.2], ar1_solution)
    end

    @testset "material extrapolation and clamp diagnostics" begin
        agent = StochasticEgmUnderTest.build_agent(
            model="ar1", β=0.98, γ=1.0, r=0.01, xFloor=1.0, tMax=3,
            a=collect(range(0.0, 6.0; length=25)), ρ=0.8,
            d=Normal(0.0, 0.1), y=[0.7, 1.0, 1.4])
        solution = StochasticEgmUnderTest.solve_egm!(agent;
            quadrature=EgmGH.gauss_hermite_rule(5))
        grid = solution.state_grids[1]
        StochasticEgmUnderTest.reset_query_diagnostics!(solution)

        StochasticEgmUnderTest.action(solution, 2.0, 1.0; t=1)
        StochasticEgmUnderTest.value_at(solution, 2.0, 1.0; t=1)
        @test solution.query_diagnostics.material_wealth_extrapolations == 0
        @test solution.query_diagnostics.material_income_clamps == 0
        @test isfinite(solution.query_diagnostics.min_lower_wealth_margin)
        @test isfinite(solution.query_diagnostics.min_upper_wealth_margin)

        StochasticEgmUnderTest.action(solution, last(grid), 1.0; t=1)
        @test solution.query_diagnostics.upper_wealth_boundary_contacts == 1
        StochasticEgmUnderTest.action(solution, nextfloat(last(grid)), 1.0; t=1)
        @test solution.query_diagnostics.material_wealth_extrapolations == 0
        @test solution.query_diagnostics.upper_wealth_boundary_contacts == 2
        StochasticEgmUnderTest.action(solution, 2.0, last(agent.y); t=1)
        @test solution.query_diagnostics.upper_income_boundary_contacts == 1
        StochasticEgmUnderTest.action(solution, last(grid) + 1.0, 2.0; t=1)
        @test solution.query_diagnostics.material_wealth_extrapolations == 1
        @test solution.query_diagnostics.material_income_clamps == 1
        @test solution.query_diagnostics.max_wealth_extrapolation ≈ 1.0
        @test solution.query_diagnostics.max_income_clamp ≈ 0.6
        @test solution.query_diagnostics.min_upper_wealth_margin ≈ -1.0
        @test solution.query_diagnostics.min_upper_income_margin ≈ -0.6

        reset = StochasticEgmUnderTest.reset_query_diagnostics!(solution)
        @test reset.consumption_queries == 0
        @test reset.value_queries == 0
        @test reset.material_wealth_extrapolations == 0
        @test reset.upper_wealth_boundary_contacts == 0
        @test reset.material_income_clamps == 0
        @test reset.upper_income_boundary_contacts == 0
        @test isinf(reset.min_lower_wealth_margin)
        @test isinf(reset.min_upper_income_margin)

        # Re-solving the mutable agent invalidates older query adapters instead
        # of silently combining old grids/diagnostics with a new policy.
        StochasticEgmUnderTest.solve_egm!(agent;
            quadrature=EgmGH.gauss_hermite_rule(5))
        @test_throws ErrorException StochasticEgmUnderTest.action(
            solution, 2.0, 1.0; t=1)
    end
end
