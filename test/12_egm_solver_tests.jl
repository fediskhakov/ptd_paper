# DCEGM solver correctness tests (src/dcegm/solver.jl and src/dcegm/simulator.jl).
using Distributions

const EgmCorrectnessUnderTest = Ptd.DcEgm


"""Directly scan the horizon-two consumption choice on a dense grid."""
function _dense_consumption_maximum(objective, wealth::Float64;
    point_count::Int=20_001)

    point_count >= 2 || throw(ArgumentError("A dense scan needs at least two points."))
    consumption = range(wealth / point_count, wealth; length=point_count)
    objectives = map(objective, consumption)
    winner = argmax(objectives)
    return (consumption=consumption[winner], objective=objectives[winner],
        step=Float64(step(consumption)))
end


function _iid_terminal_objective(agent, rule, wealth::Float64,
    consumption::Float64)

    assets = wealth - consumption
    gross_return = 1.0 + agent.bf.r
    continuation = sum(weight * EgmCorrectnessUnderTest.u(agent,
            max(gross_return * assets + income, agent.bf.xFloor))
        for (income, weight) in zip(rule.nodes, rule.weights))
    return EgmCorrectnessUnderTest.u(agent, consumption) +
           agent.bf.β * continuation
end


function _ar1_terminal_objective(agent, rule, wealth::Float64,
    income_now::Float64, consumption::Float64)

    assets = wealth - consumption
    gross_return = 1.0 + agent.bf.r
    continuation = sum(weight * EgmCorrectnessUnderTest.u(agent,
            max(gross_return * assets +
                exp(agent.ρ * log(income_now) + innovation), agent.bf.xFloor))
        for (innovation, weight) in zip(rule.nodes, rule.weights))
    return EgmCorrectnessUnderTest.u(agent, consumption) +
           agent.bf.β * continuation
end


@testset verbose = true "DCEGM direct Bellman correctness" begin
    finite_gh = Ptd.gauss_hermite_rule(7)
    asset_grid = collect(range(0.0, 8.0; length=201))

    @testset "horizon-two IID floor and interior choices" begin
        agent = EgmCorrectnessUnderTest.build_agent(
            model="iid", β=0.96, γ=1.5, r=0.03, xFloor=1.5, tMax=2,
            a=copy(asset_grid), d=LogNormal(-0.2, 0.15))
        solution = EgmCorrectnessUnderTest.solve_egm!(agent;
            quadrature=finite_gh)
        rule = EgmCorrectnessUnderTest.distribution_quadrature(
            agent.d, finite_gh)
        asset_step = maximum(diff(agent.bf.a))

        for (wealth, is_corner) in ((1.2, true), (4.0, false))
            objective = consumption -> _iid_terminal_objective(
                agent, rule, wealth, consumption)
            reference = _dense_consumption_maximum(objective, wealth)
            consumption = EgmCorrectnessUnderTest.action(
                solution, wealth; t=1)
            solved_objective = objective(consumption)
            stored_value = EgmCorrectnessUnderTest.value_at(
                solution, wealth; t=1)

            # One exogenous asset cell plus one scan cell covers both EGM
            # interpolation and discretization of the independent maximizer.
            policy_tolerance = asset_step + reference.step
            @test abs(consumption - reference.consumption) <= policy_tolerance
            @test abs(solved_objective - reference.objective) <= 1e-5
            @test isapprox(stored_value, reference.objective;
                atol=1e-4, rtol=5e-5)

            if is_corner
                @test reference.consumption == wealth
                @test consumption == wealth
                @test all(income <= agent.bf.xFloor for income in rule.nodes)
            else
                @test reference.consumption < wealth - 0.25
                @test consumption < wealth - 0.25
            end
        end
    end

    @testset "horizon-two scalar AR(1) floor and interior choices" begin
        agent = EgmCorrectnessUnderTest.build_agent(
            model="ar1", β=0.96, γ=1.5, r=0.03, xFloor=1.5, tMax=2,
            a=copy(asset_grid), ρ=0.85, d=Normal(0.0, 0.15),
            y=exp.(collect(range(log(0.65), log(1.5); length=7))))
        solution = EgmCorrectnessUnderTest.solve_egm!(agent;
            quadrature=finite_gh)
        rule = EgmCorrectnessUnderTest.distribution_quadrature(
            agent.d, finite_gh)
        income_now = first(agent.y)
        asset_step = maximum(diff(agent.bf.a))

        for (wealth, is_corner) in ((1.2, true), (4.0, false))
            objective = consumption -> _ar1_terminal_objective(
                agent, rule, wealth, income_now, consumption)
            reference = _dense_consumption_maximum(objective, wealth)
            consumption = EgmCorrectnessUnderTest.action(
                solution, wealth, income_now; t=1)
            solved_objective = objective(consumption)
            stored_value = EgmCorrectnessUnderTest.value_at(
                solution, wealth, income_now; t=1)

            policy_tolerance = asset_step + reference.step
            @test abs(consumption - reference.consumption) <= policy_tolerance
            @test abs(solved_objective - reference.objective) <= 1e-5
            @test isapprox(stored_value, reference.objective;
                atol=1e-4, rtol=5e-5)

            if is_corner
                @test reference.consumption == wealth
                @test consumption == wealth
                @test all(exp(agent.ρ * log(income_now) + innovation) <=
                          agent.bf.xFloor for innovation in rule.nodes)
            else
                @test reference.consumption < wealth - 0.25
                @test consumption < wealth - 0.25
            end
        end
    end

    @testset "adaptive-QuadGK fallback is explicit" begin
        iid = EgmCorrectnessUnderTest.build_agent(
            model="iid", β=0.96, γ=1.5, r=0.02, xFloor=1.0, tMax=2,
            a=collect(range(0.0, 4.0; length=31)), d=Uniform(0.5, 1.5))
        iid_solution = EgmCorrectnessUnderTest.solve_egm!(iid)
        @test iid_solution.solve_diagnostics.quadgk_fallbacks > 0
        @test iid_solution.solve_diagnostics.quadrature_expectations == 0

        ar1 = EgmCorrectnessUnderTest.build_agent(
            model="ar1", β=0.96, γ=1.5, r=0.02, xFloor=1.0, tMax=2,
            a=collect(range(0.0, 4.0; length=31)), ρ=0.8,
            d=Normal(0.0, 0.12),
            y=exp.(collect(range(log(0.7), log(1.4); length=5))))
        ar1_solution = EgmCorrectnessUnderTest.solve_egm!(ar1)
        @test ar1_solution.solve_diagnostics.quadgk_fallbacks > 0
        @test ar1_solution.solve_diagnostics.quadrature_expectations == 0
    end
end
