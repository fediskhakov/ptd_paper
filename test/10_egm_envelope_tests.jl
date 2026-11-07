# Tests for the DCEGM envelope and model modules (src/10-src/13).
using Distributions
using Random

const EgmUnderTest = Ptd.DcEgm

@testset verbose = true "Pure-Julia DCEGM" begin
    @testset "package API" begin
        @test isdefined(EgmUnderTest, :egm!)
        @test isdefined(EgmUnderTest, :solve_egm!)
        @test isdefined(EgmUnderTest, :evaluate_paths)
        @test isdefined(EgmUnderTest, :upper_envelope)
        @test :EGMSolution in names(EgmUnderTest)
        @test :EGMQuadratureRule in names(EgmUnderTest)
        @test :action in names(EgmUnderTest)
        @test :value_at in names(EgmUnderTest)
        @test :reset_query_diagnostics! in names(EgmUnderTest)
    end

    @testset "upper-envelope welfare ties save more" begin
        high_consumption = EgmUnderTest.Grids1d(
            [0.0, 1.0], [0.8, 0.8], [2.0, 2.0])
        low_consumption = EgmUnderTest.Grids1d(
            [0.0, 1.0], [0.2, 0.2], [2.0, 2.0])
        envelope = EgmUnderTest.upper_envelope(
            [high_consumption, low_consumption])
        @test envelope.c == [0.2, 0.2]

        # The branch winning immediately right of the crossing consumes more.
        # At the exact equal-value state, the outgoing saving-intensive policy
        # must nevertheless be retained.
        saver = EgmUnderTest.Grids1d(
            [0.0, 1.0], [0.2, 0.2], [1.0, 0.0])
        spender = EgmUnderTest.Grids1d(
            [0.0, 1.0], [0.8, 0.8], [0.0, 1.0])
        crossing = EgmUnderTest.upper_envelope([saver, spender])
        exact = findfirst(==(0.5), crossing.x)
        @test !isnothing(exact)
        @test crossing.c[exact] == 0.2

        # The saver can touch the upper envelope at one structural knot while
        # the spender wins on both sides. The exact knot must not disappear
        # when adjacent open intervals have the same winner.
        flat_spender = EgmUnderTest.Grids1d(
            [0.0, 1.0], [0.8, 0.8], [1.0, 1.0])
        isolated_saver = EgmUnderTest.Grids1d(
            [0.0, 0.5, 1.0], [0.2, 0.2, 0.2], [0.0, 1.0, 0.0])
        isolated = EgmUnderTest.upper_envelope([flat_spender, isolated_saver])
        isolated_exact = findfirst(==(0.5), isolated.x)
        @test !isnothing(isolated_exact)
        @test isolated.c[isolated_exact] == 0.2
        @test EgmUnderTest.linear_interpolation(isolated.x, isolated.c)(0.25) ≈ 0.8
        @test EgmUnderTest.linear_interpolation(isolated.x, isolated.c)(0.75) ≈ 0.8

        # Exact save-more ties at the global domain endpoints need the same
        # treatment as interior structural knots.
        left_saver = EgmUnderTest.Grids1d(
            [0.0, 1.0], [0.2, 0.2], [1.0, 0.0])
        left_tie = EgmUnderTest.upper_envelope([flat_spender, left_saver])
        @test first(left_tie.x) == 0.0
        @test first(left_tie.c) == 0.2
        @test EgmUnderTest.linear_interpolation(left_tie.x, left_tie.c)(0.25) ≈ 0.8

        right_saver = EgmUnderTest.Grids1d(
            [0.0, 1.0], [0.2, 0.2], [0.0, 1.0])
        right_tie = EgmUnderTest.upper_envelope([flat_spender, right_saver])
        @test last(right_tie.x) == 1.0
        @test last(right_tie.c) == 0.2
        @test EgmUnderTest.linear_interpolation(right_tie.x, right_tie.c)(0.75) ≈ 0.8
    end

    @testset "upper envelope keeps a hidden third branch" begin
        # A and C win at the interval endpoints, while B is strictly best in
        # the middle. The previous implementation intersected only A and C.
        branch_a = EgmUnderTest.Grids1d([0.0, 1.0], [0.1, 0.1], [3.0, 0.0])
        branch_b = EgmUnderTest.Grids1d([0.0, 1.0], [0.5, 0.5], [2.0, 2.0])
        branch_c = EgmUnderTest.Grids1d([0.0, 1.0], [0.9, 0.9], [0.0, 3.0])
        envelope = EgmUnderTest.upper_envelope([branch_a, branch_b, branch_c])
        policy = EgmUnderTest.linear_interpolation(envelope.x, envelope.c)
        value = EgmUnderTest.linear_interpolation(envelope.x, envelope.v)

        @test policy(0.2) ≈ 0.1
        @test policy(0.5) ≈ 0.5
        @test policy(0.8) ≈ 0.9
        @test value(0.5) ≈ 2.0
        @test length(envelope.x) >= 8
    end

    @testset "upper envelope matches all pairwise affine crossings" begin
        rng = MersenneTwister(20260804)
        states = collect(range(0.0, 1.0; length=101))
        for _ in 1:25
            candidates = [EgmUnderTest.Grids1d([0.0, 1.0], fill(rand(rng), 2),
                randn(rng, 2)) for _ in 1:6]
            envelope = EgmUnderTest.upper_envelope(candidates)
            value = EgmUnderTest.linear_interpolation(envelope.x, envelope.v)
            for x in states
                reference = maximum(candidate.v[1] +
                    x * (candidate.v[2] - candidate.v[1]) for candidate in candidates)
                @test value(x) ≈ reference rtol = 2e-13 atol = 2e-13
            end
        end
    end

    @testset "loopback split repeats turning points" begin
        loopback = EgmUnderTest.Grids1d(
            [0.0, 2.0, 1.0, 3.0], [0.0, 0.4, 0.6, 1.0], [0.0, 2.0, 1.5, 3.0])
        branches = EgmUnderTest.partition(loopback)
        @test length(branches) == 3
        @test branches[1].x == [0.0, 2.0]
        @test branches[2].x == [1.0, 2.0]
        @test branches[3].x == [1.0, 3.0]
    end

    @testset "hard-floor IID integration" begin
        agent = EgmUnderTest.build_agent(model="iid", β=0.98, γ=1.0, r=0.0,
            xFloor=1.0, tMax=2, a=collect(range(0.0, 2.0; length=21)),
            d=Uniform(0.0, 2.0))
        marginal = EgmUnderTest.rhs(agent, 0.5, t=1)
        expected_marginal = 0.98 / 2 * log(2.5)
        @test marginal ≈ expected_marginal rtol = 2e-12

        continuation = EgmUnderTest.ev_next(agent, 0.5, t=1)
        expected_continuation = (2.5 * log(2.5) - 1.5) / 2
        @test continuation ≈ expected_continuation rtol = 2e-12
    end

    @testset "remote Euler tail does not distort the corner branch" begin
        # With a tightly concentrated lognormal income, the nonbinding tail at
        # a=0 is genuine but tiny. Its inverse-Euler point is around 1e28 and
        # must not determine the resolution of the economically relevant grid.
        agent = EgmUnderTest.build_agent(model="iid", β=0.98, γ=1.0, r=0.1,
            xFloor=3.0, tMax=2, a=collect(range(0.0, 100.0; length=200)),
            d=LogNormal(0.0, 0.1))
        @test 0.0 < EgmUnderTest.rhs(agent, 0.0, t=1) < 1e-20
        continuation = EgmUnderTest.ev_next(agent, 0.0, t=1)
        EgmUnderTest.egm!(agent)

        for x in (3.0, 6.0, 9.0)
            @test EgmUnderTest.c_x(agent, x, t=1) ≈ x atol = 2e-12
            cornerValue = EgmUnderTest.u(agent, x) + agent.bf.β * continuation
            @test EgmUnderTest.v_x(agent, x, t=1) ≈ cornerValue atol = 5e-4
        end
        @test EgmUnderTest.c_x(agent, 10.0, t=1) < 10.0
    end

    @testset "corner grid has no roundoff tail beyond its endpoint" begin
        agent = EgmUnderTest.build_agent(model="deterministic", β=0.98,
            γ=1.0, r=0.1, xFloor=3.0, tMax=2,
            a=collect(range(0.0, 100.0; length=200)), incomes=ones(2))
        domain, _ = EgmUnderTest._state_domain(agent)
        lower = EgmUnderTest._corner_lower_bound(agent, domain)
        upper = 111.12740174696562
        corner = EgmUnderTest._corner_grid(agent, lower, upper)

        @test first(corner) == lower
        @test last(corner) == upper
        @test all((lower .<= corner) .& (corner .<= upper))

        # The interior value is deliberately dominant at the upper endpoint.
        # A one-ULP corner tail would nevertheless become the last segment and
        # make Line() extrapolate the spurious consume-all policy.
        envelope = EgmUnderTest._with_corner(agent, [5.0, upper], [1.0, 11.0],
            [100.0, 100.0], 0.0)
        @test last(envelope.x) == upper
        @test last(envelope.c) ≈ 11.0
    end

    @testset "singularity-safe corner grid for general CRRA" begin
        for γ in (2.0, 3.0, 100.0)
            agent = EgmUnderTest.build_agent(model="deterministic", β=0.98,
                γ=γ, r=0.1, xFloor=3.0, tMax=2,
                a=collect(range(0.0, 20.0; length=41)), incomes=ones(2))
            EgmUnderTest.egm!(agent)

            for x in (0.1, 1.0, 3.0, 5.0)
                consumption = EgmUnderTest.c_x(agent, x, t=1)
                @test isfinite(consumption)
                @test 0.0 < consumption <= x
                @test isfinite(EgmUnderTest.v_x(agent, x, t=1))
            end
            exactCorner = EgmUnderTest.u(agent, 1.0) +
                          agent.bf.β * EgmUnderTest.u(agent, agent.bf.xFloor)
            @test EgmUnderTest.v_x(agent, 1.0, t=1) ≈ exactCorner rtol = 2e-13
        end
    end

    @testset "solver feasibility and separate corner branch" begin
        agent = EgmUnderTest.build_agent(model="deterministic", β=0.98, γ=1.0,
            r=0.1, xFloor=3.0, tMax=4,
            a=collect(range(0.0, 20.0; length=41)), incomes=ones(4))
        EgmUnderTest.egm!(agent)
        for x in range(0.1, 20.0; length=101)
            c = EgmUnderTest.c_x(agent, x, t=1)
            @test 0.0 < c <= x
        end
    end

    @testset "deterministic DCEGM welfare refines toward exact PTD" begin
        horizon = 8
        ptd = Ptd.DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0,
            incomes=ones(1, horizon))
        Ptd.deterministic_ptd!(ptd)
        coarse = EgmUnderTest.build_agent(model="deterministic", β=0.98, γ=1.0,
            r=0.1, xFloor=3.0, tMax=horizon,
            a=collect(range(0.0, 80.0; length=101)), incomes=ones(horizon))
        fine = EgmUnderTest.build_agent(model="deterministic", β=0.98, γ=1.0,
            r=0.1, xFloor=3.0, tMax=horizon,
            a=collect(range(0.0, 80.0; length=401)), incomes=ones(horizon))
        EgmUnderTest.egm!(coarse)
        EgmUnderTest.egm!(fine)

        function egm_welfare(agent, x)
            value = 0.0
            for t in 1:horizon
                c = EgmUnderTest.c_x(agent, x, t=t)
                value += agent.bf.β^(t - 1) * EgmUnderTest.u(agent, c)
                t < horizon && (x = EgmUnderTest.xNext_a(agent, x - c, t=t))
            end
            return value
        end

        states = range(0.2, 30.0; length=151)
        coarseLosses = [Ptd.v_x(ptd, x, i=1, t=1) - egm_welfare(coarse, x)
                        for x in states]
        fineLosses = [Ptd.v_x(ptd, x, i=1, t=1) - egm_welfare(fine, x)
                      for x in states]
        @test minimum(coarseLosses) >= -2e-11
        @test minimum(fineLosses) >= -2e-11
        @test sum(fineLosses) / length(fineLosses) <
              sum(coarseLosses) / length(coarseLosses)
        @test maximum(fineLosses) < maximum(coarseLosses)
        @test sum(fineLosses) / length(fineLosses) < 5e-4
    end

    @testset "long-horizon extrapolation preserves deterministic welfare" begin
        horizon = 50
        ptd = Ptd.DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0,
            incomes=ones(1, horizon))
        Ptd.deterministic_ptd!(ptd)
        agent = EgmUnderTest.build_agent(model="deterministic", β=0.98,
            γ=1.0, r=0.1, xFloor=3.0, tMax=horizon,
            a=collect(range(0.0, 100.0; length=200)), incomes=ones(horizon))
        EgmUnderTest.egm!(agent)

        initialWealth = 50.0
        wealth = initialWealth
        peakWealth = wealth
        welfare = 0.0
        for t in 1:horizon
            consumption = EgmUnderTest.c_x(agent, wealth, t=t)
            welfare += agent.bf.β^(t - 1) * EgmUnderTest.u(agent, consumption)
            if t < horizon
                wealth = EgmUnderTest.xNext_a(agent, wealth - consumption, t=t)
                peakWealth = max(peakWealth, wealth)
            end
        end

        @test peakWealth > agent.bf.a[end]
        @test welfare ≈ Ptd.v_x(ptd, initialWealth, i=1, t=1) atol = 2e-10
    end

    @testset "simulation uses an isolated RNG" begin
        agent = EgmUnderTest.build_agent(model="iid", β=0.98, γ=1.0, r=0.0,
            xFloor=1.0, tMax=2, a=collect(range(0.0, 2.0; length=11)),
            d=Uniform(0.0, 2.0))
        model = EgmUnderTest.IIDModel(agent, [1.0, 1.5])
        expected_rng = copy(Random.default_rng())
        EgmUnderTest.init_simulation(model, seed=7)
        @test rand() == rand(expected_rng)
    end
end
