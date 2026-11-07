# Tests for src/ptd/deterministic_model.jl and src/ptd/deterministic_solver.jl.
using Random
import Ptd: PolicySegment, DeterministicProblem, u, mu, inv_mu, c_x, zNext_a,
    xNext_a, v_x, descendant_operator, last_overlapping_segment, deterministic_ptd!, fusion_boundary,
    validate_policy_collection, _assert_ordered_saving,
    _coalesce_duplicate_segments, _validate_ordered_candidates,
    _assert_representable_value_order, _within_domain_ulps, pairwise_fusion_step_bounded,
    pairwise_fusion_step_unbounded

@testset verbose = true "Deterministic Model PTD" begin

    @testset "PolicySegment" begin
        ps = PolicySegment(xStart=0.0, xEnd=Inf, b0=0.0, b1=1.0)
        @test ps.xStart == 0.0 && ps.xEnd == Inf && ps.b0 == 0.0 && ps.b1 == 1.0
    end


    @testset "DeterministicProblem constructor" begin
        incomes = fill(1.0, (2, 50))
        incomes[2, :] .= 3.0
        agentD1 = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0, incomes=incomes)
        agentD2 = DeterministicProblem(β=0.98, γ=2.0, r=0.1, xFloor=3.0, incomes=incomes)

        @test agentD1.logType == true && agentD1.tMax == 50
        @test agentD2.logType == false && agentD2.tMax == 50
        @test agentD1.policy[1, end] !== agentD1.policy[2, end]
        @test agentD1.policy[1, end] !== agentD2.policy[1, end]
        @test_throws ArgumentError DeterministicProblem(β=Inf, γ=1.0, r=0.1,
            xFloor=3.0, incomes=incomes)
        @test_throws ArgumentError DeterministicProblem(β=0.98, γ=1.0, r=0.1,
            xFloor=Inf, incomes=incomes)
    end

    @testset "utility related functions" begin
        incomes = fill(1.0, (2, 50))
        incomes[2, :] .= 3.0
        agentD1 = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0, incomes=incomes)
        agentD2 = DeterministicProblem(β=0.98, γ=2.0, r=0.1, xFloor=3.0, incomes=incomes)

        @test all([typeof(u(agentD1, 1.0)) == Float64, u(agentD1, 1.0) == 0.0])
        @test u(agentD1, [1.0, 2.0, ℯ]) == log.([1.0, 2.0, ℯ])
        @test u(agentD2, [1.0, 2.0, 3.0]) == 1.0 .- 1.0 ./ [1.0, 2.0, 3.0]

        @test mu(agentD1, [1.0, 2.0, 3.0]) == 1.0 ./ [1.0, 2.0, 3.0]
        @test mu(agentD2, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0] .^ (-2)

        @test inv_mu(agentD1, [1.0, 2.0, 3.0]) == 1 ./ [1.0, 2.0, 3.0]
        @test inv_mu(agentD2, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0] .^ (-1.0 / 2.0)

    end

    @testset "consumption policy" begin
        # method 1: consumption on a policy segment
        ps = PolicySegment(xStart=0.0, xEnd=Inf, b0=0.0, b1=1.0)
        @test c_x(ps, 60.0) == 60.0

        # method 2: consumption on time t policy
        ps1 = PolicySegment(xStart=0.0, xEnd=100.0, b0=0.0, b1=1.0)
        ps2 = PolicySegment(xStart=100.0, xEnd=Inf, b0=50.0, b1=0.5)
        policy_t = [ps1, ps2]
        @test c_x(policy_t, 150.0) == 50.0 + 0.5 * 150.0

        # method 3: consumption of agentD1 at time T, state i  
        # the default polict at T is consuming all the wealth
        incomes = fill(1.0, (2, 50))
        incomes[2, :] .= 3.0
        agentD1 = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0, incomes=incomes)
        @test c_x(agentD1, 10.0, i=1, t=50) == 10.0
    end

    @testset "wealth dynamics" begin
        incomes = fill(1.0, (2, 50))
        incomes[2, :] .= 3.0
        agentD1 = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0, incomes=incomes)

        ## without social insurance
        @test zNext_a(agentD1, 10.0; i=1, t=49) == 10.0 * (1 + 0.1) + 1.0
        @test zNext_a(agentD1, 10.0; i=2, t=49) == 10.0 * (1 + 0.1) + 3.0

        ## with social insurance
        @test xNext_a(agentD1, 1.0; i=1, t=49) == max(1.0 * (1 + 0.1) + 1.0, 3.0) == 3.0
        @test xNext_a(agentD1, 10.0; i=2, t=49) == zNext_a(agentD1, 10.0; i=2, t=49)
    end

    @testset "value function" begin
        agentD1 = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0, incomes=ones(1, 50))
        @test v_x(agentD1, 10.0, i=1, t=50) == log(10.0)
        @test v_x(agentD1, 3.0, i=1, t=50) == log(3.0)
        @test agentD1.vFloor[1, 50] == log(3.0) # side effect of v_x, avoiding repeated computation.
    end

end


@testset verbose = true "Deterministic Model Solver PTD" begin

    @testset "child policy" begin
        agentD1 = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0, incomes=ones(1, 50))
        cps = descendant_operator(agentD1, PolicySegment(xStart=0.0, xEnd=Inf, b0=0.0, b1=1.0), yNext=1.0)
        @test typeof(cps) == PolicySegment
        @test cps.xStart > 0.0 && cps.xEnd == Inf
        @test descendant_operator(agentD1, i=1, t=50) == [cps,]
    end

    @testset "closed-form inverse Euler descendant" begin
        agent = DeterministicProblem(β=0.98, γ=2.0, r=0.1, xFloor=1.0, incomes=ones(1, 3))
        parent = PolicySegment(xStart=2.0, xEnd=8.0, b0=0.3, b1=0.4)
        child = descendant_operator(agent, parent, yNext=0.2)
        R = 1 + agent.r
        κ = (agent.β * R)^(-1 / agent.γ)
        q = κ * parent.b1 * R

        @test child.b0 ≈ κ * (parent.b0 + parent.b1 * 0.2) / (1 + q)
        @test child.b1 ≈ q / (1 + q)
        for x in range(child.xStart, child.xEnd; length=7)[2:end-1]
            c = c_x(child, x)
            z = R * (x - c) + 0.2
            @test z ≈ clamp(z, parent.xStart, parent.xEnd)
            @test mu(agent, c) ≈ agent.β * R * mu(agent, c_x(parent, z)) rtol = 2e-13
        end

        unbounded = descendant_operator(agent,
            PolicySegment(xStart=2.0, xEnd=Inf, b0=0.3, b1=0.4), yNext=0.2)
        @test isinf(unbounded.xEnd)
        @test unbounded.b0 ≈ child.b0
        @test unbounded.b1 ≈ child.b1
    end

    @testset "BigFloat inverse-Euler oracle" begin
        agent = DeterministicProblem(β=0.973, γ=2.7, r=-0.17, xFloor=0.8,
            incomes=reshape([0.4, -0.35, 1.2], 1, :))
        parent = PolicySegment(xStart=1.3, xEnd=9.7, b0=0.27, b1=0.63)
        child = descendant_operator(agent, parent, yNext=-0.35)

        setprecision(256) do
            β = BigFloat(agent.β)
            γ = BigFloat(agent.γ)
            R = BigFloat(1.0 + agent.r)
            y = BigFloat(-0.35)
            b0 = BigFloat(parent.b0)
            b1 = BigFloat(parent.b1)
            κ = (β * R)^(-inv(γ))
            q = κ * b1 * R
            b0Oracle = κ * (b0 + b1 * y) / (1 + q)
            b1Oracle = q / (1 + q)

            @test BigFloat(child.b0) ≈ b0Oracle rtol = big"6e-16"
            @test BigFloat(child.b1) ≈ b1Oracle rtol = big"6e-16"

            z = BigFloat("4.25")
            x = (z - y) / R + κ * (b0 + b1 * z)
            c = BigFloat(child.b0) + BigFloat(child.b1) * x
            zRecovered = R * (x - c) + y
            @test zRecovered ≈ z rtol = big"8e-16"
        end
    end

    @testset "randomized inverse-Euler identities" begin
        rng = MersenneTwister(20260804)
        for _ in 1:200
            β = rand(rng) * 0.4 + 0.7
            γ = rand(rng) * 3.5 + 0.25
            r = rand(rng) * 0.8 - 0.3
            floor = rand(rng) * 2.0 + 0.1
            yNext = rand(rng) * 4.0 - 1.0
            lower = max(floor, yNext) + rand(rng) * 2.0
            upper = lower + rand(rng) * 8.0 + 0.1
            b1 = rand(rng) * 0.8 + 0.1
            b0 = rand(rng) * 0.8 + 0.1
            agent = DeterministicProblem(β=β, γ=γ, r=r, xFloor=floor,
                incomes=fill(yNext, 1, 2))
            parent = PolicySegment(xStart=lower, xEnd=upper, b0=b0, b1=b1)
            child = descendant_operator(agent, parent, yNext=yNext)
            R = 1.0 + r

            for z in range(lower, upper; length=5)
                x = (z - yNext) / R + c_x(parent, z) * (β * R)^(-1 / γ)
                c = c_x(child, x)
                @test R * (x - c) + yNext ≈ z rtol = 2e-13 atol = 2e-13
                @test mu(agent, c) ≈ β * R * mu(agent, c_x(parent, z)) rtol = 3e-13
            end
        end
    end

    @testset "floor clipping and degenerate descendants" begin
        agent = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0, incomes=ones(1, 2))
        parent = PolicySegment(xStart=0.0, xEnd=Inf, b0=0.0, b1=1.0)
        child = descendant_operator(agent, parent, yNext=1.0)
        κ = (agent.β * (1 + agent.r))^(-1 / agent.γ)
        expected_start = (agent.xFloor - 1.0) / (1 + agent.r) + κ * agent.xFloor
        @test child.xStart ≈ expected_start

        # A left segment ending exactly at the floor has no nondegenerate Euler
        # branch. The same shared boundary is retained by its right-owned
        # neighbor, so discarding the zero-width child loses no candidate.
        unreachable = PolicySegment(xStart=0.0, xEnd=3.0, b0=0.0, b1=1.0)
        @test isnan(descendant_operator(agent, unreachable, yNext=1.0))
        right_owned = PolicySegment(xStart=3.0, xEnd=Inf, b0=0.0, b1=1.0)
        @test descendant_operator(agent, right_owned, yNext=1.0).xStart ≈ expected_start

        # A narrow but representable parent must not be removed by an arbitrary
        # width tolerance.
        narrow_end = 3.0 + 32 * eps(3.0)
        narrow = PolicySegment(xStart=3.0, xEnd=narrow_end, b0=0.0, b1=1.0)
        @test !isnan(descendant_operator(agent, narrow, yNext=1.0))
    end

    @testset "zero-floor right-limit comparison" begin
        agent = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=0.0,
            incomes=zeros(1, 4))
        @test deterministic_ptd!(agent) === nothing
        @test all(0.0 < c_x(agent, x, i=1, t=1) <= x
                  for x in (1e-6, 0.1, 1.0, 10.0))
    end

    @testset "scale-invariant kink comparison and exact overlap" begin
        normalized_kinks = Float64[]
        for scale in (1.0, 1e12)
            agent = DeterministicProblem(β=0.98, γ=2.0, r=0.1,
                xFloor=3.0 * scale, incomes=fill(scale, 1, 2))
            deterministic_ptd!(agent)
            push!(normalized_kinks, agent.policy[1, 1][1].xEnd / scale)
        end
        # Float64 value differences are about 1e-13 in the scaled problem, so
        # allow their root a small relative error while still ruling out the
        # former absolute-tolerance result (about 4.71 instead of 7.21).
        @test normalized_kinks[1] ≈ normalized_kinks[2] rtol = 5e-4

        left = PolicySegment(xStart=0.0, xEnd=1.0, b0=0.0, b1=1.0)
        right = PolicySegment(xStart=1.0, xEnd=2.0, b0=0.2, b1=0.5)
        candidate = PolicySegment(xStart=0.5, xEnd=prevfloat(1.0),
            b0=0.1, b1=0.4)
        @test last_overlapping_segment([left, right], candidate) == 1
    end

    @testset "ordered-envelope ties, duplicates, and diagnostics" begin
        # Half-open policy lookup assigns a shared boundary to the segment on
        # the right, which is the weakly higher-saving choice.
        elder = PolicySegment(xStart=0.0, xEnd=2.0, b0=0.0, b1=1.0)
        younger = PolicySegment(xStart=2.0, xEnd=Inf, b0=0.0, b1=0.5)
        ordered = [elder, younger]
        @test validate_policy_collection(ordered) === nothing
        @test c_x(ordered, 2.0) == 1.0

        duplicate = [
            PolicySegment(xStart=0.0, xEnd=1.0, b0=0.0, b1=0.5),
            PolicySegment(xStart=1.0, xEnd=Inf, b0=0.0, b1=0.5),
        ]
        coalesced = _coalesce_duplicate_segments(duplicate)
        @test coalesced == [PolicySegment(
            xStart=0.0, xEnd=Inf, b0=0.0, b1=0.5)]

        # Identical candidates are a flat value tangency.  Returning the left
        # boundary means the younger candidate owns their entire overlap.
        agent = DeterministicProblem(β=0.96, γ=1.5, r=0.1, xFloor=1.0,
            incomes=ones(1, 2))
        tied1 = PolicySegment(xStart=1.0, xEnd=4.0, b0=0.2, b1=0.4)
        tied2 = PolicySegment(xStart=1.0, xEnd=4.0, b0=0.2, b1=0.4)
        @test fusion_boundary(agent, tied1, tied2, 1.0, 4.0, i=1, t=1) == 1.0

        badGap = [elder,
            PolicySegment(xStart=2.1, xEnd=Inf, b0=0.0, b1=0.5)]
        badSaving = [
            PolicySegment(xStart=0.0, xEnd=2.0, b0=0.0, b1=0.5),
            PolicySegment(xStart=2.0, xEnd=Inf, b0=0.0, b1=1.0),
        ]
        @test_throws ErrorException validate_policy_collection(badGap)
        @test_throws ErrorException validate_policy_collection(badSaving)
        @test_throws ErrorException _assert_ordered_saving(
            badSaving[1], badSaving[2], 2.0, 4.0, i=1, t=1)

        # Candidate domains need not remain endpoint-sorted after the inverse
        # Euler map. A later child can overlap a non-adjacent elder after the
        # intervening child has disappeared, so the inherited saving order is
        # a total pairwise invariant rather than an adjacent-only check.
        nonadjacentElder = PolicySegment(
            xStart=0.0, xEnd=10.0, b0=0.0, b1=0.5)
        disjointMiddle = PolicySegment(
            xStart=20.0, xEnd=30.0, b0=0.0, b1=0.4)
        orderedYounger = PolicySegment(
            xStart=5.0, xEnd=15.0, b0=0.0, b1=0.2)
        reversedYounger = PolicySegment(
            xStart=5.0, xEnd=15.0, b0=0.0, b1=0.8)
        @test _validate_ordered_candidates(
            [nonadjacentElder, disjointMiddle, orderedYounger]; i=1, t=1) ===
              nothing
        @test_throws ErrorException _validate_ordered_candidates(
            [nonadjacentElder, disjointMiddle, reversedYounger]; i=1, t=1)

        # Two finite probes can miss a saving-order reversal far out on an
        # unbounded overlap.  The ray check must certify the affine slope.
        unboundedElder = PolicySegment(
            xStart=10.0, xEnd=Inf, b0=0.0, b1=0.5)
        unboundedYounger = PolicySegment(
            xStart=10.0, xEnd=Inf, b0=-6.0, b1=0.6)
        @test _assert_ordered_saving(unboundedElder,
            PolicySegment(xStart=10.0, xEnd=Inf, b0=-1.0, b1=0.4),
            10.0, Inf, i=1, t=1) === nothing
        @test_throws ErrorException _assert_ordered_saving(
            unboundedElder, unboundedYounger, 10.0, Inf, i=1, t=1)

        @test_throws ErrorException _assert_representable_value_order(
            0, 1, i=1, t=1, ps1=tied1, ps2=tied2)

        unboundedTie1 = PolicySegment(
            xStart=1.0, xEnd=Inf, b0=0.2, b1=0.4)
        unboundedTie2 = PolicySegment(
            xStart=1.0, xEnd=Inf, b0=0.2, b1=0.4)
        tiedElder, tiedYounger = pairwise_fusion_step_unbounded(
            agent, unboundedTie1, unboundedTie2, i=1, t=1)
        @test isnan(tiedElder)
        @test tiedYounger == unboundedTie2

        # Mathematically shared mapped endpoints can round to adjacent Float64
        # values.  Only a tightly bounded ULP gap is snapped; a real gap fails.
        adjacentRight = PolicySegment(xStart=nextfloat(2.0), xEnd=4.0,
            b0=0.0, b1=0.5)
        adjacentLeft = PolicySegment(xStart=0.0, xEnd=2.0,
            b0=0.0, b1=1.0)
        snappedLeft, snappedRight = pairwise_fusion_step_bounded(
            agent, adjacentLeft, adjacentRight, i=1, t=1)
        @test snappedLeft == adjacentLeft
        @test snappedRight.xStart == adjacentLeft.xEnd
        @test _within_domain_ulps(2.0, nextfloat(2.0))
        fourUlps = 2.0
        for _ in 1:4
            fourUlps = nextfloat(fourUlps)
        end
        fiveUlps = nextfloat(fourUlps)
        @test _within_domain_ulps(2.0, fourUlps)
        @test !_within_domain_ulps(2.0, fiveUlps)
        genuineGap = PolicySegment(xStart=fiveUlps, xEnd=4.0,
            b0=0.0, b1=0.5)
        @test_throws ErrorException pairwise_fusion_step_bounded(
            agent, adjacentLeft, genuineGap, i=1, t=1)
        @test_throws ArgumentError c_x(ordered, -0.1)
        @test_throws ArgumentError c_x(ordered, Inf)
    end

    @testset "two-period Bellman check" begin
        for γ in (0.5, 1.0, 2.0), r in (-0.2, 0.1, 0.6)
            agent = DeterministicProblem(β=0.96, γ=γ, r=r, xFloor=1.5,
                incomes=reshape([0.4, -0.2], 1, :))
            deterministic_ptd!(agent)
            R = 1.0 + r
            for x in (0.2, 1.5, 3.0, 8.0)
                c = c_x(agent, x, i=1, t=1)
                exact = u(agent, c) + 0.96 *
                    u(agent, max(R * (x - c) - 0.2, 1.5))
                assets = range(0.0, x; length=20_001)
                dense = maximum(u(agent, x - a) + 0.96 *
                    u(agent, max(R * a - 0.2, 1.5)) for a in assets if a < x)
                @test exact ≥ dense - 2e-7
                @test v_x(agent, x, i=1, t=1) ≈ exact rtol = 2e-13 atol = 2e-13
            end
        end
    end

    @testset "randomized multi-period direct-global checks" begin
        rng = MersenneTwister(314159)
        for _ in 1:12
            horizon = rand(rng, 3:7)
            β = 0.82 + 0.17 * rand(rng)
            γ = 0.4 + 2.8 * rand(rng)
            r = -0.2 + 0.55 * rand(rng)
            floor = 0.4 + 1.8 * rand(rng)
            incomes = reshape(-0.6 .+ 2.8 .* rand(rng, horizon), 1, :)
            agent = DeterministicProblem(β=β, γ=γ, r=r, xFloor=floor,
                incomes=incomes)
            deterministic_ptd!(agent)

            for t in 1:horizon
                @test validate_policy_collection(agent.policy[1, t];
                    context="randomized test path 1, period $t") === nothing
            end

            for t in 1:horizon-1, x in (0.25, floor, floor + 2.0, 10.0)
                c = c_x(agent, x, i=1, t=t)
                @test 0.0 < c <= x
                chosen = u(agent, c) + β * v_x(agent,
                    xNext_a(agent, x - c, i=1, t=t), i=1, t=t+1)
                assetGrid = range(0.0, x; length=4_001)
                gridBest = maximum(
                    u(agent, x - a) + β * v_x(agent,
                        xNext_a(agent, a, i=1, t=t), i=1, t=t+1)
                    for a in assetGrid if a < x)
                tolerance = 5e-7 * max(1.0, abs(chosen), abs(gridBest))
                @test chosen >= gridBest - tolerance
                @test v_x(agent, x, i=1, t=t) ≈ chosen rtol = 3e-13 atol = 3e-13
            end
        end
    end

    @testset "backward-solve invariants and threaded storage" begin
        incomes = [ones(6)'; fill(1.5, 1, 6)]
        agent = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0, incomes=incomes)
        @test agent.policy isa Matrix{Vector{PolicySegment}}
        deterministic_ptd!(agent, threshold=1)
        for i in axes(agent.incomes, 1), t in 1:agent.tMax
            policy = agent.policy[i, t]
            @test policy[1].xStart == 0.0
            @test isinf(policy[end].xEnd)
            @test all(policy[j].xEnd ≈ policy[j+1].xStart for j in 1:length(policy)-1)
            @test all(isfinite(ps.b0) && isfinite(ps.b1) && 0.0 <= ps.b1 <= 1.0
                      for ps in policy)
        end
        @test 0.0 < c_x(agent, 10.0, i=1, t=1) <= 10.0

        # The only explicit reuse control is a partial deterministic re-solve;
        # it must reproduce the cold solution and does not use an initial guess.
        partial = deepcopy(agent)
        partial.policy[1, 3] = [PolicySegment(
            xStart=0.0, xEnd=Inf, b0=0.0, b1=1.0)]
        deterministic_ptd!(partial, tStart=3, threshold=typemax(Int))
        @test partial.policy == agent.policy
        for i in axes(agent.incomes, 1), t in 1:agent.tMax,
            cash in (agent.xFloor, 5.0, 20.0)
            @test c_x(partial, cash, i=i, t=t) ==
                c_x(agent, cash, i=i, t=t)
        end
    end

end
