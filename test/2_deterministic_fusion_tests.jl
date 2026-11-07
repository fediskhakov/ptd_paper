# Paper-terminology contract for src/ptd/deterministic_solver.jl.

@testset "deterministic fusion API uses paper terminology" begin
    @test isdefined(Ptd, :descendant_operator)
    @test isdefined(Ptd, :pairwise_fusion_step)
    @test isdefined(Ptd, :fusion_operator)
    @test isdefined(Ptd, :deterministic_ptd!)

    @test !isdefined(Ptd, :child_ps)
    @test !isdefined(Ptd, :fusion)
    @test !isdefined(Ptd, :glue)
end

@testset "fusion preserves every domain cell with nested Euler children" begin
    # This income path produces a finite child ending inside an elder domain.
    # Exhaustive feasible-candidate comparisons test the complete envelope,
    # including cells outside the younger child's domain.
    incomes = reshape([-0.45091766838375275, 1.119810747024876,
        0.12707630573378517, -2.0535131249215657, -0.7798970940951889,
        -0.49280566794992836, -0.5465746939266005, -1.0299304914575556], 1, :)
    problem = Ptd.DeterministicProblem(β=0.8145320042614919,
        γ=1.8325949791818565, r=0.3402408116207496,
        xFloor=1.0, incomes=incomes)
    Ptd.deterministic_ptd!(problem)
    children = Ptd.descendant_operator(problem; i=1, t=2)
    @test any(children[j].xEnd < children[j-1].xEnd for j in 2:length(children))
    corner = Ptd.PolicySegment(xStart=0.0, xEnd=Inf, b0=0.0, b1=1.0)
    candidates = vcat([corner], children)
    boundaries = [endpoint for child in children
        for endpoint in (child.xStart, child.xEnd) if isfinite(endpoint)]
    probes = vcat(collect(range(0.01, 12.0; length=2001)), boundaries)
    for cash in probes
        candidate_values = [Ptd.u(problem, Ptd.c_x(candidate, cash)) +
            problem.β * Ptd.v_x(problem, Ptd.xNext_a(problem,
                cash - Ptd.c_x(candidate, cash); i=1, t=1); i=1, t=2)
            for candidate in candidates if candidate.xStart <= cash <= candidate.xEnd]
        @test Ptd.v_x(problem, cash; i=1, t=1) >= maximum(candidate_values) - 2e-11
    end
end

@testset "unbounded fusion fails closed and records explicit diagnostic fallbacks" begin
    # Both actions lead to the same floor state. Saving one is strictly worse
    # at every finite wealth, so geometric expansion cannot find a crossing.
    problem = Ptd.DeterministicProblem(β=0.95, γ=1.0, r=0.0,
        xFloor=1.0, incomes=zeros(1, 2))
    corner = Ptd.PolicySegment(xStart=0.0, xEnd=Inf, b0=0.0, b1=1.0)
    fixed_saving = Ptd.PolicySegment(xStart=2.0, xEnd=Inf, b0=-1.0, b1=1.0)
    strict = Ptd.FusionOptions(search_bound=8.0, max_expansions=1)
    error = try
        Ptd.pairwise_fusion_step_unbounded(problem, corner, fixed_saving;
            i=1, t=1, options=strict)
        nothing
    catch caught
        caught
    end
    @test error isa Ptd.FusionTailError
    @test error.path == 1 && error.period == 1 && error.search_bound == 8.0
    @test error.parameters.beta == problem.β
    @test error.parameters.incomes == [0.0, 0.0]
    @test occursin("Experiment stopped", sprint(showerror, error))
    @test Ptd.fusion_tail_diagnostics(strict).unresolved_count == 1
    @test !Ptd.FusionOptions().allow_uncertified_tail
    options = Ptd.FusionOptions(search_bound=8.0, max_expansions=1,
        allow_uncertified_tail=true)
    for _ in 1:2
        elder, younger = @test_logs (:warn, r"finite search bound") begin
            Ptd.pairwise_fusion_step_unbounded(problem, corner, fixed_saving;
                i=1, t=1, options)
        end
        @test elder == corner
        @test isnan(younger)
    end
    @test Ptd.fusion_tail_diagnostics(options).unresolved_count == 2
    @test_throws ArgumentError Ptd.FusionOptions(search_bound=Inf)
    @test_throws ArgumentError Ptd.FusionOptions(max_expansions=0)
    @test_throws ArgumentError Ptd.pairwise_fusion_step_unbounded(
        problem, corner, fixed_saving; i=1, t=1,
        options=Ptd.FusionOptions(search_bound=1.0))

    # A small search budget must still jump to the declared cap and find a
    # genuine finite crossover rather than prematurely selecting an endpoint.
    problem = Ptd.DeterministicProblem(β=0.95, γ=1.0, r=0.0,
        xFloor=3.0, incomes=zeros(1, 2))
    child = Ptd.descendant_operator(problem, corner; yNext=0.0)
    left, right = Ptd.pairwise_fusion_step_unbounded(problem, corner, child;
        i=1, t=1, options=Ptd.FusionOptions(search_bound=100.0, max_expansions=1))
    @test child.xStart < left.xEnd < 100.0
    @test left.xEnd == right.xStart
    @test abs(Ptd.candidate_value_difference(left.xEnd, problem, corner, child;
        i=1, t=1)) < 1e-10
end
