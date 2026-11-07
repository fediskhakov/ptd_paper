# Standalone regression for the generalized algorithm printed in the paper.
module MultiBranchExampleTests
using Test
include(joinpath(@__DIR__, "..", "scripts", "6_multibranch.jl"))
const MB = MultiBranchExample

@testset "Independent Brent optimizer: interior and exact boundary optima" begin
    setprecision(BigFloat, MB.REFERENCE_PRECISION) do
        lo, hi = BigFloat(0), BigFloat(1)
        peak = BigFloat("0.375")
        @test MB.maximize_concave(a -> -(a - peak)^2, lo, hi) ≈ peak atol=big"1e-15"
        @test MB.maximize_concave(a -> -a, lo, hi) == lo
        @test MB.maximize_concave(identity, lo, hi) == hi
    end
    constrained = MB.reference_second(0.5, MB.TRANSITIONS[2][1])
    @test constrained.saving == 0.0
    @test constrained.consumption == 0.5
end

@testset "Model B: affine descendants and printed cell fusion" begin
    solution = MB.solve(transitions=MB.MODEL_B_TRANSITIONS)
    @test solution.upper == [40.0, 59.25, 119.0]
    @test MB.check_collection(solution, 1)
    @test MB.check_collection(solution, 2)
    @test length.(solution.policy) == [7, 3, 1]
    @test all(p.source != "corner-2.5" for p in solution.policy[1])
    @test any(p.source == "corner-4.0" for p in solution.policy[1])
    # Check descendant feasibility and Euler equations independently of the
    # coefficient construction, on every branch/parent intersection.
    for t in 1:2, candidate in solution.candidates[t]
        startswith(candidate.source, "child-") || continue
        _, branch_index, parent_index = split(candidate.source, '-')
        branch = solution.transitions[t][parse(Int, branch_index)]
        parent = solution.policy[t + 1][parse(Int, parent_index)]
        x = (candidate.lower + candidate.upper) / 2
        c = MB.consumption(candidate, x)
        a = x - c
        z = MB.transition(t, a; transitions=solution.transitions)
        next_c = MB.consumption(parent, z)
        @test branch.lower <= a <= branch.upper
        @test parent.lower <= z <= parent.upper
        @test inv(c^2) ≈ MB.BETA * branch.slope / next_c^2 rtol=5e-13
    end
    rows = MB.verify(solution; grid_points=201)
    @test maximum(abs(r.ptd_value - r.reference_value) for r in rows) < 2e-12
    @test maximum(abs(r.ptd_consumption - r.reference_consumption) for r in rows) < 1e-7
    @test MB.consumption(MB.policy_piece(solution, 1, 10.0), 10.0) ≈ 4.123467688002092 atol=1e-12
    @test MB.value(solution, 1, 10.0) ≈ 2.2839724017 atol=5e-11
    # Right-hand ownership implements save-more at each discontinuous join.
    for t in 1:2, i in 2:length(solution.policy[t])
        left, right = solution.policy[t][i - 1], solution.policy[t][i]
        x = right.lower
        if MB.consumption(left, x) > MB.consumption(right, x) + 1e-10
            @test MB.policy_piece(solution, t, x) === right
            @test MB.candidate_value(solution, t, left, x) ≈
                  MB.candidate_value(solution, t, right, x) atol=2e-12
        end
    end
    @test_throws DomainError MB.policy_piece(solution, 1, 41.0)
    extended = MB.solve(xmax=80.0, transitions=MB.MODEL_B_TRANSITIONS)
    for x in (0.01, 1.0, 4.85, 5.0, 6.5, 10.0, 40.0)
        @test MB.value(solution, 1, x) ≈ MB.value(extended, 1, x) atol=2e-12
    end
end

@testset "Appendix calibration: continuous transitions and visible post-jump piece" begin
    solution = MB.solve()
    original = MB.solve(transitions=MB.MODEL_B_TRANSITIONS)
    @test solution.transitions === MB.TRANSITIONS
    @test original.transitions === MB.MODEL_B_TRANSITIONS
    @test solution.upper == [40.0, 60.75, 122.0]
    @test length.(solution.policy) == [7, 3, 1]
    @test MB.check_collection(solution, 1)
    @test MB.check_collection(solution, 2)
    for branches in solution.transitions, i in 2:length(branches)
        left, right = branches[i - 1], branches[i]
        @test left.upper == right.lower
        @test left.intercept + left.slope * left.upper ≈
            right.intercept + right.slope * right.lower atol=1e-14
    end
    @test all(p.source != "corner-1.5" for p in solution.policy[1])
    @test any(p.source == "corner-4.0" for p in solution.policy[1])
    for t in 1:2, candidate in solution.candidates[t]
        startswith(candidate.source, "child-") || continue
        _, branch_index, parent_index = split(candidate.source, '-')
        branch = solution.transitions[t][parse(Int, branch_index)]
        parent = solution.policy[t + 1][parse(Int, parent_index)]
        x = (candidate.lower + candidate.upper) / 2
        c = MB.consumption(candidate, x)
        a = x - c
        z = MB.transition(t, a; transitions=solution.transitions)
        next_c = MB.consumption(parent, z)
        @test branch.lower <= a <= branch.upper
        @test parent.lower <= z <= parent.upper
        @test inv(c^2) ≈ MB.BETA * branch.slope / next_c^2 rtol=5e-13
    end
    rows = MB.verify(solution; grid_points=201)
    @test maximum(abs(r.ptd_value - r.reference_value) for r in rows) < 2e-12
    @test maximum(abs(r.ptd_consumption - r.reference_consumption) for r in rows) < 1e-7
    # Moving the first transition corner makes the piece after the first
    # consumption jump wider, while retaining all seven date-1 policy pieces.
    new_piece = only(filter(p -> p.source == "child-2-1", solution.policy[1]))
    old_piece = only(filter(p -> p.source == "child-2-1", original.policy[1]))
    @test new_piece.upper - new_piece.lower > 7 * (old_piece.upper - old_piece.lower)
    @test MB.consumption(solution.policy[1][2], new_piece.lower) >
        MB.consumption(new_piece, new_piece.lower)
    for t in 1:2, i in 2:length(solution.policy[t])
        left, right = solution.policy[t][i - 1], solution.policy[t][i]
        x = right.lower
        if MB.consumption(left, x) > MB.consumption(right, x) + 1e-10
            @test MB.policy_piece(solution, t, x) === right
            @test MB.candidate_value(solution, t, left, x) ≈
                MB.candidate_value(solution, t, right, x) atol=2e-12
        end
    end
    @test_throws DomainError MB.policy_piece(solution, 1, 41.0)
    extended = MB.solve(xmax=80.0)
    for x in (0.01, 1.0, 3.30, 4.25, 5.0, 7.0, 10.0, 40.0)
        @test MB.value(solution, 1, x) ≈ MB.value(extended, 1, x) atol=2e-12
    end
    mktempdir() do root
        path = MB.save_verification(solution, rows; root)
        @test readdir(joinpath(root, "data")) == ["multibranch-verification.jld2"]
        saved = MB.JLD2.load(path, "artifact")
        @test saved.schema == "ptd-multibranch-verification-v1"
        @test saved.rows == rows
        @test saved.summary == MB.verification_summary(solution, rows)
        @test saved.beta == MB.BETA
        @test saved.upper == solution.upper
        @test saved.transitions[1][1].upper == 1.5
        @test length.(saved.policy) == length.(solution.policy)
    end
    @test_throws ArgumentError MB.save_verification(solution, rows;
        root=joinpath(MB.PROJECT, "data"))
end

@testset "Appendix exhibits go to the requested folders" begin
    solution = MB.solve()
    rows = MB.verify(solution; grid_points=21)
    mktempdir() do root
        output = MB.render(solution, rows; root,
            table_directory=joinpath(root, "Tables"),
            graph_directory=joinpath(root, "Graphs"))
        @test output.table ==
            joinpath(root, "Tables", "multibranch_verification.tex")
        @test output.graph == joinpath(root, "Graphs", "multibranch_model_b")
        @test output.artifact ==
            joinpath(root, "data", "multibranch-verification.jld2")
        @test readdir(root) == ["Graphs", "Tables", "data"]
        @test readdir(joinpath(root, "Graphs")) == ["multibranch_model_b.png"]
        @test occursin(raw"\begin{tabular}", read(output.table, String))
    end
end
end
