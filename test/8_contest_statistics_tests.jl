# Tests for src/simulation/common/contest_statistics.jl and source-order file 8.
using Test
using Statistics
using Ptd

@testset "Paired within-contest welfare statistics" begin
    methods = [:ptd_sceq, :ptd_osa, :dcegm]
    utilities = [
        1.0 2.0 0.0
        1.2 2.2 0.2
        0.8 1.8 -0.2
        1.1 2.1 0.1
    ]
    summary = summarize_contest(1, :baseline, methods, utilities)
    @test summary.n_paths == 4
    @test summary.confidence_level == 0.95
    @test summary.expected_utility ≈ [1.025, 2.025, 0.025]
    @test summary.point_winners == [:ptd_osa]
    @test summary.point_winner == :ptd_osa
    @test summary.resolved_winner == :ptd_osa
    @test summary.classification == :resolved
    @test length(summary.comparisons) == 3
    @test all(comparison -> comparison.n_paths == 4, summary.comparisons)
    @test all(comparison -> comparison.confidence_level == 0.95,
        summary.comparisons)

    # Paired uncertainty is calculated from common-path differences.
    comparison = paired_comparison(
        :first, :second, [2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0])
    differences = [1.0, 2.0, 3.0, 4.0]
    @test comparison.mean_difference == mean(differences)
    @test comparison.standard_error ≈
        std(differences; corrected=true) / sqrt(length(differences))
    @test comparison.ci_lower < comparison.mean_difference < comparison.ci_upper

    # Reordering methods cannot alter either the point or interval winner.
    permutation = [3, 1, 2]
    reordered = summarize_contest(
        2, :baseline, methods[permutation], utilities[:, permutation])
    @test reordered.point_winner == :ptd_osa
    @test reordered.resolved_winner == :ptd_osa

    noisy = copy(utilities)
    noisy[:, 1] = [0.0, 4.0, 0.0, 4.0]
    noisy[:, 2] = [3.0, 1.1, 3.0, 1.1]
    unresolved = summarize_contest(3, :baseline, methods, noisy)
    @test unresolved.point_winner == :ptd_osa
    @test isnothing(unresolved.resolved_winner)
    @test unresolved.classification == :mc_unresolved

    # Exact sample-mean ties remain ties. They are not awarded to the first
    # column and are kept distinct from interval uncertainty around a unique
    # point leader.
    tied_utilities = [
        2.0 2.0 0.0
        3.0 3.0 0.0
        4.0 4.0 0.0
        5.0 5.0 0.0
    ]
    tied = summarize_contest(4, :baseline, methods, tied_utilities)
    @test tied.point_winners == [:ptd_sceq, :ptd_osa]
    @test isnothing(tied.point_winner)
    @test isnothing(tied.resolved_winner)
    @test tied.classification == :point_tie

    # There is no 1e-12 medal rule: a constant one-ULP paired advantage has
    # zero Monte Carlo uncertainty and is therefore resolved.
    one_ulp = [1.0 + eps(Float64) 1.0; 1.0 + eps(Float64) 1.0]
    tiny = summarize_contest(5, :baseline, [:a, :b], one_ulp)
    @test tiny.resolved_winner == :a
    @test tiny.comparisons[1].mean_difference == eps(Float64)

    aggregate = aggregate_outcomes([summary, unresolved, tied])
    @test aggregate.panel == :baseline
    @test aggregate.n_contests == 3
    @test aggregate.n_paths == 4
    @test aggregate.unresolved == 2
    @test aggregate.mc_unresolved == 1
    @test aggregate.point_ties == 1
    @test aggregate.classification_counts ==
        (resolved=1, mc_unresolved=1, point_tie=1)
    @test aggregate.unresolved_share == 2 / 3
    @test aggregate.point_tie_share == 1 / 3
    @test sum(values(aggregate.point_counts)) == 2
    @test sum(values(aggregate.point_shares)) == 2 / 3
    @test aggregate.resolved_win_shares[:ptd_osa] == 1 / 3
    @test aggregate.resolved_loss_shares[:ptd_sceq] == 1 / 3
    @test aggregate.tie_counts[(:ptd_sceq, :ptd_osa)] == 1
    # Aggregate output has neither raw expected-utility levels nor raw utility
    # gaps: gamma varies across contests, so neither quantity is poolable.
    @test :expected_utility ∉ propertynames(aggregate)
    @test :paired_gap_samples ∉ propertynames(aggregate)

    high_panel = summarize_contest(6, :high_variance, methods, utilities)
    other_level = summarize_contest(7, :baseline, methods, utilities; level=0.90)
    other_paths = summarize_contest(8, :baseline, methods, vcat(utilities, utilities))
    @test_throws ArgumentError aggregate_outcomes([summary, high_panel])
    @test_throws ArgumentError aggregate_outcomes([summary, other_level])
    @test_throws ArgumentError aggregate_outcomes([summary, other_paths])
    @test_throws ArgumentError aggregate_outcomes([summary, summary])

    @test_throws DimensionMismatch summarize_contest(
        1, :baseline, methods, utilities[:, 1:2])
    @test_throws ArgumentError summarize_contest(
        1, :unknown, methods, utilities)
    @test_throws ArgumentError summarize_contest(
        1, :baseline, [:same, :same], utilities[:, 1:2])
    @test_throws ArgumentError summarize_contest(
        1, :baseline, [:only], utilities[:, 1:1])
    @test_throws ArgumentError paired_comparison(
        :same, :same, utilities[:, 1], utilities[:, 2])
    @test_throws ArgumentError paired_comparison(
        :a, :b, utilities[:, 1], utilities[:, 2]; level=1.0)
    @test_throws ArgumentError paired_comparison(
        :a, :b, [floatmax(Float64), floatmax(Float64)],
        [-floatmax(Float64), -floatmax(Float64)])
end
