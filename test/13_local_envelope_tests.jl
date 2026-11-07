# Tests for src/g2egm/local_envelope.jl and source-order file 13.
using Test

include(joinpath(@__DIR__, "..", "src", "g2egm", "local_envelope.jl"))

const NO_FILL = EnvelopeOptions(
    index_padding=0,
    minimum_weight=0.0,
    fill_holes=false,
)

@testset "pure-Julia G2EGM local envelope" begin
    @testset "tensor triangulation and barycentric primitives" begin
        lower, upper = tensor_cell_triangles(2, 4)
        @test lower == (CartesianIndex(2, 4), CartesianIndex(2, 5),
                        CartesianIndex(3, 4))
        @test upper == (CartesianIndex(3, 5), CartesianIndex(3, 4),
                        CartesianIndex(2, 5))

        vertices = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0))
        @test collect(barycentric_weights(vertices, 0.25, 0.25)) ≈
            [0.5, 0.25, 0.25]
        @test barycentric_weights(((0.0, 0.0), (1.0, 0.0),
                                   (2.0, 0.0)), 0.5, 0.0) === nothing

        common = collect(0.0:1.0:4.0)
        narrow = ((1.2, 1.2), (1.2, 1.8), (1.8, 1.2))
        no_padding = padded_common_bbox(
            common, common, narrow; index_padding=0)
        with_padding = padded_common_bbox(
            common, common, narrow; index_padding=1)
        @test isempty(no_padding[1]) && isempty(no_padding[2])
        @test no_padding[1] == 3:2
        @test with_padding == (2:3, 2:3)
    end

    @testset "affine reconstruction recomputes value" begin
        mapped_x = [0.0 0.0; 1.0 1.0]
        mapped_y = [0.0 1.0; 0.0 1.0]
        node_consumption = 2.0 .+ mapped_x .+ 2.0 .* mapped_y
        common_x = collect(0.0:0.25:1.0)
        common_y = collect(0.0:0.25:1.0)
        calls = Ref(0)
        value_at = function (x, y, consumption)
            calls[] += 1
            return 7.0 + 3.0x - y + 5.0consumption
        end
        result = project_local_envelope(
            mapped_x, mapped_y, node_consumption, common_x, common_y;
            feasible=(x, y, consumption) -> consumption > 0.0,
            value_at,
            options=NO_FILL,
        )

        expected_consumption = [2.0 + x + 2.0y for x in common_x, y in common_y]
        expected_value = [7.0 + 3.0x - y + 5.0expected_consumption[i, j]
                          for (i, x) in enumerate(common_x),
                              (j, y) in enumerate(common_y)]
        @test all(result.direct_coverage)
        @test result.consumption ≈ expected_consumption
        @test result.value ≈ expected_value
        @test calls[] == result.audit.finite_candidates
        @test result.audit.tensor_cells == 1
        @test result.audit.triangles_total == 2
        @test result.audit.triangles_processed == 2
        @test result.audit.triangles_degenerate == 0
    end

    @testset "degenerate and structural-regime rejection" begin
        mapped_x = [0.0 0.0; 1.0 1.0]
        collapsed_y = zeros(2, 2)
        consumption = ones(2, 2)
        degenerate = project_local_envelope(
            mapped_x, collapsed_y, consumption, [0.5], [0.0];
            feasible=(x, y, c) -> true,
            value_at=(x, y, c) -> 1.0,
            options=NO_FILL,
        )
        @test degenerate.audit.triangles_total == 2
        @test degenerate.audit.triangles_degenerate == 2
        @test !any(degenerate.coverage)
        @test isinf(degenerate.audit.minimum_barycentric_weight_among_winners)

        mapped_y = [0.0 1.0; 0.0 1.0]
        regimes = [1 1; 1 2]
        common_x = [0.25, 0.75]
        common_y = [0.25, 0.75]
        segmented = project_local_envelope(
            mapped_x, mapped_y, consumption, common_x, common_y;
            feasible=(x, y, c) -> true,
            value_at=(x, y, c) -> 1.0,
            regimes,
            options=NO_FILL,
        )
        @test segmented.audit.triangles_processed == 1
        @test segmented.audit.triangles_regime_rejected == 1
        @test segmented.direct_coverage[1, 1]
        @test !segmented.direct_coverage[2, 2]

        unsegmented = project_local_envelope(
            mapped_x, mapped_y, consumption, common_x, common_y;
            feasible=(x, y, c) -> true,
            value_at=(x, y, c) -> 1.0,
            options=NO_FILL,
        )
        @test all(unsegmented.direct_coverage)
    end

    @testset "overlaps use recomputed values and save-more ties" begin
        # Two adjacent tensor cells both map to the same unit square. Their
        # candidates at (0.5, 0.5) are c=0.5 and c=0.65.
        mapped_x = [0.0 0.0; 1.0 1.0; 0.0 0.0]
        mapped_y = [0.0 1.0; 0.0 1.0; 0.0 1.0]
        consumption = [0.2 0.2; 0.8 0.8; 0.5 0.5]
        constant_value = project_local_envelope(
            mapped_x, mapped_y, consumption, [0.5], [0.5];
            feasible=(x, y, c) -> true,
            value_at=(x, y, c) -> 11.0,
            options=NO_FILL,
        )
        @test constant_value.candidate_count[1, 1] == 4
        @test constant_value.consumption[1, 1] ≈ 0.5
        @test constant_value.value[1, 1] == 11.0
        @test constant_value.audit.save_more_ties >= 1

        feasible_only = project_local_envelope(
            mapped_x, mapped_y, consumption, [0.5], [0.5];
            feasible=(x, y, c) -> c <= 0.55,
            value_at=(x, y, c) -> c,
            options=NO_FILL,
        )
        @test feasible_only.consumption[1, 1] ≈ 0.5
        @test feasible_only.audit.infeasible_candidates == 2
    end

    @testset "limited-extrapolation winner provenance" begin
        mapped_x = [0.0 0.0; 1.0 1.0]
        mapped_y = [0.0 1.0; 0.0 1.0]
        consumption = [0.4 0.4; 0.6 0.6]
        extrapolated = project_local_envelope(
            mapped_x, mapped_y, consumption, [-0.1], [0.5];
            feasible=(x, y, c) -> true,
            value_at=(x, y, c) -> 2.0 - c,
            options=EnvelopeOptions(
                index_padding=1,
                minimum_weight=-0.25,
                fill_holes=false,
            ),
        )
        @test extrapolated.direct_coverage[1, 1]
        @test extrapolated.audit.negative_barycentric_candidates >= 1
        @test extrapolated.audit.negative_barycentric_winner_updates >= 1
        @test extrapolated.audit.winners_with_negative_barycentric_weight == 1
        @test extrapolated.audit.minimum_barycentric_weight_among_winners ≈ -0.1
        @test extrapolated.winner_minimum_barycentric_weight[1, 1] ≈ -0.1
        @test extrapolated.winner_triangle[1, 1] > 0
    end

    @testset "audited radius-two nearby-choice hole envelope" begin
        common_x = [0.0, 1.0, 2.0]
        common_y = [0.0]
        consumption = reshape([0.2, NaN, 0.8], 3, 1)
        value = reshape([20.0, -Inf, 80.0], 3, 1)
        direct = BitMatrix(reshape([true, false, true], 3, 1))
        audit = EnvelopeAudit()
        repaired = fill_local_holes!(
            consumption, value, direct, common_x, common_y;
            feasible=(x, y, c) -> 0.0 < c <= x,
            value_at=(x, y, c) -> 100.0 + x + c,
            options=EnvelopeOptions(hole_radius=2),
            fill_target=trues(3, 1),
            audit,
        )
        @test all(repaired.coverage)
        @test consumption[2, 1] == 0.8
        @test value[2, 1] == 101.8
        @test length(repaired.records) == 1
        record = only(repaired.records)
        @test record.target == CartesianIndex(2, 1)
        @test record.source == CartesianIndex(3, 1)
        @test record.squared_index_distance == 1
        @test record.neighbor_candidates_checked == 2
        @test record.recomputed_value == 101.8
        @test audit.direct_holes_before_fill == 1
        @test audit.hole_neighbor_candidates_checked == 2
        @test audit.holes_filled == 1
        @test audit.holes_remaining == 0

        # Equal recomputed values deterministically select lower consumption.
        tied_consumption = reshape([0.2, NaN, 0.8], 3, 1)
        tied_value = reshape([1.0, -Inf, 1.0], 3, 1)
        tied = fill_local_holes!(
            tied_consumption, tied_value, direct, common_x, common_y;
            feasible=(x, y, c) -> true,
            value_at=(x, y, c) -> 5.0,
            options=EnvelopeOptions(hole_radius=2),
            fill_target=trues(3, 1),
        )
        @test tied.coverage[2, 1]
        @test tied_consumption[2, 1] == 0.2
        @test only(tied.records).source == CartesianIndex(1, 1)

        # Reference G2EGM evaluates every covered choice within the radius,
        # rather than stopping at the closest feasible source.
        wide_x = collect(0.0:4.0)
        wide_consumption = reshape([0.1, NaN, NaN, NaN, 0.9], 5, 1)
        wide_value = reshape([1.0, -Inf, -Inf, -Inf, 1.0], 5, 1)
        wide_direct = BitMatrix(reshape(
            [true, false, false, false, true], 5, 1))
        wide_target = falses(5, 1)
        wide_target[2, 1] = true
        wide = fill_local_holes!(
            wide_consumption, wide_value, wide_direct, wide_x, common_y;
            feasible=(x, y, c) -> true,
            value_at=(x, y, c) -> c,
            options=EnvelopeOptions(hole_radius=3),
            fill_target=wide_target,
        )
        @test wide_consumption[2, 1] == 0.9
        @test only(wide.records).source == CartesianIndex(5, 1)
        @test only(wide.records).squared_index_distance == 9
        @test only(wide.records).neighbor_candidates_checked == 2
    end

    @testset "input contract fails closed" begin
        @test_throws ArgumentError EnvelopeOptions(index_padding=-1) |>
            _validate_options
        @test_throws ArgumentError padded_common_bbox(
            [0.0, 0.0], [0.0, 1.0],
            ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
        @test_throws DimensionMismatch project_local_envelope(
            zeros(2, 2), zeros(2, 3), ones(2, 2), [0.0], [0.0];
            feasible=(x, y, c) -> true,
            value_at=(x, y, c) -> 0.0,
        )
    end
end
