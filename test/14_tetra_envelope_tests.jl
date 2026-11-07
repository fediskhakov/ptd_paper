# Tests for src/g2egm/tetra_envelope.jl and source-order file 14.
@testset "three-dimensional tetrahedral envelope" begin
    Tetra = Ptd.G2EGM
    tetrahedra = Tetra.tensor_cell_tetrahedra(1, 1, 1)
    @test length(tetrahedra) == 6
    @test length(unique(reduce(vcat, collect.(tetrahedra)))) == 8

    vertices = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0), (1.0, 1.0, 1.0))
    weights = Tetra.barycentric_weights_3d(vertices, (0.5, 0.4, 0.2))
    @test !isnothing(weights)
    @test sum(weights) ≈ 1.0
    reconstructed = ntuple(d ->
        sum(weights[q] * vertices[q][d] for q in 1:4), 3)
    @test collect(reconstructed) ≈ [0.5, 0.4, 0.2]

    common = [0.0, 0.5, 1.0]
    shape = (2, 2, 2)
    mapped_x = Array{Float64,3}(undef, shape)
    mapped_e = similar(mapped_x)
    mapped_m = similar(mapped_x)
    consumption = similar(mapped_x)
    for index in CartesianIndices(mapped_x)
        x, e, m = Float64.(Tuple(index) .- 1)
        mapped_x[index], mapped_e[index], mapped_m[index] = x, e, m
        consumption[index] = 1.0 + 0.2x + 0.1e + 0.05m
    end
    result = Tetra.project_tetra_envelope(mapped_x, mapped_e, mapped_m,
        consumption, common, common, common;
        feasible=(x, e, m, c) -> c > 0.0,
        value_at=(x, e, m, c) -> -(c -
            (1.0 + 0.2x + 0.1e + 0.05m))^2,
        options=Tetra.TetraEnvelopeOptions(index_padding=0,
            minimum_weight=0.0))
    @test all(result.coverage)
    @test result.consumption[2, 2, 2] ≈ 1.175
    @test result.audit.tetrahedra_processed == 6
end
