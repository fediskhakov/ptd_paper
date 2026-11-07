using Test

@testset "paper-facing script layout" begin
    script_directory = joinpath(pkgdir(Ptd), "scripts")
    expected = [
        "1_ptd_visualization.jl",
        "2_deterministic.jl",
        "3_iid.jl",
        "4_ar1.jl",
        "5_hsz.jl",
        "6_multibranch.jl",
    ]
    @test all(name -> isfile(joinpath(script_directory, name)), expected)

    sources = Dict(name => read(joinpath(script_directory, name), String)
        for name in expected)
    @test occursin("policy_value_T.png", sources[expected[1]])
    @test occursin("d_policies_ptd_egm.png", sources[expected[1]])
    @test occursin("deterministic_performance.tex", sources[expected[2]])
    @test occursin("iid_contest_summary.tex", sources[expected[3]])
    @test occursin("ar1_contest_summary.tex", sources[expected[4]])
    @test occursin("HSZ_TIMING_FILENAME", sources[expected[5]])
    @test occursin("hsz_utility_comparison.tex", sources[expected[5]])
    @test occursin("hsz_grid_refinement.tex", sources[expected[5]])
    @test occursin("multibranch_verification.tex", sources[expected[6]])
    @test occursin("refinement", lowercase(sources[expected[5]]))
end
