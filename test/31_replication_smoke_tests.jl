# Exercise the paper entry points with isolated output directories. Production
# Monte Carlo sizes are intentionally separate from this integration check.
module VisualizationScriptSmoke
include(joinpath(@__DIR__, "..", "scripts", "1_ptd_visualization.jl"))
end

module DeterministicScriptSmoke
include(joinpath(@__DIR__, "..", "scripts", "2_deterministic.jl"))
end

module IIDScriptSmoke
include(joinpath(@__DIR__, "..", "scripts", "3_iid.jl"))
end

module AR1ScriptSmoke
include(joinpath(@__DIR__, "..", "scripts", "4_ar1.jl"))
end

module HSZScriptSmoke
include(joinpath(@__DIR__, "..", "scripts", "5_hsz.jl"))
end

@testset "paper script execution" begin
    mktempdir() do directory
        figures = VisualizationScriptSmoke.ptd_visualization_outputs(
            graph_directory=joinpath(directory, "visualization"))
        @test all(isfile, figures)
        comparison =
            VisualizationScriptSmoke.deterministic_policy_comparison(
                graph_directory=joinpath(directory, "visualization"))
        @test comparison ==
            joinpath(directory, "visualization", "d_policies_ptd_egm.png")
        @test isfile(comparison)
        # Figure 5 draws DC-EGM on the package's copy of the Table 1 grids,
        # which must stay equal to the grids of scripts/2_deterministic.jl.
        @test collect(values(
            Ptd.FixedCalibrationTiming.deterministic_timing_grids())) ==
            DeterministicScriptSmoke.deterministic_asset_grids()

        outputs = DeterministicScriptSmoke.deterministic_outputs(
            n_draws=3,
            table_directory=joinpath(directory, "tables"))
        @test outputs isa DeterministicScriptSmoke.DeterministicOutputs
        @test isfile(outputs.artifact_path)
        @test length(outputs.benchmark.max_loss_bp) == 5
        @test outputs.benchmark.artifact.tail_safeguard.unresolved_count == 0
        @test length(outputs.benchmark.specs) == 3
        @test all(isfinite, outputs.benchmark.max_loss_bp)
        @test all(>=(0), outputs.benchmark.batch_time_seconds)
        @test isfile(outputs.table_path)
        @test occursin("DC-EGM", read(outputs.table_path, String))

        # Exercise the public CLI's render-only paths using archived data;
        # all new files remain outside the tracked Tables, Graphs and data
        # directories.
        data_directory = joinpath(pkgdir(Ptd), "data")
        iid = IIDScriptSmoke.main(["--render-only",
            joinpath(data_directory, "iid-contest.jld2"),
            joinpath(directory, "iid.tex")])
        @test isfile(iid.table_path)
        @test isfile(iid.ce_table_path)
        @test occursin("Median", read(iid.ce_table_path, String))
        @test occursin("DC-EGM", read(iid.table_path, String))

        ar1 = AR1ScriptSmoke.main(["--render-only",
            joinpath(data_directory, "ar1-contest-welfare.jld2"),
            joinpath(data_directory, "ar1-contest.jld2"),
            joinpath(directory, "ar1"), joinpath(directory, "ar1.tex")])
        @test isfile(ar1.table_path)
        @test isfile(ar1.ce_table_path)
        @test occursin("P95", read(ar1.ce_table_path, String))
        @test isfile(ar1.figures.surface_path)
        @test isfile(ar1.figures.paths_path)
        @test ar1.figures.scope === :updated_heterogeneous_paths
        configurations = ar1.figures.selected_configurations
        @test getproperty.(configurations, :contest_id) == [75, 91, 8, 19, 98]
        @test length(unique(getproperty.(configurations, :beta))) == 5
        @test length(unique(getproperty.(configurations, :gamma))) == 5
        @test all(row -> row.gamma != 1.0, configurations)

        hsz = HSZScriptSmoke.main(["--render-only", "--example-artifact",
            joinpath(data_directory, "hsz-example-medical-1p5.jld2"),
            "--utility-artifact", joinpath(data_directory, "hsz-utility-medical-1p5.jld2"),
            "--output-dir", joinpath(directory, "hsz")])
        @test isfile(hsz.graph_path)
        @test isfile(hsz.timing_path)
        @test isfile(hsz.utility_table_path)
        @test readdir(joinpath(directory, "hsz")) ==
              sort(basename.([hsz.graph_path, hsz.timing_path, hsz.utility_table_path]))
    end
end

@testset "heterogeneous AR(1) figure uses its declared CRRA preferences" begin
    updated = Ptd.AR1.load_ar1_updated_paths(
        joinpath(pkgdir(Ptd), "data", "ar1-updated-paths.jld2");
        require_current_source=false)
    traces = updated.traces
    # Reconstruct welfare directly from each plotted consumption path and its
    # own beta/gamma. This detects fixed-log utility evaluations beneath
    # heterogeneous configuration labels.
    for configuration in traces.configurations
        beta, gamma = configuration.beta, configuration.gamma
        columns = configuration.display_range
        for method in (:ptd_sceq, :ptd_osa, :g2egm)
            cash = getproperty(traces.cash, method)[:, columns]
            consumption = getproperty(traces.consumption, method)[:, columns]
            flow = (consumption .^ (1.0 - gamma) .- 1.0) ./ (1.0 - gamma)
            recomputed = vec(sum(flow .* beta .^ (0:size(flow, 1)-1); dims=1))
            @test isapprox(recomputed,
                getproperty(traces.current_utility, method)[columns];
                rtol=1e-12, atol=1e-12)
            @test consumption[end, :] == cash[end, :]
            next_cash = max.(1.1 .* (cash[1:end-1, :] - consumption[1:end-1, :]) .+
                traces.income[2:end, columns], 3.0)
            @test isapprox(cash[2:end, :], next_cash; rtol=1e-13, atol=1e-13)
        end
    end
end

@testset "real stochastic replication smoke runs" begin
    mktempdir() do directory
        # Use actual policy builders and evaluators. The earlier runner tests
        # inject lightweight policies to exercise scheduling and failure paths.
        iid = Ptd.IID.run_welfare_contest(iid_tiny_config(n_contests=2, n_paths=2);
            root=joinpath(directory, "iid"), workers=1,
            scheduler=Ptd.IID.serial_scheduler)
        @test iid.sampler_audit.completed_contests == 2
        @test iid.sampler_audit.failed_contests == 0

        # The scheduling-only fixture uses a narrow income domain. A real Q5
        # solve needs an invariant income interval and continuation cash padding.
        ar1_grid = Ptd.AR1G2ContestGridConfiguration(:real_smoke,
            collect(range(0.0, 12.0; length=17)),
            collect(range(-3.0, 3.0; length=9)),
            collect(range(1.0, 1.02 * 12.0 + exp(3.0); length=25)),
            collect(range(-3.0, 3.0; length=9)))
        ar1_config = Ptd.AR1.AR1ContestRunConfig(n_contests=2, n_paths=2,
            horizon=3, return_rate=0.02, cash_floor=1.0,
            bounds=Ptd.ParameterBounds((0.92, 0.98), (1.0, 2.0), (1.0, 4.0)),
            g2egm_grids=ar1_grid,
            ptd_osa_options=Ptd.PTDOSAOptions(direct_iterations=8))
        ar1 = Ptd.AR1.run_welfare_contest(ar1_config;
            root=joinpath(directory, "ar1"), workers=1,
            scheduler=Ptd.AR1.serial_scheduler)
        @test ar1.sampler_audit.completed_contests == 2
        @test ar1.sampler_audit.failed_contests == 0

        config = Ptd.HSZ.HSZExampleConfig(horizon=3,
            initial_cash_support=[5.5], n_paths=1, paths_per_initial_cash=1,
            asset_points=9, common_cash_points=9,
            post_earnings_points=5, post_medical_points=5,
            common_earnings_points=5, common_medical_points=5)
        hsz = Ptd.HSZ.run_hsz_example(config;
            output_path=joinpath(directory, "hsz-smoke.jld2"))
        restored = Ptd.HSZ.load_hsz_example(hsz.path)
        @test restored.julia.version == string(VERSION)
        for method in Ptd.HSZ.HSZ_EXAMPLE_METHODS
            paths = getproperty(restored.paths, method)
            @test all(isfinite, paths.utility)
            @test all(0 .< paths.consumption .<= paths.cash)
        end
    end
end
