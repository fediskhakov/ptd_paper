# Tests for src/simulation/ar1/exhibit_rendering.jl.
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Dates
using Random
using Test
import Ptd

const SingleContestFixture = Ptd.AR1
const SingleRenderFixture = Ptd.AR1
const StableFixture = Ptd

function _ar1_exhibit_render_fixture_artifact()
    grids = Ptd.AR1G2ContestGridConfiguration(
        :ar1_exhibit_render_fixture_v1,
        [0.0, 1.0, 4.0, 10.0],
        [-0.2, 0.0, 0.2],
        [3.0, 4.0, 7.0, 12.0],
        [-0.2, 0.0, 0.2])
    config = SingleContestFixture.AR1ContestExhibitConfig(
        horizon=3, initial_cash_support=[5.0, 7.0],
        n_paths=4, trace_paths=2, workers=1,
        surface_cash_grid=[3.0, 7.0],
        surface_log_income_grid=[-0.1, 0.1],
        g2egm_grids=grids)
    scientific = SingleContestFixture.scientific_configuration(config)
    scientific_fingerprint = StableFixture.stable_fingerprint(scientific)

    utility = [
        1.00 1.03 0.98
        1.10 1.08 1.04
        0.92 0.97 0.95
        1.06 1.10 1.01
    ]
    innovations = randn(Xoshiro(config.evaluation_seed),
        config.n_paths, config.horizon - 1)
    initial_cash_assignments =
        SingleContestFixture._initial_cash_assignments(config)
    statistics = SingleContestFixture._welfare_statistics(config, utility)
    welfare = (;
        methods=collect(SingleContestFixture.AR1_CONTEST_EXHIBIT_METHODS),
        utility,
        floor_hits=zeros(Int, size(utility)),
        total_transfers=zeros(size(utility)),
        expected_utility=statistics.expected_utility,
        comparisons=statistics.comparisons,
        innovations,
        shock_fingerprint=StableFixture.stable_fingerprint(innovations),
        initial_cash_assignments,
        initial_cash_assignments_fingerprint=
            StableFixture.stable_fingerprint(initial_cash_assignments),
        n_paths=config.n_paths,
        confidence_level=config.confidence_level,
    )

    cash_grid = copy(config.surface_cash_grid)
    surface_consumption = (;
        ptd_sceq=[1.00 1.10; 2.00 2.10],
        ptd_osa=[1.05 1.15; 2.05 2.15],
        g2egm=[0.98 1.08; 1.98 2.08],
    )
    surface_saving = (;
        (method => reshape(cash_grid, :, 1) .-
            getproperty(surface_consumption, method)
         for method in SingleContestFixture.AR1_CONTEST_EXHIBIT_METHODS)...)
    surface = (;
        date=1,
        cash_grid,
        log_income_grid=copy(config.surface_log_income_grid),
        consumption=surface_consumption,
        saving=surface_saving,
        query_seconds=(ptd_sceq=0.01, ptd_osa=0.02, g2egm=0.03),
        thread_ids=(ptd_sceq=[1], ptd_osa=[1], g2egm=[1]),
        outside_reported_timing=true,
    )

    trace_path_ids = SingleContestFixture._trace_path_ids(config)
    income = Matrix{Float64}(undef, config.horizon, config.trace_paths)
    income[1, :] .= config.initial_income
    trace_log_income = fill(log(config.initial_income), config.trace_paths)
    for date in 2:config.horizon
        trace_log_income .= config.unconditional_log_mean .+
            config.rho .* (trace_log_income .-
                config.unconditional_log_mean) .+
            config.sigma .* view(innovations, trace_path_ids, date - 1)
        income[date, :] .= exp.(trace_log_income)
    end
    trace_cash = (;
        ptd_sceq=[5.0 7.0; 4.8 6.9; 4.5 6.6],
        ptd_osa=[5.0 7.0; 4.7 6.8; 4.4 6.5],
        g2egm=[5.0 7.0; 4.75 6.85; 4.45 6.55],
    )
    trace_consumption = (;
        ptd_sceq=[1.1 1.1; 1.2 1.2; 4.5 6.6],
        ptd_osa=[1.0 1.0; 1.2 1.2; 4.4 6.5],
        g2egm=[1.05 1.05; 1.2 1.2; 4.45 6.55],
    )
    traces = (;
        dates=collect(1:config.horizon),
        path_ids=trace_path_ids,
        initial_cash=initial_cash_assignments[trace_path_ids],
        income,
        cash=trace_cash,
        consumption=trace_consumption,
        replay_seconds=(ptd_sceq=0.1, ptd_osa=0.2, g2egm=0.3),
        replayed_utility=(ptd_sceq=utility[trace_path_ids, 1],
            ptd_osa=utility[trace_path_ids, 2],
            g2egm=utility[trace_path_ids, 3]),
        innovation_rows=innovations[trace_path_ids, :],
        outside_reported_timing=true,
        adds_welfare_observations=false,
    )

    construction = (ptd_sceq=0.1, ptd_osa=0.2, g2egm=0.3)
    evaluation = (ptd_sceq=0.4, ptd_osa=0.5, g2egm=0.6)
    timing = (;
        convention=:one_construction_plus_one_common_path_block,
        construction_seconds=construction,
        evaluation_seconds=evaluation,
        end_to_end_seconds=(;
            (method => construction[method] + evaluation[method]
             for method in SingleContestFixture.AR1_CONTEST_EXHIBIT_METHODS)...),
        evaluation_boundary_thread_ids=(
            ptd_sceq=[1], ptd_osa=[1], g2egm=[1]),
        partition_ranges=[1:4],
        julia_threads=1,
        configured_workers=1,
        blas_threads=1,
        timing_observations_per_method=1,
        compilation_warmup_outside=true,
        shock_generation_outside=true,
        validation_outside=true,
        io_outside=true,
        surface_queries_outside=true,
        trace_replay_outside=true,
        no_timing_dispersion_estimate=true,
    )
    process = Ptd.LogAR1Process(rho=config.rho, sigma=config.sigma,
        unconditional_log_mean=config.unconditional_log_mean)
    rule = Ptd.gauss_hermite_rule(process, 5)
    node_bits = SingleContestFixture._float_bits(rule.nodes)
    weight_bits = SingleContestFixture._float_bits(rule.weights)
    quadrature = (;
        ptd_sceq=(method=:ptd_sceq, uses_quadrature=false),
        ptd_osa=(method=:ptd_osa, uses_quadrature=true,
            nodes_per_dimension=5, base_nodes=rule.nodes,
            base_weights=rule.weights),
        g2egm=(method=:g2egm, uses_quadrature=true,
            nodes_per_dimension=5, base_nodes=rule.nodes,
            base_weights=rule.weights),
        exact_shared_q5=true,
        node_bits,
        weight_bits,
    )
    diagnostic = (;
        ptd_sceq=(calls=1,),
        ptd_osa=[(certification_failures=0, fallback_uses=0,
            calls=1, certified_calls=1)],
        g2egm=(solver=(;
            query=(domain_errors=0, asset_ceiling_choices=0),
            solve=(policy_domain_asset_ceiling_winners=0,)),),
    )
    solver_diagnostics = (;
        scope=:welfare_surface_and_trace_queries,
        display_queries_outside_reported_timing=true,
        final=diagnostic,
    )
    files = [(path="fixture.jl", sha256=repeat("0", 64))]
    provenance = (;
        simulation=(source=(commit="fixture-commit",),),
        input_tree=(files, fingerprint=StableFixture.stable_fingerprint(files)),
        upstream=(repository="JeppeDruedahl/G2EGM",
            commit=scientific.g2egm.upstream_commit,
            solver_schema=scientific.g2egm.solver_schema,
            runtime_dependencies=:pure_julia_no_matlab_mex_or_cpp),
    )
    result_payload = SingleContestFixture._result_payload(
        scientific_fingerprint, quadrature, solver_diagnostics,
        welfare, timing, surface, traces)
    result_fingerprint = StableFixture.stable_fingerprint(result_payload)
    timing_fingerprint = StableFixture.stable_fingerprint(timing)
    provenance_fingerprint = StableFixture.stable_fingerprint(provenance)
    completed_utc = "2026-08-05T00:00:00.000Z"
    authentication_fingerprint =
        SingleContestFixture._artifact_authentication(
            scientific_fingerprint, result_fingerprint,
            timing_fingerprint, provenance, completed_utc)
    artifact = (;
        metadata=(;
            artifact_schema=SingleContestFixture.AR1_CONTEST_EXHIBIT_SCHEMA,
            authentication_schema=
                SingleContestFixture.AR1_CONTEST_EXHIBIT_AUTHENTICATION_SCHEMA,
            status=:complete,
            completed_utc,
            scientific_fingerprint,
            result_fingerprint,
            timing_fingerprint,
            provenance_fingerprint,
            authentication_fingerprint,
        ),
        scientific_configuration=scientific,
        provenance,
        quadrature,
        solver_diagnostics,
        welfare,
        timing,
        surface,
        traces,
    )
    SingleContestFixture.validate_ar1_contest_exhibit_artifact(
        artifact; expected_config=config, require_publication=false)
    return config, artifact
end

@testset verbose=true "AR(1) contest-exhibit artifact-only rendering" begin
    renderer_source = read(joinpath(pkgdir(Ptd), "src", "simulation", "ar1",
        "exhibit_rendering.jl"), String)
    for forbidden in ("build_ar1_contest_policy", "evaluate_policy_paths",
                      "randn(", "using Random")
        @test !occursin(forbidden, renderer_source)
    end
    @test occursin("load_ar1_contest_exhibit_artifact", renderer_source)
    @test occursin("Plots.surface", renderer_source)
    @test occursin("using Plots.PlotMeasures", renderer_source)

    config, artifact = _ar1_exhibit_render_fixture_artifact()
    mktempdir() do root
        artifact_path = joinpath(root, "contest.jld2")
        SingleContestFixture.save_ar1_contest_exhibit_artifact(
            artifact_path, artifact; compress=false,
            require_publication=false)
        graph_directory = joinpath(root, "Graphs")
        table_directory = joinpath(root, "Tables")
        result = SingleRenderFixture.render_ar1_contest_exhibits(
            artifact_path; graph_directory, table_directory,
            require_publication=false,
            completed_at=DateTime(2026, 8, 5), compress=false,
            surface_size=(720, 260), trace_size=(720, 520), dpi=72)

        @test basename(result.paths_path) ==
            SingleRenderFixture.AR1_CONTEST_EXHIBIT_DIAGNOSTIC_PATHS_FILENAME
        @test !isfile(joinpath(graph_directory,
            SingleRenderFixture.AR1_CONTEST_EXHIBIT_PATHS_FILENAME))
        @test_throws ArgumentError SingleRenderFixture.render_ar1_paper_figures(
            artifact_path; graph_directory, require_publication=false)

        for path in (result.surface_path, result.paths_path)
            bytes = read(path)
            @test length(bytes) > 8
            @test bytes[1:8] == UInt8[
                0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        end
        timing_tex = read(result.timing_path, String)
        welfare_tex = read(result.welfare_path, String)
        for tex in (timing_tex, welfare_tex)
            @test occursin("\\begin{tabular}", tex)
            @test occursin("\\end{tabular}", tex)
            @test occursin("Source authentication fingerprint", tex)
            @test occursin("Row-data fingerprint", tex)
        end
        @test occursin("4-path evaluation", timing_tex)
        @test occursin("Construction (s)", timing_tex)
        @test occursin("End-to-end (s)", timing_tex)
        @test occursin("Amortized seconds/path", timing_tex)
        @test occursin("plug-in consumption-equivalent transforms", welfare_tex)
        @test occursin("not asserted to be population confidence intervals",
            welfare_tex)

        manifest = SingleRenderFixture.
            load_ar1_contest_exhibit_render_manifest(
                result.manifest_path; validate_source=true)
        @test manifest.metadata.status === :complete
        @test length(manifest.timing_rows) == 3
        @test all(row.amortized_seconds_per_path ==
            row.end_to_end_seconds / row.evaluation_paths
            for row in manifest.timing_rows)
        @test length(manifest.welfare_rows) == 3
        @test length(manifest.render_configuration.traces.path_ids) ==
            config.trace_paths
        @test manifest.render_configuration.surface.contest_markers ===
            false
        @test manifest.render_configuration.surface.
            slice_log_income_levels == [-0.1, 0.1]
        @test manifest.render_configuration.traces.initial_cash == [5.0, 7.0]
        @test manifest.render_configuration.traces.path_legend === false
        @test manifest.render_configuration.traces.
            outside_reported_timing === true
        @test manifest.render_configuration.traces.
            adds_welfare_observations === false

        original_timing = read(result.timing_path)
        open(result.timing_path, "a") do io
            write(io, "% tamper\n")
        end
        @test_throws ArgumentError SingleRenderFixture.
            load_ar1_contest_exhibit_render_manifest(
                result.manifest_path; validate_source=true)
        open(result.timing_path, "w") do io
            write(io, original_timing)
        end
        @test SingleRenderFixture.
            load_ar1_contest_exhibit_render_manifest(
                result.manifest_path; validate_source=true).metadata.status ===
            :complete
    end
end

@testset "shipped AR(1) exhibit records the current solver identity" begin
    path = joinpath(pkgdir(Ptd), "data", "ar1-contest.jld2")
    artifact = Ptd.AR1.load_ar1_contest_exhibit_artifact(path;
        require_publication=true, require_live_inputs=false)
    timing = artifact.timing
    # The run recorded 24 default-pool threads and one interactive thread.
    # Julia numbers interactive threads first, so the workers had IDs 2:25.
    @test hasproperty(timing, :thread_pools)
    @test timing.julia_threads == timing.configured_workers == 24
    @test Ptd.AR1._recorded_default_thread_ids(timing,
        timing.julia_threads) == 2:25
    # Timing records without thread-pool fields keep the 1:julia_threads
    # contract, whether they are named tuples or dictionaries.
    for legacy_timing in ((; julia_threads=3), Dict("julia_threads" => 3))
        @test Ptd.AR1._recorded_default_thread_ids(legacy_timing, 3) == 1:3
    end
    @test artifact.scientific_configuration.g2egm.solver_schema ==
        Ptd.G2EGM.G2EGM_SOLVER_SCHEMA
    @test artifact.scientific_configuration.g2egm.upstream_commit ==
        Ptd.G2EGM.G2EGM_REFERENCE_COMMIT
    # The exhibit is the paper configuration at its recorded worker count.
    recorded_config = Ptd.AR1.AR1ContestExhibitConfig(
        workers=timing.configured_workers)
    @test artifact.metadata.scientific_fingerprint ==
        Ptd.AR1.scientific_fingerprint(recorded_config)
    @test Ptd.AR1._is_publication_scientific_configuration(
        artifact.scientific_configuration)
    # A one-ulp alteration is within the numerical reconstruction tolerance,
    # but must still fail the unchanged exact stored-payload authentication.
    altered_rows = copy(artifact.welfare.expected_utility)
    altered_rows[1] = merge(altered_rows[1],
        (mean_utility=nextfloat(altered_rows[1].mean_utility),))
    @test Ptd.AR1._welfare_summary_rows_reproduce(
        altered_rows, artifact.welfare.expected_utility)
    altered = merge(artifact, (welfare=merge(artifact.welfare,
        (expected_utility=altered_rows,)),))
    authentication_error = try
        Ptd.AR1.validate_ar1_contest_exhibit_artifact(altered)
        nothing
    catch error
        error
    end
    @test authentication_error isa ArgumentError
    @test occursin("result fingerprint mismatch",
        sprint(showerror, authentication_error))
    wrong_classification = [merge(artifact.welfare.comparisons[1],
        (classification=:invalid,))]
    @test !Ptd.AR1._welfare_summary_rows_reproduce(wrong_classification,
        artifact.welfare.comparisons[1:1])
    @test Ptd.AR1.load_ar1_contest_exhibit_artifact(path;
        expected_config=recorded_config,
        require_live_inputs=false).metadata == artifact.metadata
    @test_throws ArgumentError Ptd.AR1.load_ar1_contest_exhibit_artifact(path;
        expected_config=Ptd.AR1.AR1ContestExhibitConfig(
            workers=timing.configured_workers, beta=0.97),
        require_live_inputs=false)
    mktempdir() do root
        figures = Ptd.AR1.render_ar1_stored_paper_figures(path;
            graph_directory=root, surface_size=(600, 480),
            trace_size=(600, 700), dpi=72)
        @test figures.scope === :stored_fixed_calibration_traces
        @test figures.calibration == (beta=0.98, gamma=1.0)
        @test figures.n_paths == 100
        @test figures.trace_fingerprint == Ptd.stable_fingerprint(artifact.traces)
        @test figures.source_authentication_fingerprint ==
            artifact.metadata.authentication_fingerprint
        @test isfile(figures.surface_path) && isfile(figures.paths_path)
    end
end
