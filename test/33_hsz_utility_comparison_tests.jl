using Test
using JLD2
using LinearAlgebra
using Statistics
using Distributions: TDist, quantile

@testset "HSZ compact CE estimate and interval formatting" begin
    H = Ptd.HSZ
    format = H._utility_tex_mean_interval
    @test format(-0.0, 0.0, -0.0) == raw"0\;[0,\,0]"
    @test format(-2.7041, -4.2327, -1.1389) == raw"-2.7\;[-4.23,\,-1.14]"
    @test format(0.004, -0.001, 0.01) == raw"0.004\;[-0.001,\,0.01]"
    tiny = (-9.284e-6, -4.334e-5, 2.474e-5)
    expected = raw"\bigl(-0.928\;[-4.33,\,2.47]\bigr)\times10^{-5}"
    @test format(tiny...) == expected
    @test length(findall(raw"\times10", format(tiny...))) == 1
    @test format(3.276e-7, -6.344e-7, 1.286e-6) ==
        raw"\bigl(0.328\;[-0.634,\,1.29]\bigr)\times10^{-6}"

    # Invert the log-utility CE transform independently, then check the exact
    # historical interval display retained for archived refinement tables.
    config = H.HSZExampleConfig(horizon=3)
    annuity = sum(config.beta .^ (0:config.horizon-1))
    differences = annuity .* log1p.(tiny ./ 100)
    comparison = Ptd.PairedComparison(:g2egm, :ptd_sceq, 100, 0.95,
        differences[1], 0.0, differences[2], differences[3])
    @test H._utility_tex_ce_comparison(config, comparison) == expected
end

module HSZUtilityScriptSmoke
include(joinpath(@__DIR__, "..", "scripts", "5_hsz.jl"))
end

@testset "HSZ common shocks, one construction per method, and utility evidence" begin
    H = Ptd.HSZ
    config = H.HSZExampleConfig(horizon=3,
        initial_cash_support=[5.5, 10.75], n_paths=4, paths_per_initial_cash=2,
        asset_points=9, common_cash_points=9,
        post_earnings_points=5, post_medical_points=5,
        common_earnings_points=5, common_medical_points=5)
    mktempdir() do root
        source_path = joinpath(root, "source.jld2")
        artifact_path = joinpath(root, "utility.jld2")
        source = H.run_hsz_example(config; output_path=source_path).artifact
        original_bytes = read(source_path)
        result = H.run_hsz_utility_comparison(source_path;
            config, output_path=artifact_path)
        saved = H.load_hsz_utility_comparison(artifact_path; require_current_source=true)
        @test read(source_path) == original_bytes
        @test saved.innovations == source.innovations
        @test saved.initial_cash == [5.5, 5.5, 5.5, 5.5, 10.75, 10.75, 10.75, 10.75]
        @test saved.shock_row == [1, 2, 3, 4, 1, 2, 3, 4]
        @test saved.config.gamma == 1.0
        @test saved.scientific_fingerprint == result.scientific_fingerprint
        @test_throws ArgumentError H.run_hsz_utility_comparison(source_path;
            config, output_path=source_path, overwrite=true)
        @test_throws ArgumentError H.run_hsz_utility_comparison(source_path;
            config, output_path=artifact_path)
        @test_throws ArgumentError H.run_hsz_utility_comparison(source_path;
            output_path=joinpath(root, "wrong-calibration.jld2"))

        # Count actual policy construction calls across all five cash groups.
        # The G2EGM adapter solves once and shares its immutable policy workers.
        calls = Dict(method => 0 for method in H.HSZ_EXAMPLE_METHODS)
        function counted_builder(method, model, process, rule, grid)
            calls[method] += 1
            policies = H._build_worker_policies(method, model, process, rule, grid)
            if method === :g2egm
                @test all(policy === first(policies) for policy in policies)
            end
            return policies
        end
        five_cash = repeat([5.5, 10.75, 16.0, 21.25, 26.5]; inner=4)
        five_shocks = saved.innovations[repeat(1:4, 5), :, :]
        paths = H._utility_method_paths(config, five_shocks, five_cash;
            policy_builder=counted_builder)
        @test all(==(1), values(calls))
        @test all(size(trace.cash) == (3, 20) for trace in values(paths))

        for method in H.HSZ_EXAMPLE_METHODS
            trace = saved.paths[method]
            @test size(trace.cash) == size(trace.consumption) == (3, 8)
            @test trace.cash[end, :] == trace.consumption[end, :]
            @test trace.utility ≈ vec(sum(config.beta .^ (0:2) .* log.(trace.consumption); dims=1))
            @test all(0 .<= trace.floor_hits .<= 2)
        end
        rows = H.hsz_utility_summary(saved)
        @test getfield.(rows, :initial_cash) == [5.5, 10.75]
        @test all(row.n_paths == 4 for row in rows)
        @test rows[1].mean_utility[3] == mean(saved.paths[:g2egm].utility[1:4])

        # Large shared shock variation cancels in paired differences. The
        # interval must use their variance, not either method's marginal spread.
        crafted = deepcopy(saved)
        crafted.paths[:ptd_sceq].utility[1:4] .= [1000.0, 2000.0, 3000.0, 4000.0]
        crafted.paths[:g2egm].utility[1:4] .= [1001.0, 2002.0, 3003.0, 4004.0]
        paired = H.hsz_utility_summary(crafted)[1].comparisons[1]
        @test paired.method_a === :g2egm && paired.method_b === :ptd_sceq
        @test paired.mean_difference == 2.5
        @test paired.standard_error ≈ std([1.0, 2.0, 3.0, 4.0]) / 2
        @test paired.confidence_level ≈ 1 - 0.05 / 4
        @test (paired.ci_upper - paired.mean_difference) / paired.standard_error ≈
            quantile(TDist(3), 1 - 0.05 / (2 * 4)) atol=1e-10
        pointwise = H.hsz_utility_summary(crafted; adjustment=:none)[1].comparisons[1]
        @test paired.ci_lower < pointwise.ci_lower < pointwise.ci_upper < paired.ci_upper
        annuity = sum(config.beta .^ (0:config.horizon-1))
        @test H.hsz_ce_percent(config, annuity * log(1.02)) ≈ 2.0 atol=1e-12
        # A 20% shortfall in one direction is a 25% gain in the reverse direction.
        @test H.hsz_ce_percent(config, annuity * log(0.8)) ≈ -20.0 atol=1e-12
        @test H.hsz_ce_percent(config, -annuity * log(0.8)) ≈ 25.0 atol=1e-12
        @test H.hsz_ce_percent(config, 0.0) == 0.0
        @test H.hsz_ce_percent(config, paired.ci_lower) <
            H.hsz_ce_percent(config, paired.mean_difference) < H.hsz_ce_percent(config, paired.ci_upper)
        @test_throws ArgumentError H.hsz_utility_summary(crafted; family_size=1)

        # Quantiles describe paired realized-path gaps, rather than intervals
        # around the expected utility difference. Transform each path before
        # interpolating quantiles: exp(quantile(delta)) is not equivalent.
        quantile_fixture = deepcopy(saved)
        target_gaps = [-20.0, -1.0, 2.0, 30.0]
        common_utility = [1000.0, 2000.0, 3000.0, 4000.0]
        quantile_fixture.paths[:g2egm].utility[1:4] .= common_utility
        quantile_fixture.paths[:ptd_sceq].utility[1:4] .=
            common_utility .+ annuity .* log1p.(target_gaps ./ 100)
        quantile_fixture.paths[:ptd_osa].utility[1:4] .=
            quantile_fixture.paths[:g2egm].utility[1:4]
        quantile_rows = H.hsz_utility_quantiles(quantile_fixture)
        @test getproperty.(quantile_rows, :initial_cash) == [5.5, 10.75]
        @test all(row.n_paths == 4 for row in quantile_rows)
        @test quantile_rows[1].floor_paths == rows[1].floor_paths
        @test collect(values(quantile_rows[1].ce_quantiles[1])) ≈
            [-17.15, 0.5, 25.8] atol=1e-10
        @test values(quantile_rows[1].ce_quantiles[2]) == (0.0, 0.0, 0.0)
        differences = quantile_fixture.paths[:ptd_sceq].utility[1:4] .-
            quantile_fixture.paths[:g2egm].utility[1:4]
        @test !isapprox(quantile_rows[1].ce_quantiles[1].median,
            H.hsz_ce_percent(config, median(differences)); atol=1e-5)

        table_directory = joinpath(root, "table")
        table_path = joinpath(table_directory, "utility.tex")
        rendered = H.render_hsz_utility_comparison(artifact_path; table_path)
        @test rendered.table_path == table_path
        @test readdir(table_directory) == ["utility.tex"]
        @test occursin("PTD-SCEQ vs G2EGM", read(table_path, String))
        @test occursin("PTD-OSA vs G2EGM", read(table_path, String))
        @test occursin("P5 & Median & P95 & P5 & Median & P95", read(table_path, String))
        @test !occursin("simultaneous interval", read(table_path, String))
        # Julia raw strings treat backslashes immediately before a closing
        # quote specially; every LaTeX table row must retain two backslashes.
        latex_rows = filter(line -> occursin('&', line), readlines(table_path))
        @test length(latex_rows) == 4
        @test all(endswith(line, "\\\\") for line in latex_rows)
        @test all(length(split(line, '&')) == 10 for line in latex_rows[3:4])
        @test H._utility_tex_number(-0.0) == "0"
        @test H._utility_tex_number(-0.0000321) == "-3.21\\times10^{-5}"
        @test occursin("4/4", read(table_path, String)) || occursin("0/4", read(table_path, String))
        cli_directory = joinpath(root, "cli")
        cli = HSZUtilityScriptSmoke.main(["--render-only",
            "--example-artifact", source_path, "--utility-artifact", artifact_path,
            "--output-dir", cli_directory])
        @test read(cli.utility_table_path) == read(table_path)
        @test isfile(cli.graph_path)
        @test isfile(cli.timing_path)
        @test readdir(cli_directory) ==
              sort(basename.([cli.graph_path, cli.timing_path, cli.utility_table_path]))

        mismatched_directory = joinpath(root, "mismatched")
        @test_throws ArgumentError HSZUtilityScriptSmoke.main([
            "--example-artifact", H.default_hsz_example_artifact_path(),
            "--utility-artifact", artifact_path, "--output-dir", mismatched_directory])
        @test !isdir(mismatched_directory) || isempty(readdir(mismatched_directory))

        # Rendering must not overwrite the input through an alias, or write
        # two distinct output products onto the same underlying file.
        utility_bytes = read(artifact_path)
        for link_kind in (:symbolic, :hard)
            alias = joinpath(root, "artifact-$(link_kind).jld2")
            if link_kind === :symbolic
                symlink(artifact_path, alias)
            else
                Base.Filesystem.hardlink(artifact_path, alias)
            end
            @test_throws ArgumentError H.render_hsz_utility_comparison(artifact_path;
                table_path=alias)
            @test read(artifact_path) == utility_bytes
        end
        @test_throws ArgumentError H.render_hsz_utility_comparison(artifact_path;
            table_path=artifact_path)
        @test read(artifact_path) == utility_bytes
        table_bytes = read(table_path)
        for link_kind in (:symbolic, :hard)
            collision_directory = joinpath(root, "outputs-$(link_kind)")
            mkpath(collision_directory)
            utility_table = joinpath(collision_directory, "hsz_utility_comparison.tex")
            timing_table = joinpath(collision_directory, H.HSZ_TIMING_FILENAME)
            write(utility_table, table_bytes)
            if link_kind === :symbolic
                symlink(utility_table, timing_table)
            else
                Base.Filesystem.hardlink(utility_table, timing_table)
            end
            @test_throws ArgumentError HSZUtilityScriptSmoke.main(["--render-only",
                "--example-artifact", source_path, "--utility-artifact", artifact_path,
                "--output-dir", collision_directory])
            @test read(utility_table) == table_bytes
            @test readdir(collision_directory) ==
                  sort(basename.([utility_table, timing_table]))
        end

        # Refinement reuses every baseline policy path and solves only the finer
        # G2EGM grid. Simulate a baseline produced before this driver-only edit.
        driver = H.HSZ_REFINEMENT_DRIVER
        current_files = H._utility_source_files()
        old_driver_files = copy(current_files)
        old_driver_files[driver] = repeat("a", 64)
        function baseline_with_sources(files)
            provenance = H.HSZUtilityProvenance((name === :source_files ? files :
                name === :source_fingerprint ? Ptd.stable_fingerprint(files) :
                getfield(saved.provenance, name) for name in fieldnames(H.HSZUtilityProvenance))...)
            return H.HSZUtilityArtifact((name === :provenance ? provenance :
                getfield(saved, name) for name in fieldnames(H.HSZUtilityArtifact))...)
        end
        old_driver_baseline = baseline_with_sources(old_driver_files)
        baseline_path = joinpath(root, "baseline-old-driver.jld2")
        JLD2.jldsave(baseline_path; artifact=old_driver_baseline,
            payload_fingerprint=Ptd.stable_fingerprint(H._utility_record(old_driver_baseline)))
        baseline_bytes = read(baseline_path)
        @test_throws ArgumentError H.load_hsz_utility_comparison(baseline_path;
            require_current_source=true)
        @test H._refinement_source_compatibility(current_files, current_files)["mode"] ==
              "identical_sources"
        refinement_path = joinpath(root, "refinement.jld2")
        refined = H.run_hsz_grid_refinement(baseline_path;
            finer_points=[7], output_path=refinement_path)
        loaded_refinement = H.load_hsz_grid_refinement(refinement_path, baseline_path;
            require_current_source=true)
        @test Set(keys(refined["runs"])) == Set(["5", "7"])
        @test read(source_path) == original_bytes
        @test read(artifact_path) == utility_bytes
        @test read(baseline_path) == baseline_bytes
        compatibility = refined["baseline_source_compatibility"]
        @test compatibility["mode"] == "refinement_driver_only"
        @test compatibility["baseline_driver_sha256"] == old_driver_files[driver]
        @test compatibility["refinement_driver_sha256"] == current_files[driver]
        @test refined["source_files"] == current_files
        baseline_run = loaded_refinement.artifact["runs"]["5"]
        @test baseline_run["reused_baseline"]
        @test H._utility_record(baseline_run["trace"]) == H._utility_record(saved.paths[:g2egm])
        @test isnothing(baseline_run["construction_seconds"])
        @test isnothing(baseline_run["evaluation_seconds"])
        @test isnothing(baseline_run["diagnostics"])
        for (key, run) in loaded_refinement.artifact["runs"]
            @test run["config"].common_earnings_points == parse(Int, key)
            @test run["config"].unconditional_log_medical == config.unconditional_log_medical
            @test run["config"].asset_points == config.asset_points
            @test size(run["trace"].cash) == size(saved.paths[:g2egm].cash)
            @test run["evaluated_paths"] == length(saved.initial_cash)
            if key != "5"
                @test !run["reused_baseline"]
                @test run["diagnostics"].solve.monotonicity_violations_remaining == 0
                @test run["construction_seconds"] >= 0 && run["evaluation_seconds"] >= 0
            end
        end
        refinement_table = joinpath(root, "refinement.tex")
        H.render_hsz_grid_refinement(refinement_path, baseline_path; table_path=refinement_table)
        @test occursin("Bonferroni family size: 8", read(refinement_table, String))
        @test !occursin("timing not remeasured", read(refinement_table, String))
        @test !occursin("replay (8 paths)", read(refinement_table, String))
        @test all(endswith(line, "\\\\") for line in readlines(refinement_table) if occursin('&', line))
        @test_throws ArgumentError H.run_hsz_grid_refinement(baseline_path;
            finer_points=[5], output_path=joinpath(root, "invalid-refinement.jld2"))

        function save_refinement_fixture(name, artifact)
            path = joinpath(root, name)
            JLD2.jldsave(path; artifact,
                payload_fingerprint=Ptd.stable_fingerprint(H._utility_record(artifact)))
            return path
        end
        # Historical rendering checks recorded baseline/refinement identities;
        # require_current_source additionally rejects an outdated driver.
        historical = deepcopy(refined)
        historical["source_files"][driver] = repeat("b", 64)
        historical["baseline_source_compatibility"] = H._refinement_source_compatibility(
            old_driver_files, historical["source_files"])
        historical_path = save_refinement_fixture("historical-refinement.jld2", historical)
        @test H._utility_record(H.load_hsz_grid_refinement(historical_path, baseline_path).artifact) ==
              H._utility_record(historical)
        @test_throws ArgumentError H.load_hsz_grid_refinement(historical_path, baseline_path;
            require_current_source=true)

        # The exception is specific to this driver; it cannot admit a stale
        # solver, changed file set, forged timing, or altered compatibility record.
        solver_path = "src/g2egm/hsz_solver.jl"
        stale_files = copy(old_driver_files)
        stale_files[solver_path] = repeat("c", 64)
        stale_baseline = baseline_with_sources(stale_files)
        stale_path = joinpath(root, "stale-solver-baseline.jld2")
        JLD2.jldsave(stale_path; artifact=stale_baseline,
            payload_fingerprint=Ptd.stable_fingerprint(H._utility_record(stale_baseline)))
        @test_throws ArgumentError H.run_hsz_grid_refinement(stale_path;
            finer_points=[7], output_path=joinpath(root, "stale-result.jld2"))
        @test !isfile(joinpath(root, "stale-result.jld2"))
        stale_refinement = deepcopy(refined)
        stale_refinement["source_files"][solver_path] = repeat("c", 64)
        stale_result = save_refinement_fixture("stale-refinement.jld2", stale_refinement)
        @test_throws ArgumentError H.load_hsz_grid_refinement(stale_result, baseline_path)
        missing_file = copy(old_driver_files)
        delete!(missing_file, solver_path)
        @test_throws ArgumentError H._refinement_source_compatibility(missing_file, current_files)
        false_timing = deepcopy(refined)
        false_timing["runs"]["5"]["construction_seconds"] = 0.0
        false_timing_path = save_refinement_fixture("invented-timing.jld2", false_timing)
        @test_throws ArgumentError H.load_hsz_grid_refinement(false_timing_path, baseline_path)
        false_compatibility = deepcopy(refined)
        false_compatibility["baseline_source_compatibility"]["reason"] = "unrecorded exception"
        false_compatibility_path = save_refinement_fixture("false-compatibility.jld2", false_compatibility)
        @test_throws ArgumentError H.load_hsz_grid_refinement(false_compatibility_path, baseline_path)

        # Digest checks detect any changed payload; independent reconstruction
        # still rejects inconsistent economic outcomes even with a new digest.
        payload = JLD2.load(artifact_path)
        tampered = deepcopy(payload["artifact"])
        tampered.paths[:g2egm].utility[1] += 1.0
        bad_path = joinpath(root, "bad.jld2")
        JLD2.jldsave(bad_path; artifact=tampered,
            payload_fingerprint=payload["payload_fingerprint"])
        @test_throws ArgumentError H.load_hsz_utility_comparison(bad_path)
        JLD2.jldsave(bad_path; artifact=tampered,
            payload_fingerprint=Ptd.stable_fingerprint(H._utility_record(tampered)))
        @test_throws ArgumentError H.load_hsz_utility_comparison(bad_path)
        tampered = deepcopy(saved)
        tampered.paths[:g2egm].floor_hits[1] += 1
        JLD2.jldsave(bad_path; artifact=tampered,
            payload_fingerprint=Ptd.stable_fingerprint(H._utility_record(tampered)))
        @test_throws ArgumentError H.load_hsz_utility_comparison(bad_path)
        tampered = deepcopy(saved)
        tampered.shock_row[5] = 2
        JLD2.jldsave(bad_path; artifact=tampered,
            payload_fingerprint=Ptd.stable_fingerprint(H._utility_record(tampered)))
        @test_throws ArgumentError H.load_hsz_utility_comparison(bad_path)
    end
end
