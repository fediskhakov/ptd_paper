# Tests for src/simulation/ar1/rendering.jl and source-order file 29.
using Dates
using JLD2
using Printf
using Test
import Ptd

const AR1RenderStage = Ptd.AR1

@testset "AR(1) paper sample note is a presentation option" begin
    contest_path = joinpath(pkgdir(Ptd), "data", "ar1-contest-welfare.jld2")
    exhibit_path = joinpath(pkgdir(Ptd), "data", "ar1-contest.jld2")
    original_hashes = [AR1RenderStage._ar1_render_sha256_file(path)
        for path in (contest_path, exhibit_path)]
    mktempdir() do directory
        complete_path = joinpath(directory, "complete.tex")
        compact_path = joinpath(directory, "compact.tex")
        AR1RenderStage.render_ar1_paper_table(contest_path, exhibit_path;
            output_path=complete_path)
        AR1RenderStage.render_ar1_paper_table(contest_path, exhibit_path;
            output_path=compact_path, include_sample_note=false)
        complete, compact = read(complete_path, String), read(compact_path, String)
        contest = AR1RenderStage.load_ar1_contest_welfare(contest_path)
        sample = Ptd.IID._completion_status(contest)
        if isnothing(sample.original_failed_contests)
            @test occursin("the original solver-failure count is unavailable", complete)
        else
            @test occursin("Original solver failures: $(sample.original_failed_contests)/$(sample.target_contests)", complete)
            if sample.replacement_count > 0
                @test occursin("$(sample.replacement_count) replacement configurations", complete)
            else
                @test !occursin("replacement configurations are included", complete)
            end
        end
        @test !occursin("\\emph{Sample:}", compact)
        @test first(split(complete, "\\par\\smallskip"; limit=2)) == compact
        @test endswith(compact, "\\end{tabular}\n")
        # A missing independent single-thread artifact is never replaced by
        # the archived heterogeneous contest's batch times.
        @test occursin("not reported", compact)
        @test !occursin("Batch seconds", compact)
        @test occursin("Mean time", compact)
        @test occursin("Median time", compact)
        @test occursin("\\makecell{Point win\\\\(\\%)}", compact)
        @test occursin("\\makecell{Resolved win\\\\(\\%)}", compact)
        independent_path = joinpath(directory, "without-exhibit.tex")
        AR1RenderStage.render_ar1_paper_table(contest_path, "absent-exhibit.jld2";
            output_path=independent_path, include_sample_note=false)
        @test read(independent_path, String) == compact

        @test [AR1RenderStage._ar1_render_sha256_file(path)
            for path in (contest_path, exhibit_path)] == original_hashes
    end
end

@testset verbose=true "authenticated processed-to-render AR(1) stage" begin
    @test AR1RenderStage._ar1_render_method_label(:ptd_sceq) == "PTD-SCEQ"
    @test AR1RenderStage._ar1_render_method_label(:ptd_osa) == "PTD-OSA"
    @test AR1RenderStage._ar1_render_method_label(:g2egm) == "G2EGM"
    @test_throws ArgumentError AR1RenderStage._ar1_render_method_label(:unknown)
    @test AR1RenderStage.AR1_ENVIRONMENT_LABEL ==
        "AR(1) (\$\\rho=0.9\$, \$\\sigma=0.1\$, \$Y_1=1\$)"

    source_code = read(joinpath(pkgdir(Ptd), "src", "simulation", "ar1",
        "rendering.jl"), String)
    @test !occursin("AR1ContestRunner", source_code)
    @test !occursin("build_ar1_contest_policy", source_code)
    @test !occursin("process_ar1_contest(", source_code)

    render_script = read(joinpath(
        pkgdir(Ptd), "scripts", "4_ar1.jl"), String)
    @test occursin("@quickactivate \"Ptd\"", render_script)

    mktempdir() do root
        config = ar1_tiny_config(n_contests=2, n_paths=4)
        hooks = ar1_fake_hooks()
        context = AR1RenderStage._capture_run_context(config, pkgdir(Ptd))
        welfare = AR1RenderStage._run_welfare_unlocked(config, hooks;
            workers=1, scheduler=AR1RenderStage.serial_scheduler, context)
        selection = (design=welfare.design,
            plain_design=welfare.outer_design,
            audit=welfare.sampler_audit)
        timing = AR1RenderStage._run_timing(config, hooks, selection;
            workers=1, scheduler=AR1RenderStage.serial_scheduler, context)
        processed_path = joinpath(root, "ar1-contest-welfare.jld2")
        AR1RenderStage.publish_ar1_contest(config, welfare, timing;
            output_path=processed_path, context,
            completed_at=DateTime(2026, 8, 5, 10, 0, 0), compress=false)
        output_root = joinpath(root, "rendered")
        completed_at = DateTime(2026, 8, 5, 14, 0, 0)
        result = AR1RenderStage.render_ar1_contest(processed_path;
            output_root,
            completed_at,
            compress=false,
        )
        @test result.status === :computed
        @test isdir(result.output_directory)
        @test isfile(result.manifest_path)
        @test Set(keys(result.table_paths)) ==
            Set((:outcome_shares, :ce_gaps, :timings))
        @test all(isfile, values(result.table_paths))
        paper_ce = AR1RenderStage.render_ar1_ce_gap_table(processed_path;
            output_path=joinpath(root, "paper-ce.tex"))
        @test read(paper_ce) == read(result.table_paths[:ce_gaps])
        paper_summary = AR1RenderStage.render_ar1_paper_table(processed_path,
            "unused-exhibit"; output_path=joinpath(root, "paper-summary.tex"),
            include_sample_note=false)
        # Winning both of the two contests is 100 percent, not a count of two.
        @test occursin("G2EGM & 100 & 100 & ", read(paper_summary, String))


        manifest = AR1RenderStage.load_ar1_render_manifest(
            result.manifest_path)
        metadata = manifest["metadata"]
        configuration = manifest["render_configuration"]
        @test metadata.artifact_schema ==
            AR1RenderStage.AR1_RENDER_CONTEST_SCHEMA_VERSION
        @test metadata.artifact_schema == "ptd-ar1-contest-render-v2"
        @test metadata.status === :complete
        @test metadata.completed_utc == "2026-08-05T14:00:00.000Z"
        @test manifest["source_provenance"].status === :authenticated
        @test configuration.calibration ==
            (rho=0.9, sigma=0.1, initial_income=1.0)
        @test configuration.environment_label ==
            AR1RenderStage.AR1_ENVIRONMENT_LABEL
        @test configuration.method_display.g2egm == "G2EGM"
        @test configuration.n_contests == 2
        @test configuration.n_paths == 4
        @test configuration.timing_paths == 8
        @test length(manifest["outcome_rows"]) == 3
        @test length(manifest["ce_gap_rows"]) == 3
        @test length(manifest["timing_rows"]) == 3
        @test all(row.panel === :ar1
            for row in manifest["outcome_rows"])
        @test all(row.panel === :ar1
            for row in manifest["ce_gap_rows"])
        @test all(row.panel === :ar1
            for row in manifest["timing_rows"])
        @test all(length(row.ce_gap_samples_bps) == 2
            for row in manifest["ce_gap_rows"])
        @test all(row.interval_endpoint_kind ===
            :plugin_ce_scaled_paired_utility_gap_interval &&
            row.interval_has_nominal_ce_coverage === false
            for row in manifest["ce_gap_rows"])
        @test all(row.thread_count == 1
            for row in manifest["timing_rows"])
        @test all(row.timing_paths == 8 &&
            row.amortized_seconds_per_path == row.total_seconds / 8
            for row in manifest["timing_rows"])

        for path in values(result.table_paths)
            contents = read(path, String)
            @test occursin("% Processed result fingerprint: ", contents)
            @test occursin("% Row-data fingerprint: ", contents)
            @test occursin("\\begin{tabular}", contents)
            @test count(line -> occursin(
                    AR1RenderStage.AR1_ENVIRONMENT_LABEL, line),
                split(contents, '\n')) == (path == result.table_paths[:ce_gaps] ? 1 : 3)
        end
        outcome_tex = read(result.table_paths[:outcome_shares], String)
        ce_tex = read(result.table_paths[:ce_gaps], String)
        timing_tex = read(result.table_paths[:timings], String)
        @test count(occursin("G2EGM"),
            split(outcome_tex, '\n')) == 1
        @test length(findall("G2EGM", ce_tex)) == 2
        @test count(occursin("G2EGM"),
            split(timing_tex, '\n')) == 1
        @test occursin("Total seconds (2 tasks)", timing_tex)
        @test occursin("Amortized seconds/path", timing_tex)
        @test count("\\begin{tabular}{llrrrrrr}", ce_tex) == 1
        @test occursin("P5 & Median & P95 & A resolved & B resolved & Unresolved", ce_tex)
        @test occursin("One basis point equals 0.01\\%", ce_tex)
        # The merged table retains each pair's three quantiles and resolution shares.
        fields = (:q05_bps, :median_bps, :q95_bps,
            :method_a_resolved_share, :method_b_resolved_share,
            :pairwise_unresolved_share)
        digits = (3, 3, 3, 2, 2, 2)
        lines = filter(line -> startswith(line, "PTD-"), split(ce_tex, '\n'))
        @test length(lines) == length(manifest["ce_gap_rows"])
        for (line, row) in zip(lines, manifest["ce_gap_rows"])
            cells = strip.(split(replace(line, "\\\\" => ""), '&'))
            @test length(cells) == 8
            expected = [round(getproperty(row, field) *
                (endswith(string(field), "_share") ? 100 : 1); digits=precision)
                for (field, precision) in zip(fields, digits)]
            @test parse.(Float64, cells[3:end]) == expected
        end
        @test !occursin("Median plug-in", ce_tex)
        @test !occursin("P25", ce_tex) && !occursin("P75", ce_tex)
        @test !occursin("\\emph{Sample:}", ce_tex)
        @test occursin("\\fnote{Notes:", ce_tex)
        @test occursin("paired 95\\% utility-difference interval", ce_tex)

        reused = AR1RenderStage.render_ar1_contest(processed_path;
            output_root,
            completed_at=completed_at + Day(1),
            compress=false,
        )
        @test reused.status === :reused
        @test reused.metadata.completed_utc == metadata.completed_utc

        corrupted_root = joinpath(root, "corrupted-render")
        corrupted = AR1RenderStage.render_ar1_contest(processed_path;
            output_root=corrupted_root,
            completed_at,
            compress=false,
        )
        open(corrupted.table_paths[:timings], "a") do io
            println(io, "% corruption")
        end
        @test_throws ArgumentError AR1RenderStage.load_ar1_render_manifest(
            corrupted.manifest_path)
        @test_throws ArgumentError AR1RenderStage.render_ar1_contest(
            processed_path;
            output_root=corrupted_root,
            completed_at,
            compress=false,
        )

        invalid = JLD2.load(processed_path)
        invalid["metadata"] = merge(invalid["metadata"],
            (status=:incomplete,))
        invalid_path = joinpath(root, "invalid-processed.jld2")
        JLD2.jldopen(invalid_path, "w"; compress=false) do file
            for (name, value) in invalid
                file[name] = value
            end
        end
        rejected_root = joinpath(root, "must-not-exist")
        @test_throws ArgumentError AR1RenderStage.render_ar1_contest(
            invalid_path;
            output_root=rejected_root,
            completed_at,
            compress=false,
        )
        @test !ispath(rejected_root)

        missing_root = joinpath(root, "missing-input-must-not-exist")
        @test_throws ArgumentError AR1RenderStage.render_ar1_contest(
            joinpath(root, "missing-processed.jld2");
            output_root=missing_root,
            completed_at,
            compress=false,
        )
        @test !ispath(missing_root)
    end
end
@testset "authenticated heterogeneous AR(1) path replay" begin
    config = ar1_tiny_config(n_contests=2, n_paths=4)
    hooks = ar1_fake_hooks(evaluate=_ar1_trace_evaluate)
    mktempdir() do root
        context = AR1RenderStage._capture_run_context(config, pkgdir(Ptd))
        welfare = AR1RenderStage._run_welfare_unlocked(config, hooks;
            workers=1, scheduler=AR1RenderStage.serial_scheduler, context)
        selection = (design=welfare.design,
            plain_design=welfare.outer_design,
            audit=welfare.sampler_audit)
        timing = AR1RenderStage._run_timing(config, hooks, selection;
            workers=1, scheduler=AR1RenderStage.serial_scheduler, context)
        artifact_path = joinpath(root, "ar1-contest-welfare.jld2")
        AR1RenderStage.publish_ar1_contest(config, welfare, timing;
            output_path=artifact_path, context,
            completed_at=DateTime(2026, 8, 17, 10, 0, 0),
            compress=false)

        targets = (welfare.design[2].initial_cash,
            welfare.design[1].initial_cash)
        traces = AR1RenderStage.replay_ar1_contest_paths(artifact_path;
            config, targets, paths_per_configuration=2,
            workers=1, hooks)
        @test getproperty.(traces.configurations, :contest_id) == [2, 1]
        @test traces.path_ids == collect(1:4)
        @test traces.configuration_groups == [1, 1, 2, 2]
        @test traces.source_contest_ids == [2, 2, 1, 1]
        @test traces.source_path_ids == [1, 2, 1, 2]
        @test size(traces.income) == (config.horizon, 4)
        @test all(size(getproperty(traces.cash, method)) ==
            (config.horizon, 4) for method in config.methods)
        @test all(size(getproperty(traces.consumption, method)) ==
            (config.horizon, 4) for method in config.methods)
        @test all(getproperty(traces.cash, method)[1, :] ==
            traces.initial_cash for method in config.methods)
        design_config = AR1RenderStage.contest_design_config(config)
        expected_innovations = vcat(
            _ar1_fake_predraw(welfare.design[2], design_config, :ar1)[1:2, :],
            _ar1_fake_predraw(welfare.design[1], design_config, :ar1)[1:2, :])
        @test traces.innovation_rows == expected_innovations
        artifact = AR1RenderStage.load_ar1_contest_welfare(artifact_path)
        for (group, configuration) in pairs(traces.configurations)
            columns = configuration.display_range
            for (method_index, method) in pairs(config.methods)
                @test getproperty(traces.replayed_utility, method)[columns] ==
                    artifact["pathwise_utilities"][1:2, method_index,
                        configuration.contest_index, 1]
            end
        end
        repeated = AR1RenderStage.replay_ar1_contest_paths(artifact_path;
            config, targets, paths_per_configuration=2,
            workers=2, hooks)
        @test repeated.replay_fingerprint == traces.replay_fingerprint

        bad_predraw = function (spec, design, panel)
            innovations = _ar1_fake_predraw(spec, design, panel)
            innovations[1, 1] += 1.0
            return innovations
        end
        bad_hooks = AR1RenderStage.AR1RunnerHooks(
            predraw=bad_predraw,
            build_policy=hooks.build_policy,
            evaluate=hooks.evaluate,
            diagnostics=hooks.diagnostics,
            reset_diagnostics! = hooks.reset_diagnostics!,
            quadrature_audit=hooks.quadrature_audit,
            validate_policy=hooks.validate_policy)
        @test_throws ArgumentError AR1RenderStage.replay_ar1_contest_paths(
            artifact_path; config, targets=(targets[1],),
            paths_per_configuration=2, workers=1, hooks=bad_hooks)

        tampered = copy(artifact)
        tampered["pathwise_utilities"] =
            copy(artifact["pathwise_utilities"])
        first_contest = traces.configurations[1].contest_index
        tampered["pathwise_utilities"][1, 1, first_contest, 1] += 1.0
        specs = AR1RenderStage._ar1_contest_specs(artifact)
        chosen = only(
            AR1RenderStage._select_ar1_contest_path_configurations(
                specs, (targets[1],)))
        @test_throws ArgumentError AR1RenderStage._replay_ar1_contest_configuration(
            tampered, config, design_config, chosen, 2, hooks)
        @test traces.outside_reported_timing
        @test !traces.adds_welfare_observations

        figure = AR1RenderStage._ar1_contest_trace_figure(traces;
            size=(720, 520), dpi=72)
        figure_path = joinpath(root, "ar1_policy_paths.png")
        AR1RenderStage._atomic_savefig(figure, figure_path)
        @test isfile(figure_path)
        @test read(figure_path)[1:8] ==
            UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
    end

    specs = [Ptd.ContestSpec(1, 0.95, 1.0, 4.0),
        Ptd.ContestSpec(2, 0.96, 2.0, 6.0),
        Ptd.ContestSpec(3, 0.97, 3.0, 8.0)]
    ties = AR1RenderStage._select_ar1_contest_path_configurations(
        specs, (5.0, 8.0))
    @test getproperty.(getproperty.(ties, :spec), :id) == [1, 3]
    @test_throws ArgumentError AR1RenderStage._select_ar1_contest_path_configurations(
        specs, ())
    @test_throws ArgumentError AR1RenderStage._select_ar1_contest_path_configurations(
            specs, (4.0, 6.0, 8.0, 10.0))
end

@testset "AR(1) path replay maps successful rows to original IDs" begin
    config = ar1_tiny_config(n_contests=3, n_paths=4)
    calls = Ref(0)
    evaluate = function (args...)
        calls[] += 1
        calls[] == 2 && error("injected failure before compact replay")
        return _ar1_trace_evaluate(args...)
    end
    hooks = ar1_fake_hooks(; evaluate)
    context = AR1RenderStage._capture_run_context(config, pkgdir(Ptd))
    welfare = AR1RenderStage._run_welfare_unlocked(config, hooks;
        workers=1, scheduler=AR1RenderStage.serial_scheduler, context)
    selection = (design=welfare.design, plain_design=welfare.outer_design,
        audit=welfare.sampler_audit)
    timing = AR1RenderStage._run_timing(config, hooks, selection;
        workers=1, scheduler=AR1RenderStage.serial_scheduler, context)
    mktempdir() do root
        path = joinpath(root, "ar1-replay-failures.jld2")
        AR1RenderStage.publish_ar1_contest(config, welfare, timing;
            output_path=path, context, compress=false)
        traces = AR1RenderStage.replay_ar1_contest_paths(path;
            config, targets=(2.0, 3.0), paths_per_configuration=2,
            workers=1, hooks=ar1_fake_hooks(evaluate=_ar1_trace_evaluate))
        @test sort(unique(traces.source_contest_ids)) == [2, 3]
        @test sort(getproperty.(traces.configurations, :contest_index)) == [1, 2]
        @test sort(getproperty.(traces.configurations, :contest_id)) == [2, 3]
        @test size(traces.income) == (config.horizon, 4)
    end
end
