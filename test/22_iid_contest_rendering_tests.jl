# Tests for src/simulation/iid/rendering.jl.
using Dates
using JLD2
using Test
import Ptd

const RenderProcessing = Ptd.IID
const RenderStage = Ptd.IID
const RenderFingerprint = Ptd.stable_fingerprint

function render_write_jld2(path, artifact)
    JLD2.jldopen(path, "w"; compress=false) do file
        for (name, value) in pairs(artifact)
            file[String(name)] = value
        end
    end
    return path
end

function render_processed_fixture(path)
    config = iid_tiny_config(n_contests=2, n_paths=4)
    hooks = iid_fake_hooks()
    context = RenderProcessing._capture_run_context(config, pkgdir(Ptd))
    welfare = RenderProcessing._run_welfare_unlocked(config, hooks;
        workers=1, scheduler=RenderProcessing.serial_scheduler, context)
    timing = RenderProcessing._run_timing_unlocked(config, hooks;
        design=welfare.design,
        plain_design=welfare.outer_design,
        sampler_audit=welfare.sampler_audit,
        workers=1,
        scheduler=RenderProcessing.serial_scheduler,
        context)
    RenderProcessing.publish_iid_contest(config, welfare, timing;
        output_path=path,
        context,
        completed_at=DateTime(2026, 8, 4, 10, 0, 0),
        compress=false)
    return path
end

function utility_for_consumption_scale(reference_utility, beta, gamma,
                                       horizon, scale)
    discount_mass = sum(Float64(beta) .^ (0:Int(horizon)-1))
    if gamma == 1.0
        return reference_utility + discount_mass * log(scale)
    end
    exponent = 1.0 - gamma
    reference_power = discount_mass + exponent * reference_utility
    return (scale^exponent * reference_power - discount_mass) / exponent
end

@testset verbose=true "authenticated processed-to-render IID contest stage" begin
    for gamma in (0.5, 1.0, 2.0)
        beta = 0.96
        horizon = 5
        reference = 0.15
        scale = 1.0125
        candidate = utility_for_consumption_scale(
            reference, beta, gamma, horizon, scale)
        @test RenderStage.consumption_equivalent_bps(
            candidate, reference, beta, gamma, horizon) ≈
            10_000 * (scale - 1.0) atol=1e-10
        @test RenderStage.consumption_equivalent_bps(
            reference, candidate, beta, gamma, horizon) ≈
            10_000 * (inv(scale) - 1.0) atol=1e-10

        lower_scale = 0.99
        upper_scale = 1.025
        lower_candidate = utility_for_consumption_scale(
            reference, beta, gamma, horizon, lower_scale)
        upper_candidate = utility_for_consumption_scale(
            reference, beta, gamma, horizon, upper_scale)
        interval = RenderStage._consumption_equivalent_interval_bps(
            reference, lower_candidate - reference,
            upper_candidate - reference, beta, gamma, horizon)
        @test interval.lower_bps ≈
            10_000 * (lower_scale - 1.0) atol=1e-10
        @test interval.upper_bps ≈
            10_000 * (upper_scale - 1.0) atol=1e-10
        @test interval.lower_bps < interval.upper_bps
    end
    @test_throws ArgumentError RenderStage.consumption_equivalent_bps(
        4.0, 0.0, 0.95, 2.0, 3)
    expected_escape = string("\\textbackslash{}\\{\\}\\", Char(0x24),
        "\\&\\#\\%\\_\\textasciicircum{}\\textasciitilde{}")
    @test RenderStage._tex_escape("\\{}\$&#%_^~") == expected_escape
    @test RenderStage._panel_label(:new_panel) == "new\\_panel"
    @test RenderStage._panel_label(:baseline, 0.1) ==
        "Baseline (\$\\sigma=0.1\$)"
    panel_sigmas = RenderStage._panel_specifications(
        ((id=:high_variance, sigma=0.5), (id=:baseline, sigma=0.1)),
        [:baseline, :high_variance], "test panels")
    @test panel_sigmas ==
        ((id=:baseline, sigma=0.1), (id=:high_variance, sigma=0.5))
    @test_throws ArgumentError RenderStage._panel_specifications(
        ((id=:baseline, sigma=0.1), (id=:baseline, sigma=0.5)),
        [:baseline, :high_variance], "duplicate test panels")
    @test_throws ArgumentError RenderStage._panel_specifications(
        ((id=:baseline, sigma=NaN), (id=:high_variance, sigma=0.5)),
        [:baseline, :high_variance], "invalid test panels")

    reversed = (comparisons=[(;
        method_a=:ptd_osa,
        method_b=:ptd_sceq,
        confidence_level=0.95,
        ci_lower=-0.4,
        ci_upper=-0.2,
    )],)
    oriented = RenderStage._oriented_comparison(
        reversed, :ptd_sceq, :ptd_osa, "reversed fixture")
    @test oriented.confidence_level == 0.95
    @test oriented.ci_lower == 0.2
    @test oriented.ci_upper == 0.4

    source_code = read(joinpath(pkgdir(Ptd), "src", "simulation", "iid",
        "rendering.jl"), String)
    @test !occursin("ContestRunner", source_code)
    @test !occursin("ContestChunks", source_code)
    @test !occursin("process_iid_contest", source_code)

    mktempdir() do root
        processed_path = render_processed_fixture(
            joinpath(root, "processed-only.jld2"))
        @test !ispath(joinpath(root, "raw"))
        output_root = joinpath(root, "rendered")
        completed_at = DateTime(2026, 8, 4, 14, 0, 0)
        result = RenderStage.render_iid_contest(processed_path;
            output_root, completed_at, compress=false)
        @test result.status === :computed
        @test isdir(result.output_directory)
        @test isfile(result.manifest_path)
        @test Set(keys(result.table_paths)) ==
            Set((:outcome_shares, :ce_gaps, :timings))
        @test all(isfile, values(result.table_paths))
        paper_ce = RenderStage.render_iid_ce_gap_table(processed_path;
            output_path=joinpath(root, "paper-ce.tex"))
        @test read(paper_ce) == read(result.table_paths[:ce_gaps])


        manifest = RenderStage.load_render_manifest(result.manifest_path)
        metadata = manifest["metadata"]
        @test metadata.status === :complete
        @test metadata.completed_utc == "2026-08-04T14:00:00.000Z"
        @test manifest["source_provenance"].status === :authenticated
        @test length(manifest["outcome_rows"]) == 6
        @test length(manifest["ce_gap_rows"]) == 6
        @test length(manifest["timing_rows"]) == 6
        @test all(length(row.ce_gap_samples_bps) == 2
            for row in manifest["ce_gap_rows"])
        @test all(length(row.ce_ci_lower_samples_bps) == 2 &&
            length(row.ce_ci_upper_samples_bps) == 2
            for row in manifest["ce_gap_rows"])
        @test all(row.ci_endpoint_reference ===
            :method_b_expected_utility for row in manifest["ce_gap_rows"])
        @test all(row.interval_endpoint_kind ===
            :plugin_ce_scaled_paired_utility_gap_interval &&
            row.interval_has_nominal_ce_coverage === false
            for row in manifest["ce_gap_rows"])
        @test manifest["render_configuration"].paired_interval_endpoint_kind ===
            :plugin_ce_scaled_paired_utility_gap_interval
        @test manifest["render_configuration"].paired_interval_has_nominal_ce_coverage ===
            false
        @test manifest["render_configuration"].unattainable_plugin_endpoint_policy ===
            :refuse_render
        @test manifest["render_configuration"].paired_interval_endpoint_summary ===
            :median_across_contests
        @test manifest["render_configuration"].paired_interval_summary_is_ci_for_median ===
            false
        @test manifest["render_configuration"].panel_sigmas ==
            ((id=:baseline, sigma=0.1),
             (id=:high_variance, sigma=0.5))
        @test all(isfinite, [row.median_bps
            for row in manifest["ce_gap_rows"]])
        @test all(row.median_ci_lower_bps <= row.median_ci_upper_bps
            for row in manifest["ce_gap_rows"])
        ptd_sceq_vs_osa = only(row for row in manifest["ce_gap_rows"]
            if row.panel === :baseline && row.method_a === :ptd_sceq &&
               row.method_b === :ptd_osa)
        processed = RenderProcessing.load_iid_contest(processed_path)
        baseline_summaries = sort!([summary
            for summary in processed["contest_summaries"]
            if summary.panel === :baseline]; by=summary -> summary.contest_id)
        expected_samples = [RenderStage.consumption_equivalent_bps(
            summary.expected_utility[1], summary.expected_utility[2],
            summary.beta, summary.gamma, 3) for summary in baseline_summaries]
        @test ptd_sceq_vs_osa.ce_gap_samples_bps ≈ expected_samples
        expected_intervals = [begin
            comparison = RenderStage._oriented_comparison(
                summary, :ptd_sceq, :ptd_osa, "exact fixture")
            RenderStage._consumption_equivalent_interval_bps(
                summary.expected_utility[2], comparison.ci_lower,
                comparison.ci_upper, summary.beta, summary.gamma, 3)
        end for summary in baseline_summaries]
        @test ptd_sceq_vs_osa.ce_ci_lower_samples_bps ≈
            getfield.(expected_intervals, :lower_bps)
        @test ptd_sceq_vs_osa.ce_ci_upper_samples_bps ≈
            getfield.(expected_intervals, :upper_bps)
        timing_keys = [(row.panel, row.method)
            for row in manifest["timing_rows"]]
        @test length(unique(timing_keys)) == 6
        @test all(row -> isfinite(row.total_seconds) &&
            row.total_seconds >= 0.0, manifest["timing_rows"])
        @test all(row.timing_paths == 8 &&
            row.amortized_seconds_per_path == row.total_seconds / 8
            for row in manifest["timing_rows"])

        for path in values(result.table_paths)
            contents = read(path, String)
            @test occursin("% Processed result fingerprint: ", contents)
            @test occursin("% Row-data fingerprint: ", contents)
            @test occursin("\\begin{tabular}", contents)
            @test count(line -> occursin(
                    "Baseline (\$\\sigma=0.1\$)", line),
                split(contents, '\n')) == (path == result.table_paths[:ce_gaps] ? 1 : 3)
            @test count(line -> occursin(
                    "Higher variance (\$\\sigma=0.5\$)", line),
                split(contents, '\n')) == (path == result.table_paths[:ce_gaps] ? 1 : 3)
        end
        timing_tex = read(result.table_paths[:timings], String)
        @test count(line -> startswith(line, "Baseline (\$\\sigma=0.1\$) &") ||
            startswith(line, "Higher variance (\$\\sigma=0.5\$) &"),
            split(timing_tex, '\n')) == 6
        @test count(occursin("Amortized seconds/path"),
            split(timing_tex, '\n')) == 1
        ce_tex = read(result.table_paths[:ce_gaps], String)
        blocks = split(ce_tex, "\\begin{tabular}{llrrrrrr}")
        @test length(blocks) == 2
        @test occursin("One basis point equals 0.01\\%", ce_tex)
        # The compact table retains every panel/pair and the six displayed statistics.
        fields = (:q05_bps, :median_bps, :q95_bps,
            :method_a_resolved_share, :method_b_resolved_share,
            :pairwise_unresolved_share)
        digits = (3, 3, 3, 2, 2, 2)
        lines = filter(line -> startswith(line, "PTD-"), split(blocks[2], '\n'))
        @test length(lines) == length(manifest["ce_gap_rows"])
        for (line, row) in zip(lines, manifest["ce_gap_rows"])
            cells = strip.(split(replace(line, "\\\\" => ""), '&'))
            @test length(cells) == 8
            expected = [round(getproperty(row, field) *
                (endswith(string(field), "_share") ? 100 : 1); digits=precision)
                for (field, precision) in zip(fields, digits)]
            @test parse.(Float64, cells[3:end]) == expected
        end
        @test occursin("CEG (basis points)", ce_tex)
        @test occursin("Pairwise resolution (\\%)", ce_tex)
        @test occursin("P5 & Median & P95", ce_tex)
        @test !occursin("Median plug-in", ce_tex)
        @test occursin(raw"\fnote{Notes: CEG is the consumption-equivalent gap; positive values favor method A. One basis point equals 0.01\%. A or B is resolved when its paired 95\% utility-difference interval favors that method.}", ce_tex)

        reused = RenderStage.render_iid_contest(processed_path;
            output_root, completed_at=completed_at + Day(1), compress=false)
        @test reused.status === :reused
        @test reused.metadata.completed_utc == metadata.completed_utc

        paper_table = joinpath(root, "iid_contest_summary.tex")
        script = joinpath(pkgdir(Ptd), "scripts", "3_iid.jl")
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(pkgdir(Ptd)) $script --render-only $processed_path $paper_table`
        @test success(command)
        @test isfile(paper_table)
        @test occursin("Generated by scripts/3_iid.jl",
            read(paper_table, String))
        @test occursin("Mean time", read(paper_table, String))
        @test occursin("Median time", read(paper_table, String))
        @test !occursin("Batch seconds", read(paper_table, String))

        corrupted_root = joinpath(root, "corrupted-render")
        corrupted = RenderStage.render_iid_contest(processed_path;
            output_root=corrupted_root, completed_at, compress=false)
        open(corrupted.table_paths[:timings], "a") do io
            println(io, "% corruption")
        end
        @test_throws ArgumentError RenderStage.load_render_manifest(
            corrupted.manifest_path)
        @test_throws ArgumentError RenderStage.render_iid_contest(
            processed_path; output_root=corrupted_root, completed_at,
            compress=false)

        invalid = JLD2.load(processed_path)
        invalid["metadata"] = merge(invalid["metadata"],
            (status=:incomplete,))
        invalid_path = joinpath(root, "invalid-processed.jld2")
        render_write_jld2(invalid_path, invalid)
        rejected_root = joinpath(root, "must-not-exist")
        @test_throws ArgumentError RenderStage.render_iid_contest(
            invalid_path; output_root=rejected_root, completed_at,
            compress=false)
        @test !ispath(rejected_root)
    end
end
