module SinglePathSummaryTests
using Test
using JLD2
using Printf
using Statistics
import Ptd

const TimingDefinition = Ptd.FixedCalibrationTiming

module IIDDefinition
include(joinpath(@__DIR__, "..", "scripts", "3_iid.jl"))
end
module AR1Definition
include(joinpath(@__DIR__, "..", "scripts", "4_ar1.jl"))
end
module HSZDefinition
include(joinpath(@__DIR__, "..", "scripts", "5_hsz.jl"))
end

function resign(artifact)
    configuration = artifact.configuration
    provenance = merge(artifact.provenance,
        (; configuration=Ptd.configuration_metadata(configuration)))
    payload = (; (name => getproperty(artifact, name) for name in keys(artifact)
        if name ∉ (:authentication_fingerprint, :provenance))..., provenance)
    return merge(payload, (; authentication_fingerprint=Ptd.stable_fingerprint(payload)))
end

function fixture()
    cases = TimingDefinition.timing_cases()
    configurations = [merge(TimingDefinition._timing_case_configuration(case, i),
        (; repeats=3)) for (i, case) in enumerate(cases)]
    sessions = ["fixture-session"]
    raw = NamedTuple[]
    rows = NamedTuple[]
    index = 0
    for case in cases, method in case.methods
        index += 1
        offset = index / 8
        # Median construction + median evaluation differs from median total.
        # Extra archived samples must not enter the selected three-run summary.
        samples = [(construction_seconds=c + offset, evaluation_seconds=e,
            total_seconds=c + offset + e, utility=42.0, session=only(sessions))
            for (c, e) in ((1.0, 100.0), (4.0, 4.0), (100.0, 1.0), (999.0, 999.0))]
        selected = samples[1:3]
        push!(raw, (; panel=case.panel, method, samples))
        push!(rows, (; panel=case.panel, method, repeats=3, available_repeats=4,
            reused_repeats=1, construction_seconds=4.0 + offset, evaluation_seconds=4.0,
            total_seconds=101.0 + offset, minimum_total_seconds=8.0 + offset,
            maximum_total_seconds=101.0 + offset, utility=42.0, samples=selected))
    end
    configuration = (; smoke=false, repeats_override=3,
        repeats_by_panel=(deterministic=3, iid_baseline=3, iid_high_variance=3, ar1=3, hsz=3),
        selected_panels=getproperty.(cases, :panel), configurations,
        julia_threads=1, blas_threads=1, paths_per_measurement=1,
        warmup=:disposable_full_construction_and_path, warmups_per_method_per_measured_session=1,
        cache=:fresh_policy_each_repetition,
        total=:sum_of_separately_measured_construction_and_evaluation,
        summary=:median_of_each_component_and_repetition_total,
        single_repetition_summary=:single_measurement,
        sample_selection=:first_requested_repetitions_in_collection_order,
        sessions, outside_timers=(:shock_generation, :explicit_policy_alignment,
            :output_validation, :file_io, :warmup))
    input_files = Ptd._single_path_source_files()
    payload = (; schema="ptd-single-path-timing-v2", configuration, rows,
        raw_measurements=raw, input_files, input_fingerprint=Ptd.stable_fingerprint(input_files),
        provenance=Ptd.simulation_metadata(configuration, pkgdir(Ptd)))
    return resign(payload)
end

function saved(path, artifact)
    JLD2.jldsave(path; artifact)
    return path
end

@testset "authenticated single-thread summaries and paper columns" begin
    mktempdir() do directory
        artifact = fixture()
        timing_path = saved(joinpath(directory, "timing.jld2"), artifact)
        summary = Ptd.load_single_path_timing_summary(timing_path;
            required_panels=(:deterministic, :iid_baseline, :iid_high_variance, :ar1))
        @test IIDDefinition.iid_timing_arguments(["--render-only", "accuracy", "--timing-artifact", timing_path]) ==
            (["--render-only", "accuracy"], timing_path, nothing, false)
        @test AR1Definition.ar1_timing_arguments(["--render-only", "accuracy", "exhibit", "--timing-artifact", timing_path]) ==
            (["--render-only", "accuracy", "exhibit"], timing_path)
        @test isnothing(IIDDefinition.iid_timing_arguments(["--render-only", "accuracy"])[2])
        @test isnothing(last(AR1Definition.ar1_timing_arguments(["--render-only", "accuracy", "exhibit"])))
        @test isfile(HSZDefinition._hsz_options(["--render-only"]).timing_artifact_path)
        @test isnothing(HSZDefinition._hsz_options([
            "--render-only", "--example-artifact", "custom.jld2"]).timing_artifact_path)
        @test HSZDefinition._hsz_options([
            "--render-only", "--timing-artifact", timing_path]).timing_artifact_path == timing_path
        @test length(summary.rows) == 17
        @test summary.rows[(:deterministic, :ptd)].total_seconds == 101.125
        @test summary.rows[(:deterministic, :ptd)].median_total_seconds == 101.125
        @test summary.rows[(:deterministic, :ptd)].mean_total_seconds ==
            mean([101.125, 8.125, 101.125])
        @test summary.rows[(:deterministic, :ptd)].construction_seconds +
            summary.rows[(:deterministic, :ptd)].evaluation_seconds == 8.125
        @test summary.artifact_fingerprint == artifact.authentication_fingerprint
        @test summary.source_fingerprint == artifact.input_fingerprint
        @test length(summary.artifact_sha256) == 64
        @test Ptd.load_single_path_timing_summary(timing_path;
            require_current_source=true).source_files == artifact.input_files

        # A copy without git history records no commit hash, and its timing
        # provenance still counts as complete.
        unversioned_source = mktempdir() do outside
            withenv("GIT_CEILING_DIRECTORIES" => dirname(outside)) do
                Ptd.git_metadata(outside)
            end
        end
        @test unversioned_source.commit == Ptd.UNVERSIONED_COMMIT
        unversioned_provenance = merge(artifact.provenance,
            (; source=unversioned_source))
        unversioned = saved(joinpath(directory, "unversioned-timing.jld2"),
            resign(merge(artifact, (; provenance=unversioned_provenance))))
        @test length(Ptd.load_single_path_timing_summary(unversioned).rows) ==
            17

        # Historical artifacts remain authenticated by their original inventory;
        # a current-source requirement still rejects the retired script path.
        historical_files = [file.path == "src/simulation/common/fixed_calibration_timing.jl" ?
            merge(file, (; path="scripts/6_timing.jl")) : file for file in artifact.input_files]
        historical = saved(joinpath(directory, "historical-timing.jld2"),
            resign(merge(artifact, (; input_files=historical_files,
                input_fingerprint=Ptd.stable_fingerprint(historical_files)))))
        @test Ptd.load_single_path_timing_summary(historical).source_files == historical_files
        @test_throws ArgumentError Ptd.load_single_path_timing_summary(historical;
            require_current_source=true)
        missing_engine = filter(file -> file.path != "scripts/6_timing.jl", historical_files)
        incomplete = saved(joinpath(directory, "incomplete-timing.jld2"),
            resign(merge(artifact, (; input_files=missing_engine,
                input_fingerprint=Ptd.stable_fingerprint(missing_engine)))))
        @test_throws ArgumentError Ptd.load_single_path_timing_summary(incomplete)
        @test_throws ArgumentError Ptd.load_single_path_timing_summary(timing_path;
            required_panels=(:unavailable,))

        damaged = saved(joinpath(directory, "damaged.jld2"),
            merge(artifact, (; input_fingerprint="altered")))
        @test_throws ArgumentError Ptd.load_single_path_timing_summary(damaged)
        rows = copy(artifact.rows)
        rows[1] = merge(rows[1], (; total_seconds=8.125))
        saved(damaged, resign(merge(artifact, (; rows))))
        @test_throws ArgumentError Ptd.load_single_path_timing_summary(damaged)
        rows = copy(artifact.rows)
        rows[1] = merge(rows[1], (; repeats=4))
        saved(damaged, resign(merge(artifact, (; rows))))
        @test_throws ArgumentError Ptd.load_single_path_timing_summary(damaged)
        rows = copy(artifact.rows)
        rows[1] = merge(rows[1], (; samples=reverse(rows[1].samples)))
        saved(damaged, resign(merge(artifact, (; rows))))
        @test_throws ArgumentError Ptd.load_single_path_timing_summary(damaged)
        for changed in ((julia_threads=24,), (blas_threads=2,), (paths_per_measurement=100,),
            (smoke=true,), (outside_timers=(:file_io,),))
            saved(damaged, resign(merge(artifact,
                (; configuration=merge(artifact.configuration, changed)))))
            @test_throws ArgumentError Ptd.load_single_path_timing_summary(damaged)
        end
        configs = copy(artifact.configuration.configurations)
        configs[1] = merge(configs[1], (; gamma=2.0))
        saved(damaged, resign(merge(artifact,
            (; configuration=merge(artifact.configuration, (; configurations=configs))))))
        @test_throws ArgumentError Ptd.load_single_path_timing_summary(damaged)
        files = copy(artifact.input_files)
        files[1] = merge(files[1], (; sha256=repeat("0", 64)))
        saved(damaged, resign(merge(artifact,
            (; input_files=files, input_fingerprint=Ptd.stable_fingerprint(files)))))
        @test Ptd.load_single_path_timing_summary(damaged).source_files == files
        @test_throws ArgumentError Ptd.load_single_path_timing_summary(damaged;
            require_current_source=true)

        # IID timings must represent the saved heterogeneous accuracy agents,
        # so the old fixed-agent summary is no longer an acceptable substitute.
        @test_throws ArgumentError Ptd.IID.render_iid_paper_table(
            joinpath(pkgdir(Ptd), "data", "iid-contest.jld2");
            output_path=joinpath(directory, "invalid-iid.tex"),
            timing_artifact_path=timing_path)

        # Archived AR(1) accuracy shares accompany independent fixed-calibration
        # means and medians. Batch timings never enter the paper table.
        for (family, contest_name, methods, panels) in (
            (:ar1, "ar1-contest-welfare.jld2", (:ptd_sceq, :ptd_osa, :g2egm), (:ar1,)),)
            contest_path = joinpath(pkgdir(Ptd), "data", contest_name)
            output = joinpath(directory, "$(family).tex")
            if family === :iid
                Ptd.IID.render_iid_paper_table(contest_path; output_path=output,
                    timing_artifact_path=timing_path)
            else
                Ptd.AR1.render_ar1_paper_table(contest_path, "unused-exhibit";
                    output_path=output, timing_artifact_path=timing_path)
            end
            tex = read(output, String)
            @test !occursin("Batch seconds", tex)
            @test occursin("Mean time", tex)
            @test occursin("Median time", tex)
            @test occursin("accuracy varies preferences and initial cash", tex)
            @test occursin("3 measured repetitions", tex)
            @test occursin(summary.artifact_fingerprint, tex)
            @test occursin(summary.source_fingerprint, tex)
            for panel in panels, method in methods
                @test occursin(@sprintf("%.3f", summary.rows[(panel, method)].mean_total_seconds), tex)
                @test occursin(@sprintf("%.3f", summary.rows[(panel, method)].median_total_seconds), tex)
            end
            contest = family === :iid ? Ptd.IID.load_iid_contest(contest_path) :
                Ptd.AR1.load_ar1_contest_welfare(contest_path)
            for row in Ptd.IID._outcome_rows(contest, collect(methods), collect(panels))
                @test occursin(" & $(round(Int, 100 * row.point_winner_share)) & $(round(Int, 100 * row.resolved_win_share)) & ", tex)
            end
            if family === :iid
                Ptd.IID.render_iid_paper_table(contest_path; output_path=output)
            else
                Ptd.AR1.render_ar1_paper_table(contest_path, "unused-exhibit"; output_path=output)
            end
            missing = read(output, String)
            @test occursin("not reported", missing)
            @test !occursin("Batch seconds", missing)
        end

        hsz = Ptd.HSZ.load_hsz_example(Ptd.HSZ.default_hsz_example_artifact_path())
        hsz_table = joinpath(directory, "hsz.tex")
        Ptd.HSZ._write_timing(hsz_table, hsz; timing_artifact_path=timing_path)
        tex = read(hsz_table, String)
        @test occursin(summary.artifact_fingerprint, tex)
        @test occursin("Mean time", tex) && occursin("Median time", tex)
        @test !occursin("Construction (s)", tex)
        for method in (:ptd_sceq, :ptd_osa, :g2egm)
            @test occursin(@sprintf("%.3f", summary.rows[(:hsz, method)].mean_total_seconds), tex)
            @test occursin(@sprintf("%.3f", summary.rows[(:hsz, method)].median_total_seconds), tex)
        end
        incompatible_hsz = merge(hsz, (; scientific_configuration=merge(
            hsz.scientific_configuration, (; return_rate=0.05))))
        @test_throws ArgumentError Ptd.HSZ._write_timing(hsz_table, incompatible_hsz;
            timing_artifact_path=timing_path)

        # One measured path has one elapsed-time column; repeated timings retain
        # distinct means and medians computed from the authenticated samples.
        single_configuration = merge(artifact.configuration, (;
            repeats_by_panel=merge(artifact.configuration.repeats_by_panel, (; hsz=1)),
            configurations=[case.panel === :hsz ? merge(case, (; repeats=1)) : case
                for case in artifact.configuration.configurations]))
        single_rows = map(artifact.rows) do row
            row.panel === :hsz || return row
            sample = first(row.samples)
            merge(row, (; repeats=1, samples=[sample],
                construction_seconds=sample.construction_seconds,
                evaluation_seconds=sample.evaluation_seconds,
                total_seconds=sample.total_seconds,
                minimum_total_seconds=sample.total_seconds,
                maximum_total_seconds=sample.total_seconds))
        end
        single_path = saved(joinpath(directory, "single.jld2"), resign(merge(
            artifact, (; configuration=single_configuration, rows=single_rows))))
        Ptd.HSZ._write_timing(hsz_table, hsz; timing_artifact_path=single_path)
        single_tex = read(hsz_table, String)
        @test occursin(raw"\begin{tabular}{lr}", single_tex)
        @test occursin(raw"\makecell{Time\\(s)}", single_tex)
        @test !occursin("Mean time", single_tex) && !occursin("Median time", single_tex)
        for method in (:ptd_sceq, :ptd_osa, :g2egm)
            row = only(filter(row -> row.panel === :hsz && row.method === method, single_rows))
            @test occursin(@sprintf("%.3f", row.total_seconds), single_tex)
        end
    end
end
end
