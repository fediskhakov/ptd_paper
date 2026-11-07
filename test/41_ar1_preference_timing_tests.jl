module AR1PreferenceTimingTests
using Test
using JLD2
using LinearAlgebra
using Printf
using SHA
using Statistics
import Ptd
include(joinpath(@__DIR__, "..", "scripts", "4_ar1.jl"))

function reauthenticate(artifact)
    payload = (; (name => getproperty(artifact, name) for name in keys(artifact)
        if name !== :authentication_fingerprint)...)
    return merge(payload, (; authentication_fingerprint=Ptd.stable_fingerprint(payload)))
end

function saved(path, artifact)
    JLD2.jldsave(path; artifact)
    return path
end

@testset "AR(1) timing compares five preferences on one common path" begin
    parsed = ar1_timing_run_arguments(["--timing-only", "accuracy.jld2",
        "--timing-output", "timing.jld2", "--resume", "--smoke"])
    @test parsed == (["--timing-only", "accuracy.jld2"], abspath("timing.jld2"), true, true)
    @test ar1_timing_run_arguments(["--timing-only", "accuracy.jld2"]) ==
        (["--timing-only", "accuracy.jld2"], nothing, false, false)
    for args in (["--timing-output"], ["--timing-output", "--smoke"],
        ["--timing-output", "one", "--timing-output", "two"],
        ["--resume", "--resume"], ["--smoke", "--smoke"])
        @test_throws ErrorException ar1_timing_run_arguments(args)
    end
    for args in (["--timing-only"], ["--timing-only", "--smoke"],
        ["--timing-only", "accuracy", "--paths-artifact", "paths"],
        ["--timing-only", "accuracy", "--timing-artifact", "timing"],
        ["--timing-output", "timing"], ["--resume"], ["--smoke"])
        @test_throws ErrorException main(args)
    end
    accuracy_path = joinpath(pkgdir(Ptd), "data", "ar1-contest-welfare.jld2")
    accuracy = Ptd.AR1.load_ar1_contest_welfare(accuracy_path)
    configuration = Ptd.AR1._ar1_preference_timing_configuration(accuracy)
    @test configuration.beta == 0.98
    @test configuration.gammas == (1.0, 1.5, 2.0, 2.5, 3.0)
    @test configuration.initial_cash == 16.0
    @test configuration.n_agents == 5
    @test configuration.warmup_gammas == (1.0,)
    @test configuration.warmups_per_method_per_measured_session == 1
    historical_configuration = Ptd.AR1._ar1_preference_timing_configuration(accuracy;
        warmup_gammas=(1.0, 1.5))
    @test historical_configuration.warmups_per_method_per_measured_session == 2
    @test_throws ArgumentError Ptd.AR1._ar1_preference_timing_configuration(accuracy;
        warmup_gammas=(2.0,))
    @test configuration.methods == (:ptd_sceq, :ptd_osa, :g2egm)
    @test size(configuration.innovations) == (1, 49)
    @test configuration.shock_fingerprint == Ptd.stable_fingerprint(configuration.innovations)
    @test Ptd.AR1._ar1_preference_timing_configuration(accuracy) == configuration
    order = Ptd.AR1._ar1_preference_timing_order(configuration)
    @test length(order) == 15
    @test Set((row.gamma, row.method) for row in order) ==
        Set((gamma, method) for gamma in configuration.gammas for method in configuration.methods)
    @test getproperty.(order[1:6], :method) ==
        [:ptd_sceq, :ptd_osa, :g2egm, :ptd_osa, :g2egm, :ptd_sceq]

    if Threads.nthreads() == 1
        previous = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            mktempdir() do directory
                timing_path = joinpath(directory, "preferences.jld2")
                accuracy_sha = bytes2hex(sha256(read(accuracy_path)))
                artifact = main(["--timing-only", accuracy_path,
                    "--timing-output", timing_path, "--smoke"])
                summary = Ptd.AR1.load_ar1_preference_timing_summary(timing_path, accuracy_path;
                    require_production=false, require_current_source=true)
                @test bytes2hex(sha256(read(accuracy_path))) == accuracy_sha
                @test summary.artifact == artifact
                @test artifact.schema == Ptd.AR1.AR1_PREFERENCE_TIMING_SCHEMA
                @test length(artifact.measurements) == 15
                @test artifact.configuration.gammas == configuration.gammas
                @test artifact.configuration.warmup_gammas == (1.0,)
                @test artifact.configuration.warmups_per_method_per_measured_session == 1
                @test all(sample -> sample.beta == 0.98 && sample.initial_cash == 16.0 &&
                    sample.shock_fingerprint == artifact.configuration.shock_fingerprint,
                    artifact.measurements)
                @test Set((sample.gamma, sample.method) for sample in artifact.measurements) ==
                    Set((row.gamma, row.method) for row in order)
                @test all(sample -> sample.total_seconds == sample.construction_seconds +
                    sample.evaluation_seconds && isfinite(sample.utility), artifact.measurements)
                for row in artifact.rows
                    samples = filter(sample -> sample.method === row.method, artifact.measurements)
                    @test row.n_agents == length(samples) == 5
                    @test row.mean_total_seconds == mean(sample.total_seconds for sample in samples)
                    @test row.median_total_seconds == median(getproperty.(samples, :total_seconds))
                    @test summary.rows[(:ar1, row.method)] == row
                end
                @test_throws ArgumentError Ptd.AR1.load_ar1_preference_timing_summary(timing_path, accuracy_path)
                @test_throws ArgumentError run_ar1_preference_timings(accuracy_path;
                    output_path=timing_path, smoke=true, require_production=false)
                @test_throws ArgumentError run_ar1_preference_timings(accuracy_path;
                    output_path=accuracy_path, smoke=true, require_production=false)

                # Skewed durations distinguish the mean and median across agents.
                # Each component has a different ordering, so their medians must
                # not be added to obtain the median of each full solve and path.
                construction = (1.0, 4.0, 100.0, 2.0, 7.0)
                evaluation = (100.0, 4.0, 1.0, 3.0, 2.0)
                samples = [merge(sample, (;
                    construction_seconds=construction[sample.agent_index],
                    evaluation_seconds=evaluation[sample.agent_index],
                    total_seconds=construction[sample.agent_index] + evaluation[sample.agent_index]))
                    for sample in artifact.measurements]
                synthetic = reauthenticate(merge(artifact, (; measurements=samples,
                    rows=Ptd.AR1._ar1_preference_timing_rows(samples, artifact.configuration))))
                synthetic_path = saved(joinpath(directory, "synthetic.jld2"), synthetic)
                synthetic_summary = Ptd.AR1.load_ar1_preference_timing_summary(
                    synthetic_path, accuracy_path; require_production=false)
                @test all(row -> row.mean_total_seconds == 44.8 && row.median_total_seconds == 9.0,
                    values(synthetic_summary.rows))
                @test median(collect(construction)) + median(collect(evaluation)) != 9.0
                output = joinpath(directory, "table.tex")
                @test_throws ArgumentError Ptd.AR1.render_ar1_paper_table(accuracy_path, "unused";
                    output_path=output, timing_artifact_path=synthetic_path)
                Ptd.AR1.render_ar1_paper_table(accuracy_path, "unused";
                    output_path=output, timing_artifact_path=synthetic_path,
                    require_production_timing=false)
                tex = read(output, String)
                @test occursin("Mean time", tex) && occursin("Median time", tex)
                @test occursin("44.800 & 9.000", tex)
                @test occursin("5 agents", tex)
                @test !occursin("measured repetitions", tex)
                @test occursin(synthetic.authentication_fingerprint, tex)
                @test occursin(synthetic.source_fingerprint, tex)

                validator = changed -> Ptd.AR1._validate_ar1_preference_timing_artifact(
                    changed, accuracy; require_production=false)
                # Old filenames remain valid as authenticated provenance;
                # strict current-source checks still require script 4's inventory.
                script = only(filter(file -> file.path == "scripts/4_ar1.jl",
                    artifact.source_files))
                for old_path in ("scripts/11_ar1_preference_timing.jl",
                    "scripts/ar1_preference_timing.jl")
                    old_script = merge(script, (; path=old_path))
                    old_sources = [file == script ? old_script : file for file in artifact.source_files]
                    old_artifact = reauthenticate(merge(artifact, (; source_files=old_sources,
                        source_fingerprint=Ptd.stable_fingerprint(old_sources))))
                    @test isnothing(validator(old_artifact))
                    @test_throws ArgumentError Ptd.AR1._validate_ar1_preference_timing_artifact(
                        old_artifact, accuracy; require_production=false, require_current_source=true)
                    for invalid_sources in (
                        filter(file -> file != script, artifact.source_files),
                        vcat(artifact.source_files, [old_script]))
                        @test_throws ArgumentError validator(reauthenticate(merge(artifact,
                            (; source_files=invalid_sources,
                                source_fingerprint=Ptd.stable_fingerprint(invalid_sources)))))
                    end
                end
                # Synthetic historical-format fixture; production records stay unchanged.
                historical = reauthenticate(merge(artifact, (;
                    configuration=Ptd.AR1._ar1_preference_timing_configuration(accuracy;
                        smoke=true, require_production=false, warmup_gammas=(1.0, 1.5)))))
                historical_path = saved(joinpath(directory, "historical.jld2"), historical)
                @test Ptd.AR1.load_ar1_preference_timing_summary(historical_path, accuracy_path;
                    require_production=false).configuration.warmups_per_method_per_measured_session == 2
                @test_throws ArgumentError validator(reauthenticate(merge(artifact, (;
                    configuration=merge(artifact.configuration,
                        (; warmups_per_method_per_measured_session=2))))))
                changed_rows = copy(artifact.rows)
                changed_rows[1] = merge(changed_rows[1],
                    (; mean_total_seconds=changed_rows[1].mean_total_seconds + 1.0))
                @test_throws ArgumentError validator(reauthenticate(merge(artifact, (; rows=changed_rows))))
                @test_throws ArgumentError validator(reauthenticate(merge(artifact,
                    (; measurements=artifact.measurements[1:end-1]))))
                changed_samples = copy(artifact.measurements)
                changed_samples[2] = changed_samples[1]
                @test_throws ArgumentError validator(reauthenticate(merge(artifact,
                    (; measurements=changed_samples))))
                for change in ((beta=0.97,), (gammas=(1.0, 1.5, 2.0, 2.5, 4.0),),
                    (initial_cash=17.0,), (julia_threads=24,), (blas_threads=2,))
                    @test_throws ArgumentError validator(reauthenticate(merge(artifact,
                        (; configuration=merge(artifact.configuration, change)))))
                end
                changed_innovations = copy(artifact.configuration.innovations)
                changed_innovations[1] += 1.0
                @test_throws ArgumentError validator(reauthenticate(merge(artifact,
                    (; configuration=merge(artifact.configuration, (; innovations=changed_innovations))))))
                @test_throws ArgumentError validator(merge(artifact, (; authentication_fingerprint="changed")))
                @test_throws ArgumentError validator(reauthenticate(merge(artifact,
                    (; schema="ptd-single-path-timing-v2"))))

                # Resume preserves completed agent/method samples and warms up
                # disposable policies before timing the missing fresh solves.
                resume_path = joinpath(directory, "resumed.jld2")
                checkpoint_path = resume_path * ".partial.jld2"
                _ar1_preference_timing_checkpoint(checkpoint_path, artifact.configuration,
                    artifact.accuracy, artifact.source_files, artifact.sessions,
                    artifact.measurements[1:4])
                @test_throws ArgumentError run_ar1_preference_timings(accuracy_path;
                    output_path=resume_path, smoke=true, require_production=false)
                checkpoint = JLD2.load(checkpoint_path, "artifact")
                saved(checkpoint_path, reauthenticate(merge(checkpoint,
                    (; configuration=merge(checkpoint.configuration, (; beta=0.97))))))
                @test_throws ArgumentError run_ar1_preference_timings(accuracy_path;
                    output_path=resume_path, smoke=true, resume=true, require_production=false)
                saved(checkpoint_path, checkpoint)
                resumed = run_ar1_preference_timings(accuracy_path; output_path=resume_path,
                    smoke=true, resume=true, require_production=false)
                @test resumed.measurements[1:4] == artifact.measurements[1:4]
                @test length(resumed.sessions) == 2
                @test all(sample -> sample.session_index == 2, resumed.measurements[5:end])
                @test Ptd.AR1.load_ar1_preference_timing_summary(resume_path, accuracy_path;
                    require_production=false).artifact == resumed
            end
        finally
            BLAS.set_num_threads(previous)
        end
    else
        @test_throws ArgumentError run_ar1_preference_timings("unused"; output_path="unused")
    end
end
end
