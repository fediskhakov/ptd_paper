module IIDRandomTimingTests
using Test
include(joinpath(@__DIR__, "..", "scripts", "3_iid.jl"))

function reauthenticate(artifact)
    payload = (; (name => getproperty(artifact, name) for name in keys(artifact)
        if name !== :authentication_fingerprint)...)
    return _iid_random_timing_authenticate(payload)
end

function tiny_accuracy(path)
    config = Ptd.IID.IIDContestRunConfig(n_contests=2, n_paths=3, horizon=3,
        bounds=Ptd.ParameterBounds((0.94, 0.98), (0.8, 1.2), (4.0, 8.0)),
        dcegm_asset_points=31, dcegm_asset_maximum=30.0)
    context = Ptd.IID._capture_run_context(config, pkgdir(Ptd))
    hooks = Ptd.IID.RunnerHooks()
    welfare = Ptd.IID._run_welfare_unlocked(config, hooks;
        workers=1, scheduler=Ptd.IID.serial_scheduler, context)
    timing = Ptd.IID._run_timing_unlocked(config, hooks;
        design=welfare.design, plain_design=welfare.outer_design,
        sampler_audit=welfare.sampler_audit, workers=1,
        scheduler=Ptd.IID.serial_scheduler, context)
    return Ptd.IID.publish_iid_contest(config, welfare, timing; output_path=path, context).artifact
end

@testset "IID timings preserve agents, shocks, utility, and raw evidence" begin
    args = iid_timing_arguments(["--timing-only", "accuracy.jld2", "--timing-output", "timing.jld2", "--resume"])
    @test args[1] == ["--timing-only", "accuracy.jld2"]
    @test isnothing(args[2]) && args[3] == abspath("timing.jld2") && args[4]
    @test_throws ErrorException iid_timing_arguments(["--timing-output"])
    @test_throws ErrorException iid_timing_arguments(["--resume", "--resume"])
    @test_throws ErrorException main(["--timing-only"])
    @test_throws ErrorException main(["--timing-output", "timing.jld2"])
    @test_throws ErrorException main(["--resume"])
    sources = _iid_random_timing_sources()
    @test sources == Ptd.IID._computational_input_tree(pkgdir(Ptd)).files
    @test length(unique(file.path for file in sources)) == length(sources)
    @test count(file -> file.path == "scripts/3_iid.jl", sources) == 1
    @test !any(file -> file.path in ("scripts/10_iid_random_timing.jl",
        "scripts/iid_random_timing.jl"), sources)
    if Threads.nthreads() == 1
        previous = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            mktempdir() do directory
                accuracy_path = joinpath(directory, "accuracy.jld2")
                timing_path = joinpath(directory, "timing.jld2")
                accuracy = tiny_accuracy(accuracy_path)
                original = read(accuracy_path)
                @test_throws ArgumentError run_iid_random_timings(accuracy_path; output_path=timing_path)
                artifact = run_iid_random_timings(accuracy_path;
                    output_path=timing_path, require_production=false)
                summary = Ptd.IID.load_iid_random_timing_summary(timing_path, accuracy_path;
                    require_production=false)
                @test read(accuracy_path) == original
                @test artifact.configuration.n_agents == 2
                @test length(artifact.measurements) == 12
                @test length(artifact.configuration.shock_records) == 4
                @test artifact.configuration.specs == accuracy["outer_design"]
                @test getproperty.(artifact.measurements[1:6], :method) ==
                    [:ptd_sceq, :ptd_osa, :dcegm, :ptd_osa, :dcegm, :ptd_sceq]
                for row in artifact.rows
                    samples = filter(sample -> sample.panel === row.panel && sample.method === row.method,
                        artifact.measurements)
                    @test row.n_agents == length(samples) == 2
                    @test row.mean_seconds == mean(sample.total_seconds for sample in samples)
                    @test row.median_seconds == median(getproperty.(samples, :total_seconds))
                    @test summary.rows[(row.panel, row.method)] == row
                end
                table = Ptd.IID.render_iid_paper_table(accuracy_path;
                    output_path=joinpath(directory, "summary.tex"), timing_artifact_path=timing_path,
                    require_production_timing=false)
                tex = read(table, String)
                @test occursin("Mean time", tex) && occursin("Median time", tex)
                @test !occursin("Batch", tex) && !occursin("not reported", tex)
                @test occursin("same 2 agents", tex)
                @test_throws ArgumentError run_iid_random_timings(accuracy_path;
                    output_path=timing_path, require_production=false)
                @test_throws ArgumentError run_iid_random_timings(accuracy_path;
                    output_path=accuracy_path, require_production=false)

                validator = changed -> Ptd.IID._validate_iid_random_timing_artifact(changed, accuracy;
                    require_production=false)
                # The old helper was recorded alongside script 3; keep that
                # authenticated provenance valid when loading archived timings.
                script = only(filter(file -> file.path == "scripts/3_iid.jl", artifact.source_files))
                for old_path in ("scripts/10_iid_random_timing.jl", "scripts/iid_random_timing.jl")
                    old_sources = sort(vcat(artifact.source_files, [merge(script, (; path=old_path))]);
                        by=file -> file.path)
                    old_artifact = reauthenticate(merge(artifact, (; source_files=old_sources,
                        source_fingerprint=Ptd.stable_fingerprint(old_sources))))
                    @test isnothing(validator(old_artifact))
                end
                changed = copy(artifact.rows)
                changed[1] = merge(changed[1], (; mean_seconds=changed[1].mean_seconds + 1.0))
                @test_throws ArgumentError validator(reauthenticate(merge(artifact, (; rows=changed))))
                changed_samples = copy(artifact.measurements)
                changed_samples[1] = merge(changed_samples[1], (; utility=changed_samples[1].utility + 1.0))
                @test_throws ArgumentError validator(reauthenticate(merge(artifact, (; measurements=changed_samples))))
                @test_throws ArgumentError validator(reauthenticate(merge(artifact,
                    (; measurements=artifact.measurements[1:end-1]))))
                changed_config = merge(artifact.configuration, (; specs=reverse(artifact.configuration.specs)))
                @test_throws ArgumentError validator(reauthenticate(merge(artifact, (; configuration=changed_config))))
                changed_records = deepcopy(artifact.configuration.shock_records)
                changed_records[1].shocks[1] += 1.0
                changed_config = merge(artifact.configuration, (; shock_records=changed_records))
                @test_throws ArgumentError validator(reauthenticate(merge(artifact, (; configuration=changed_config))))
                @test_throws ArgumentError validator(merge(artifact, (; authentication_fingerprint="changed")))

                # A partial checkpoint resumes only the missing fresh-policy samples.
                resume_path = joinpath(directory, "resumed.jld2")
                checkpoint_path = resume_path * ".partial.jld2"
                _iid_random_timing_checkpoint(checkpoint_path, artifact.configuration, artifact.accuracy,
                    artifact.source_files, artifact.sessions, artifact.measurements[1:4])
                @test_throws ArgumentError run_iid_random_timings(accuracy_path;
                    output_path=resume_path, require_production=false)
                resumed = run_iid_random_timings(accuracy_path; output_path=resume_path,
                    resume=true, require_production=false)
                @test resumed.measurements[1:4] == artifact.measurements[1:4]
                @test length(resumed.sessions) == 2
                @test all(sample -> sample.session_index == 2, resumed.measurements[5:end])
                @test Ptd.IID.load_iid_random_timing_summary(resume_path, accuracy_path;
                    require_production=false).artifact == resumed
            end
        finally
            BLAS.set_num_threads(previous)
        end
    else
        @test_throws ArgumentError run_iid_random_timings("unused"; output_path="unused")
    end
end
end
