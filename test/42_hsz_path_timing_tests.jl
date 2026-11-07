module HSZPathTimingTests
using Test
using JLD2
using LinearAlgebra
using SHA
using Statistics
import Ptd

const H = Ptd.HSZ

function reauthenticate(artifact)
    payload = (; (name => getproperty(artifact, name) for name in keys(artifact)
        if name !== :authentication_fingerprint)...)
    return merge(payload, (; authentication_fingerprint=Ptd.stable_fingerprint(payload)))
end

function saved(path, artifact)
    JLD2.jldsave(path; artifact)
    return path
end

function tiny_accuracy(directory)
    config = H.HSZExampleConfig(horizon=3,
        initial_cash_support=[16.0], n_paths=4, paths_per_initial_cash=4,
        asset_points=9, common_cash_points=9,
        post_earnings_points=5, post_medical_points=5,
        common_earnings_points=5, common_medical_points=5)
    source_path = joinpath(directory, "example.jld2")
    accuracy_path = joinpath(directory, "utility.jld2")
    source = H.run_hsz_example(config; output_path=source_path).artifact
    accuracy = H.run_hsz_utility_comparison(source_path;
        config, output_path=accuracy_path)
    return (; source, source_path, accuracy, accuracy_path)
end

@testset "HSZ retained-policy timing measures individual paired paths" begin
    if Threads.nthreads() == 1
        previous_blas_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            mktempdir() do directory
                fixture = tiny_accuracy(directory)
                accuracy, accuracy_path = fixture.accuracy, fixture.accuracy_path
                original_accuracy_sha = bytes2hex(sha256(read(accuracy_path)))
                output_path = joinpath(directory, "timing.jld2")
                built = Dict(method => Any[] for method in H.HSZ_EXAMPLE_METHODS)
                evaluated = Dict(method => Any[] for method in H.HSZ_EXAMPLE_METHODS)
                shocks_seen = Dict(method => Any[] for method in H.HSZ_EXAMPLE_METHODS)
                function counted_builder(method, model, process; kwargs...)
                    policy = Ptd.build_hsz_contest_policy(method, model, process; kwargs...)
                    push!(built[method], policy)
                    return policy
                end
                function counted_evaluator(policy, method, model, process, initial, shocks)
                    push!(evaluated[method], policy)
                    push!(shocks_seen[method], copy(shocks))
                    return Ptd.evaluate_policy_paths(policy, method, model, process, initial, shocks)
                end
                artifact = H.run_hsz_path_timings(accuracy_path; output_path,
                    require_production=false, policy_builder=counted_builder,
                    evaluator=counted_evaluator)
                summary = H.load_hsz_path_timing_summary(output_path, accuracy_path;
                    require_production=false, require_current_source=true)
                @test summary.artifact == artifact
                @test bytes2hex(sha256(read(accuracy_path))) == original_accuracy_sha
                @test artifact.schema == H.HSZ_PATH_TIMING_SCHEMA
                @test artifact.configuration.initial_cash == 16.0
                @test artifact.configuration.beta == 0.98
                @test artifact.configuration.gamma == 1.0
                @test artifact.configuration.n_paths == 4
                @test artifact.configuration.innovations == accuracy.innovations
                @test artifact.configuration.julia_threads == artifact.configuration.blas_threads == 1
                @test :result_validation in artifact.configuration.outside_timers
                @test :result_validation ∉ artifact.configuration.inside_timers
                @test artifact.configuration.implementation === :injected
                @test length(artifact.measurements) == length(artifact.rows) == 3
                for method in H.HSZ_EXAMPLE_METHODS
                    # The warmup object is discarded; every measured path uses
                    # the same fresh object, including its retained PTD cache.
                    @test length(built[method]) == 2
                    @test built[method][1] !== built[method][2]
                    @test length(evaluated[method]) == 5
                    @test evaluated[method][1] === built[method][1]
                    @test all(policy -> policy === built[method][2], evaluated[method][2:end])
                    @test shocks_seen[method][1] == accuracy.innovations[1:1, :, :]
                    @test all(shocks_seen[method][path + 1] == accuracy.innovations[path:path, :, :]
                        for path in 1:4)
                    sample = only(filter(sample -> sample.method === method, artifact.measurements))
                    @test length(sample.evaluation_seconds) == 4
                    @test sample.utility ≈ accuracy.paths[method].utility atol=1e-9 rtol=1e-11
                    @test sample.floor_hits == accuracy.paths[method].floor_hits
                    @test sample.total_transfers ≈ accuracy.paths[method].total_transfers atol=1e-9 rtol=1e-11
                    @test all(t -> isfinite(t) && t >= 0, sample.evaluation_seconds)
                    @test sample.total_seconds == sample.construction_seconds + sum(sample.evaluation_seconds)
                    row = summary.rows[(:hsz, method)]
                    @test row.construction_seconds == sample.construction_seconds
                    @test row.mean_evaluation_seconds == mean(sample.evaluation_seconds)
                    @test row.median_evaluation_seconds == median(sample.evaluation_seconds)
                end
                @test_throws ArgumentError H.load_hsz_path_timing_summary(output_path, accuracy_path)
                @test_throws ArgumentError H.run_hsz_path_timings(accuracy_path;
                    output_path, require_production=false)
                @test_throws ArgumentError H.run_hsz_path_timings(accuracy_path;
                    output_path=accuracy_path, require_production=false)

                # A deliberately skewed sample distinguishes the actual path
                # median from a batch average. Each summary adds the same full
                # construction time, and all three columns remain in seconds.
                durations = [0.001, 0.003, 0.005, 0.031]
                synthetic_samples = [merge(sample, (;
                    construction_seconds=2.0, evaluation_seconds=durations,
                    total_seconds=2.0 + sum(durations))) for sample in artifact.measurements]
                synthetic = reauthenticate(merge(artifact, (; measurements=synthetic_samples,
                    rows=H._hsz_path_timing_rows(synthetic_samples, artifact.configuration))))
                synthetic_path = saved(joinpath(directory, "synthetic.jld2"), synthetic)
                synthetic_summary = H.load_hsz_path_timing_summary(synthetic_path, accuracy_path;
                    require_production=false)
                @test all(row -> row.mean_evaluation_seconds == 0.010 &&
                    row.median_evaluation_seconds == 0.004, values(synthetic_summary.rows))
                table_path = joinpath(directory, "table.tex")
                H._write_hsz_path_timing_table(table_path, fixture.source, synthetic_summary)
                tex = read(table_path, String)
                @test occursin(raw"\makecell{Construction\\(s)}", tex)
                @test occursin(raw"\makecell{Mean time\\(s)}", tex)
                @test occursin(raw"\makecell{Median time\\(s)}", tex)
                @test count("2.000 & 2.010 & 2.004", tex) == 3
                @test occursin(synthetic.authentication_fingerprint, tex)
                @test occursin(synthetic.source_fingerprint, tex)

                validate = changed -> H._validate_hsz_path_timing_artifact(changed, accuracy;
                    require_production=false)
                # Legacy script names remain valid only as recorded provenance.
                script = only(filter(file -> file.path == "scripts/5_hsz.jl",
                    artifact.source_files))
                for old_path in ("scripts/hsz_path_timing.jl", "scripts/12_hsz_path_timing.jl")
                    legacy_script = merge(script, (; path=old_path))
                    legacy_sources = [file == script ? legacy_script : file for file in artifact.source_files]
                    legacy = reauthenticate(merge(artifact, (; source_files=legacy_sources,
                        source_fingerprint=Ptd.stable_fingerprint(legacy_sources))))
                    @test isnothing(validate(legacy))
                    @test_throws ArgumentError H._validate_hsz_path_timing_artifact(
                        legacy, accuracy; require_production=false, require_current_source=true)
                    for invalid_sources in (
                        filter(file -> file != script, artifact.source_files),
                        vcat(artifact.source_files, [legacy_script]))
                        @test_throws ArgumentError validate(reauthenticate(merge(artifact,
                            (; source_files=invalid_sources,
                                source_fingerprint=Ptd.stable_fingerprint(invalid_sources)))))
                    end
                end
                @test_throws ArgumentError validate(merge(artifact, (; authentication_fingerprint="tampered")))
                for change in ((beta=0.97,), (initial_cash=21.25,), (n_paths=3,),
                    (julia_threads=24,), (blas_threads=2,), (reuse=:fresh_policy_each_path,))
                    @test_throws ArgumentError validate(reauthenticate(merge(artifact,
                        (; configuration=merge(artifact.configuration, change)))))
                end
                changed_shocks = copy(artifact.configuration.innovations)
                changed_shocks[1] += 1.0
                @test_throws ArgumentError validate(reauthenticate(merge(artifact,
                    (; configuration=merge(artifact.configuration, (; innovations=changed_shocks))))))
                changed_rows = copy(artifact.rows)
                changed_rows[1] = merge(changed_rows[1], (; median_evaluation_seconds=123.0))
                @test_throws ArgumentError validate(reauthenticate(merge(artifact, (; rows=changed_rows))))
                @test_throws ArgumentError validate(reauthenticate(merge(artifact,
                    (; measurements=artifact.measurements[1:end-1]))))
                for field in (:utility, :floor_hits, :total_transfers)
                    samples = copy(artifact.measurements)
                    changed_values = copy(getproperty(samples[1], field))
                    changed_values[1] += 1
                    samples[1] = merge(samples[1], NamedTuple{(field,)}((changed_values,)))
                    @test_throws ArgumentError validate(reauthenticate(merge(artifact,
                        (; measurements=samples))))
                end
                changed_sessions = [merge(only(artifact.sessions), (; blas_threads=2))]
                @test_throws ArgumentError validate(reauthenticate(merge(artifact, (; sessions=changed_sessions))))
                changed_sources = copy(artifact.source_files)
                changed_sources[1] = merge(changed_sources[1], (; sha256=repeat("0", 64)))
                historical = reauthenticate(merge(artifact, (; source_files=changed_sources,
                    source_fingerprint=Ptd.stable_fingerprint(changed_sources))))
                @test isnothing(validate(historical))
                @test_throws ArgumentError H._validate_hsz_path_timing_artifact(historical, accuracy;
                    require_production=false, require_current_source=true)
                wrong_parent = saved(joinpath(directory, "wrong-parent.jld2"),
                    reauthenticate(merge(artifact, (; accuracy=merge(artifact.accuracy,
                        (; file_sha256=repeat("0", 64)))))))
                @test_throws ArgumentError H.load_hsz_path_timing_summary(wrong_parent, accuracy_path;
                    require_production=false)

                # Resume only at a completed-method boundary. Preserve the
                # first method's exact measurements and warm fresh policies for
                # each unfinished method; reject edited checkpoint inputs.
                resumed_path = joinpath(directory, "resumed.jld2")
                checkpoint_path = resumed_path * ".partial.jld2"
                checkpoint = H._authenticate_hsz_path_timing((;
                    schema=H.HSZ_PATH_TIMING_CHECKPOINT_SCHEMA,
                    configuration=artifact.configuration, accuracy=artifact.accuracy,
                    source_files=artifact.source_files, sessions=artifact.sessions,
                    measurements=artifact.measurements[1:1]))
                saved(checkpoint_path, checkpoint)
                @test_throws ArgumentError H.run_hsz_path_timings(accuracy_path;
                    output_path=resumed_path, require_production=false,
                    policy_builder=counted_builder, evaluator=counted_evaluator)
                saved(checkpoint_path, reauthenticate(merge(checkpoint,
                    (; configuration=merge(checkpoint.configuration, (; n_paths=3))))))
                @test_throws ArgumentError H.run_hsz_path_timings(accuracy_path;
                    output_path=resumed_path, resume=true, require_production=false,
                    policy_builder=counted_builder, evaluator=counted_evaluator)
                saved(checkpoint_path, checkpoint)
                foreach(empty!, values(built))
                foreach(empty!, values(evaluated))
                foreach(empty!, values(shocks_seen))
                resumed = H.run_hsz_path_timings(accuracy_path; output_path=resumed_path,
                    resume=true, require_production=false,
                    policy_builder=counted_builder, evaluator=counted_evaluator)
                @test resumed.measurements[1] == artifact.measurements[1]
                @test length(resumed.sessions) == 2
                @test all(sample -> sample.session_index == 2, resumed.measurements[2:end])
                @test isempty(built[:ptd_sceq]) && isempty(evaluated[:ptd_sceq])
                @test all(length(built[method]) == 2 && length(evaluated[method]) == 5
                    for method in (:ptd_osa, :g2egm))
                @test H.load_hsz_path_timing_summary(resumed_path, accuracy_path;
                    require_production=false, require_current_source=true).artifact == resumed

                # Reference checks happen after the timed path calls. A replay
                # with altered utility must never produce a completed method
                # checkpoint or a timing artifact, even when actions are valid.
                altered_calls = Ref(0)
                function altered_evaluator(policy, method, model, process, initial, shocks)
                    altered_calls[] += 1
                    result = Ptd.evaluate_policy_paths(policy, method, model, process, initial, shocks)
                    return (; method=result.method, utility=result.utility .+ 1.0,
                        floor_hits=result.floor_hits, total_transfers=result.total_transfers)
                end
                invalid_path = joinpath(directory, "invalid-replay.jld2")
                @test_throws ArgumentError H.run_hsz_path_timings(accuracy_path;
                    output_path=invalid_path, require_production=false,
                    evaluator=altered_evaluator)
                @test altered_calls[] == 5  # One discarded warmup, four measured paths.
                @test !isfile(invalid_path)
                @test isempty(JLD2.load(invalid_path * ".partial.jld2", "artifact").measurements)

                BLAS.set_num_threads(2)
                @test_throws ArgumentError H.run_hsz_path_timings(accuracy_path;
                    output_path=joinpath(directory, "multithreaded.jld2"), require_production=false)
                BLAS.set_num_threads(1)
            end
        finally
            BLAS.set_num_threads(previous_blas_threads)
        end
    else
        @test_throws ArgumentError H.run_hsz_path_timings("unused"; output_path="unused")
    end
end

end
