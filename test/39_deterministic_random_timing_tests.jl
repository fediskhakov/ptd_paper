module DeterministicRandomTimingTests
using Test
include(joinpath(@__DIR__, "..", "scripts", "2_deterministic.jl"))

function authenticated(payload)
    contents = (; (name => getproperty(payload, name) for name in keys(payload)
        if name !== :authentication_fingerprint)...)
    return merge(contents, (; authentication_fingerprint=stable_fingerprint(contents)))
end

@testset "heterogeneous deterministic timing sample and raw evidence" begin
    indices = _deterministic_random_indices(10, 4, DETERMINISTIC_RANDOM_TIMING_SEED)
    @test indices == _deterministic_random_indices(10, 4, DETERMINISTIC_RANDOM_TIMING_SEED)
    @test length(unique(indices)) == 4 && all(index -> 1 <= index <= 10, indices)
    @test_throws ArgumentError _deterministic_random_indices(10, 0, UInt64(1))
    @test_throws ArgumentError _deterministic_random_indices(10, 11, UInt64(1))
    if Threads.nthreads() == 1
        previous_blas_threads = LinearAlgebra.BLAS.get_num_threads()
        LinearAlgebra.BLAS.set_num_threads(1)
        try
            mktempdir() do directory
                accuracy_path = joinpath(directory, "accuracy.jld2")
                timing_path = joinpath(directory, "random-timing.jld2")
                benchmark = randomized_performance(n_draws=4, t_max=4, parallel=false,
                    asset_grids=[collect(range(0.0, 50.0; length=21)),
                        Ptd.iid_dcegm_asset_grid(31; asset_maximum=100.0)],
                    artifact_path=accuracy_path).artifact
                original_accuracy = read(accuracy_path)
                artifact = run_deterministic_random_timings(accuracy_path;
                    n_draws=2, seed=321, output_path=timing_path)
                summary = load_deterministic_random_timings(timing_path, benchmark)
                @test read(accuracy_path) == original_accuracy
                @test artifact.schema == DETERMINISTIC_RANDOM_TIMING_SCHEMA
                @test summary.configuration.n_draws == 2
                @test summary.configuration.population_size == 4
                @test summary.configuration.specs ==
                    benchmark.specs[summary.configuration.sample_indices]
                @test summary.configuration.julia_threads == summary.configuration.blas_threads == 1
                @test length(artifact.measurements) == 6
                @test getproperty.(artifact.measurements, :method) ==
                    [:ptd, :dcegm1, :dcegm2, :dcegm1, :dcegm2, :ptd]
                @test Tuple(row.method for row in summary.rows) == (:ptd, :dcegm1, :dcegm2)
                for row in summary.rows
                    samples = filter(s -> s.method === row.method, artifact.measurements)
                    totals = getproperty.(samples, :total_seconds)
                    @test row.n_draws == length(samples) == 2
                    @test row.mean_seconds == mean(totals)
                    @test row.median_seconds == median(totals)
                    @test all(s -> s.total_seconds == s.construction_seconds + s.evaluation_seconds,
                        samples)
                    @test Set(s.accuracy_index for s in samples) ==
                        Set(summary.configuration.sample_indices)
                end
                # Attaching display timing does not change the parent accuracy identity.
                attached = merge(benchmark, (; timing=(display_only=true,)))
                @test load_deterministic_random_timings(timing_path, attached).rows == summary.rows
                changed_accuracy = deepcopy(benchmark)
                changed_accuracy.utilities[1, 1] += 1.0
                @test_throws ArgumentError load_deterministic_random_timings(timing_path, changed_accuracy)
                @test_throws ArgumentError run_deterministic_random_timings(accuracy_path;
                    n_draws=5, output_path=joinpath(directory, "too-many.jld2"))
                @test_throws ArgumentError run_deterministic_random_timings(accuracy_path;
                    n_draws=1, output_path=accuracy_path)
                @test_throws ArgumentError run_deterministic_random_timings(accuracy_path;
                    n_draws=1, output_path=timing_path)
                @test_throws ArgumentError run_deterministic_random_timings(accuracy_path; n_draws=1, seed=-1)

                # Recomputed fingerprints cannot hide inconsistent underlying evidence.
                changed_rows = copy(artifact.rows)
                changed_rows[1] = merge(changed_rows[1], (; mean_seconds=changed_rows[1].mean_seconds + 1.0))
                invalid_summary = authenticated(merge(artifact, (; rows=changed_rows)))
                @test_throws ArgumentError _validate_deterministic_random_artifact(invalid_summary, benchmark)
                changed_measurements = copy(artifact.measurements)
                changed_measurements[1] = merge(changed_measurements[1],
                    (; utility=changed_measurements[1].utility + 1.0))
                invalid_utility = authenticated(merge(artifact, (; measurements=changed_measurements)))
                @test_throws ArgumentError _validate_deterministic_random_artifact(invalid_utility, benchmark)
                different_indices = reverse(artifact.configuration.sample_indices)
                invalid_sample = authenticated(merge(artifact, (; configuration=merge(
                    artifact.configuration, (; sample_indices=different_indices)))))
                @test_throws ArgumentError _validate_deterministic_random_artifact(invalid_sample, benchmark)
                missing_sample = authenticated(merge(artifact, (; measurements=artifact.measurements[1:end-1])))
                @test_throws ArgumentError _validate_deterministic_random_artifact(missing_sample, benchmark)
                invalid_fingerprint = merge(artifact, (; authentication_fingerprint="tampered"))
                @test_throws ArgumentError _validate_deterministic_random_artifact(invalid_fingerprint, benchmark)
            end
        finally
            LinearAlgebra.BLAS.set_num_threads(previous_blas_threads)
        end
    else
        @test_throws ArgumentError run_deterministic_random_timings("unused.jld2")
    end
end
end
