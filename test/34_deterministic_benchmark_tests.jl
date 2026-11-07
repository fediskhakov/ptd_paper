module DeterministicBenchmarkTests
using Test
import Ptd
include(joinpath(@__DIR__, "..", "scripts", "2_deterministic.jl"))

@testset "scale-aware deterministic outcomes and wide grid" begin
    small = deterministic_outcomes([1.0, 1.0 + 1e-15, 0.9], [2.0, 2.0, 1.0])
    large = deterministic_outcomes(1e6 .* [1.0, 1.0 + 1e-15, 0.9],
        1e6 .* [2.0, 2.0, 1.0])
    @test small.ties == large.ties == [true, true, false]
    @test !any(small.strict_wins) && !any(large.strict_wins)
    clear = deterministic_outcomes([1.0, 1.001, 0.9], [2.0, 2.0, 1.0])
    @test clear.strict_wins == [false, true, false]
    @test !any(clear.ties)
    cancellation = deterministic_outcomes([0.0, 1e-8], [1e8, 1e8])
    @test all(cancellation.ties)
    # Halving consumption at every date costs 50 percent under either utility.
    @test ce_gap_basis_points(0.0, 2log(2.0);
        beta=1.0, gamma=1.0, t_max=2) ≈ -5000.0
    @test ce_gap_basis_points(1.0, 1.5;
        beta=1.0, gamma=2.0, t_max=2) ≈ -5000.0
    grids = deterministic_asset_grids()
    @test length(grids) == 4
    @test length(grids[4]) == 400
    @test first(grids[4]) == 0.0 && last(grids[4]) == 10_000.0
    @test all(diff(grids[4]) .> 0)
    @test isapprox(diff(log1p.(grids[4])),
        fill(log(10_001.0) / 399, 399); rtol=1e-11)
end

@testset "deterministic raw evidence regenerates outcomes, losses, and table" begin
    mktempdir() do directory
        artifact_path = joinpath(directory, "benchmark.jld2")
        table_path = joinpath(directory, "table.tex")
        benchmark = randomized_performance(; n_draws=4, t_max=4,
            asset_grids=[collect(range(0.0, 50.0; length=21)),
                Ptd.iid_dcegm_asset_grid(81; asset_maximum=60_000.0)],
            save_path=table_path, artifact_path)
        artifact = JLD2.load(artifact_path)["artifact"]
        @test artifact.tail_safeguard.unresolved_count == 0
        @test artifact.provenance.threads == Threads.nthreads()
        @test artifact.provenance.parallel_draws
        @test artifact.provenance.scope == "accuracy_and_method_batch_wall_time"
        @test !hasproperty(artifact, :times_seconds)
        @test !hasproperty(artifact.summary, :median_time_ms)
        @test isnothing(artifact.timing)
        @test size(benchmark.data) == (3, 7)
        @test all(==("not reported"), benchmark.data[:, 3:4])
        @test all(isfinite, benchmark.batch_time_seconds)
        @test all(>=(0.0), benchmark.batch_time_seconds)
        @test artifact.batch_timing.threads == Threads.nthreads()
        @test artifact.batch_timing.scope === :complete_method_accuracy_pass
        @test artifact.provenance.parameter_design_fingerprint ==
            stable_fingerprint(artifact.specs)
        expected_specs = build_contest_design(ContestDesignConfig(n_contests=4,
            horizon=4, floor=3.0))
        @test artifact.specs == [(; s.id, s.beta, s.gamma, s.initial_cash)
            for s in expected_specs]
        @test size(artifact.segment_counts) == (4, 4)
        @test all(artifact.segment_counts[:, end] .== 1)
        @test benchmark.mean_segments == mean(artifact.segment_counts)
        @test benchmark.max_segments == maximum(artifact.segment_counts)
        strict = zeros(Int, 3)
        ties = zeros(Int, 3)
        for j in axes(artifact.utilities, 1)
            outcome = deterministic_outcomes(artifact.utilities[j, :],
                artifact.absolute_utility_sums[j, :])
            strict .+= outcome.strict_wins
            ties .+= outcome.ties
        end
        @test strict == benchmark.strict_wins
        @test ties == benchmark.ties
        @test benchmark.winners == strict + ties
        loss = max.(-artifact.ce_gaps_bp, 0.0)
        @test benchmark.median_loss_bp == vec(median(loss; dims=1))
        @test benchmark.p95_loss_bp == [quantile(loss[:, m], 0.95) for m in 1:3]
        @test benchmark.max_loss_bp == vec(maximum(loss; dims=1))
        serial = randomized_performance(; n_draws=4, t_max=4,
            asset_grids=artifact.configuration.asset_grids, parallel=false)
        @test serial.artifact.specs == artifact.specs
        @test serial.artifact.utilities == artifact.utilities
        @test serial.artifact.absolute_utility_sums == artifact.absolute_utility_sums
        @test serial.artifact.ce_gaps_bp == artifact.ce_gaps_bp
        @test serial.artifact.segment_counts == artifact.segment_counts
        for field in (:strict_wins, :ties, :median_loss_bp, :p95_loss_bp,
            :max_loss_bp, :mean_segments, :max_segments)
            @test getproperty(serial.artifact.summary, field) == getproperty(artifact.summary, field)
        end
        rerendered = joinpath(directory, "rerendered.tex")
        render_deterministic_benchmark(artifact_path; table_path=rerendered)
        @test read(table_path) == read(rerendered)
        untimed_tex = read(table_path, String)
        @test occursin("Mean", untimed_tex) && occursin("Median", untimed_tex)
        @test !occursin("Batch time", untimed_tex)
        @test !occursin("warmed solve per draw", untimed_tex)
        timing = (; rows=[(; method, n_draws=1000,
                    mean_seconds=m * 0.00125, median_seconds=m * 0.001)
                for (m, method) in enumerate((:ptd, :dcegm1, :dcegm2))],
            configuration=(; n_draws=1000),
            artifact_fingerprint="synthetic-test-timing", source_fingerprint="synthetic-test-source")
        # Rendering uses a synthetic 1,000-of-10,000 timing subset; the raw
        # four-draw accuracy fixture and all scientific checks remain unchanged.
        with_timing = merge(artifact, (; timing,
            configuration=merge(artifact.configuration, (; n_draws=10_000))))
        timed_data, _ = _deterministic_table_data(with_timing)
        @test timed_data[:, 3] == ["1.250", "2.500", "3.750"]
        @test timed_data[:, 4] == ["1.000", "2.000", "3.000"]
        @test timed_data[:, [1, 2, 5, 6, 7]] == benchmark.data[:, [1, 2, 5, 6, 7]]
        @test timed_data[:, 7] == [@sprintf("%.2f", loss / 100)
            for loss in artifact.summary.max_loss_bp]
        timed_table = joinpath(directory, "timed.tex")
        _write_deterministic_table(timed_table, with_timing)
        timed_tex = read(timed_table, String)
        @test occursin(r"1,?000", timed_tex) && occursin(r"10,?000", timed_tex)
        @test occursin("warmup", lowercase(timed_tex))
        @test occursin(r"(?i)(one|single)[- ](?:Julia |BLAS )?thread", timed_tex)
        @test occursin(r"(?i)ties.*(?:sharing|shared|largest|best)", timed_tex)
        @test occursin(r"(?i)(?:loss|shortfall).*relative to PTD", timed_tex)
        @test occursin("Timing artifact fingerprint: synthetic-test-timing", timed_tex)
        @test occursin("Timing source fingerprint: synthetic-test-source", timed_tex)
        @test !occursin("\\resizebox", timed_tex)
        @test !occursin("A. Grids and computational cost", timed_tex)
        @test !occursin("B. Accuracy", timed_tex)
        @test !occursin("Batch time", timed_tex)
        @test !occursin("Median CEG", timed_tex) && !occursin("P95", timed_tex)
        blocks = split(timed_tex, "\\begin{tabular}")
        @test length(blocks) == 2
        # The merged table preserves every source cell in the declared order.
        rows = filter(line -> startswith(line, "PTD &") ||
            startswith(line, "DC-EGM"), split(blocks[2], '\n'))
        @test length(rows) == size(timed_data, 1)
        for (m, line) in enumerate(rows)
            cells = strip.(split(replace(line, "\\\\" => ""), '&'))
            @test cells == string.(timed_data[m, :])
        end
        artifact.utilities[1, 1] += 1.0
        tampered = joinpath(directory, "tampered.jld2")
        JLD2.jldsave(tampered; artifact,
            payload_fingerprint=JLD2.load(artifact_path)["payload_fingerprint"])
        @test_throws ArgumentError render_deterministic_benchmark(tampered;
            table_path=rerendered)
    end
end

@testset "parallel deterministic draws propagate the earliest failure" begin
    corner = Ptd.PolicySegment(xStart=0.0, xEnd=Inf, b0=0.0, b1=1.0)
    fatal = Ptd.FusionTailError(1, 1, 1e16, (; beta=0.95), corner, corner)
    for parallel in (false, true)
        visited = zeros(Int, 8)
        caught = try
            _run_deterministic_draws!(collect(1:8); parallel) do j, spec
                visited[j] += 1
                j == 2 && throw(fatal)
                j == 5 && error("later draw failure")
            end
            nothing
        catch error
            error
        end
        @test caught === fatal
        @test visited[1:2] == [1, 1]
        @test all(visited .<= 1)
    end
end
end

@testset "uncertified tails abort retained-failure contest runs" begin
    config = Ptd.ContestDesignConfig(n_contests=1, horizon=2)
    corner = Ptd.PolicySegment(xStart=0.0, xEnd=Inf, b0=0.0, b1=1.0)
    fatal = Ptd.FusionTailError(1, 1, 1e16, (; beta=0.95), corner, corner)
    failed_task = Task(() -> throw(fatal))
    schedule(failed_task)
    wrapped = try
        wait(failed_task)
        nothing
    catch error
        error
    end
    @test wrapped isa TaskFailedException
    @test Ptd._has_uncertified_tail(wrapped)
    @test Ptd._has_uncertified_tail(CompositeException([wrapped]))
    @test !Ptd._has_uncertified_tail(ErrorException("ordinary solver failure"))
    serial = (f, items, workers) -> foreach(f, items)
    for error in (fatal, wrapped, CompositeException([wrapped]))
        caught = try
            Ptd.complete_contest_design(config;
                replacement_seed=UInt64(9876), attempt=spec -> throw(error),
                scheduler=serial, workers=1, audit_schema="test-tail-guard",
                failure_policy=:retain)
            nothing
        catch propagated
            propagated
        end
        @test caught === error
    end
    retained = Ptd.complete_contest_design(config;
        replacement_seed=UInt64(9876), attempt=spec -> error("ordinary solver failure"),
        scheduler=serial, workers=1, audit_schema="test-tail-guard", failure_policy=:retain)
    @test only(retained.results) isa Ptd.ContestFailure
end
