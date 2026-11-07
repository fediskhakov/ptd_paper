# Tests for src/simulation/ar1/exhibits.jl and source-order file 25.
module AR1ContestExhibitsTests

using Dates
using JLD2
using LinearAlgebra
using Random
using Test
using Ptd

const AR1Exhibits = Ptd.AR1

function tiny_grid()
    return Ptd.AR1G2ContestGridConfiguration(
        :ar1_contest_exhibit_test_v1,
        [0.0, 1.0, 4.0, 20.0],
        [-3.0, 0.0, 3.0],
        [3.0, 7.0, 15.0, 30.0, 45.0],
        [-3.0, 0.0, 3.0],
    )
end

function tiny_config()
    return AR1Exhibits.AR1ContestExhibitConfig(
        beta=0.98,
        gamma=1.0,
        return_rate=0.10,
        cash_floor=3.0,
        horizon=3,
        rho=0.9,
        sigma=0.1,
        unconditional_log_mean=0.0,
        initial_income=1.0,
        initial_cash_support=[5.0, 7.0],
        n_paths=4,
        trace_paths=2,
        evaluation_seed=UInt64(0x4152315f54455354),
        confidence_level=0.95,
        quadrature_nodes_per_dimension=5,
        workers=min(2, Threads.nthreads()),
        surface_date=1,
        surface_cash_grid=[3.0, 5.0, 7.0],
        surface_log_income_grid=[-0.1, 0.0, 0.1],
        g2egm_grids=tiny_grid(),
        ptd_osa_options=Ptd.PTDOSAOptions(direct_iterations=8),
    )
end

@testset "AR(1) contest exhibits" begin
    production = AR1Exhibits.production_ar1_contest_exhibit_config()
    @test production.beta == 0.98
    @test production.gamma == 1.0
    @test production.initial_cash_support ==
        [5.5, 10.75, 16.0, 21.25, 26.5]
    @test production.rho == 0.9
    @test production.sigma == 0.1
    @test production.initial_income == 1.0
    @test production.horizon == 50
    @test production.n_paths == 1_000
    @test production.trace_paths == 100
    @test AR1Exhibits._paths_per_initial_cash(production) == 200
    @test AR1Exhibits._trace_paths_per_initial_cash(production) == 20
    @test [count(==(cash), AR1Exhibits._initial_cash_assignments(production))
           for cash in production.initial_cash_support] == fill(200, 5)
    @test [count(==(cash), AR1Exhibits._initial_cash_assignments(production)[
               AR1Exhibits._trace_path_ids(production)])
           for cash in production.initial_cash_support] == fill(20, 5)
    @test production.workers == min(production.n_paths, Threads.nthreads())
    @test AR1Exhibits.AR1ContestExhibitConfig(n_paths=1).workers == 1
    for workers in (1, 3, 24)
        parallel_config = AR1Exhibits.AR1ContestExhibitConfig(; workers)
        @test AR1Exhibits._is_publication_config(parallel_config)
        @test AR1Exhibits.scientific_configuration(parallel_config).
            evaluation.workers == workers
        @test !AR1Exhibits._is_publication_config(
            AR1Exhibits.AR1ContestExhibitConfig(; workers, beta=0.97))
    end
    @test AR1Exhibits.scientific_fingerprint(
        AR1Exhibits.AR1ContestExhibitConfig(workers=1)) !=
        AR1Exhibits.scientific_fingerprint(
            AR1Exhibits.AR1ContestExhibitConfig(workers=24))
    @test length(production.surface_cash_grid) == 81
    @test length(production.surface_log_income_grid) == 41
    @test AR1Exhibits.validate_ar1_contest_exhibit_config(production) === production
    @test AR1Exhibits.scientific_fingerprint(production) ==
        AR1Exhibits.scientific_fingerprint(
            AR1Exhibits.production_ar1_contest_exhibit_config())
    scientific = AR1Exhibits.scientific_configuration(production)
    @test scientific.quadrature.ptd_sceq_uses_quadrature === false
    @test scientific.quadrature.
        ptd_osa_and_g2egm_share_exact_q5 === true

    config = tiny_config()
    @test AR1Exhibits.validate_ar1_contest_exhibit_config(config) === config
    @test_throws ArgumentError AR1Exhibits.validate_ar1_contest_exhibit_config(
        AR1Exhibits.AR1ContestExhibitConfig(
            n_paths=4, trace_paths=5, workers=1))

    BLAS.set_num_threads(1)
    @test isnothing(AR1Exhibits.warm_ar1_contest_exhibits!(production))
    @test isnothing(AR1Exhibits.warm_ar1_contest_exhibits!(config))
    artifact = AR1Exhibits.build_ar1_contest_exhibit_artifact(config;
        warmup=false,
        require_publication=false,
        completed_at=DateTime(2026, 8, 5, 0, 0, 0))
    @test AR1Exhibits.validate_ar1_contest_exhibit_artifact(artifact;
        expected_config=config, require_publication=false) === artifact
    # Publication validation, the default, accepts only the paper's
    # configuration, so this valid but tiny exhibit is rejected.
    @test_throws "not a recognized paper exhibit" (
        AR1Exhibits.validate_ar1_contest_exhibit_artifact(artifact;
            expected_config=config))

    @test artifact.metadata.artifact_schema ==
        AR1Exhibits.AR1_CONTEST_EXHIBIT_SCHEMA
    @test artifact.metadata.status === :complete
    @test artifact.welfare.methods ==
        collect(AR1Exhibits.AR1_CONTEST_EXHIBIT_METHODS)
    @test size(artifact.welfare.utility) == (4, 3)
    @test size(artifact.welfare.floor_hits) == (4, 3)
    @test size(artifact.welfare.total_transfers) == (4, 3)
    @test size(artifact.welfare.innovations) == (4, 2)
    @test artifact.welfare.initial_cash_assignments == [5.0, 5.0, 7.0, 7.0]
    @test artifact.welfare.innovations ==
        randn(Xoshiro(config.evaluation_seed), 4, 2)
    @test length(artifact.welfare.expected_utility) == 3
    @test length(artifact.welfare.comparisons) == 3
    @test artifact.quadrature.ptd_sceq.uses_quadrature === false
    @test artifact.quadrature.ptd_osa.nodes_per_dimension == 5
    @test artifact.quadrature.g2egm.nodes_per_dimension == 5
    @test artifact.quadrature.exact_shared_q5 === true
    @test artifact.timing.thread_pools.default_threads == Threads.nthreads(:default)
    @test artifact.timing.thread_pools.interactive_threads == Threads.nthreads(:interactive)
    @test artifact.timing.thread_pools.default_thread_ids == Threads.threadpooltids(:default)
    @test artifact.timing.thread_pools.interactive_thread_ids == Threads.threadpooltids(:interactive)
    @test isnothing(AR1Exhibits._validate_timing(
        artifact.timing, artifact.scientific_configuration;
        require_publication=true))
    @test_throws "did not use BLAS=1" AR1Exhibits._validate_timing(
        merge(artifact.timing, (blas_threads=2,)),
        artifact.scientific_configuration; require_publication=true)

    for method in AR1Exhibits.AR1_CONTEST_EXHIBIT_METHODS
        construction = artifact.timing.construction_seconds[method]
        evaluation = artifact.timing.evaluation_seconds[method]
        @test construction >= 0.0
        @test evaluation >= 0.0
        @test artifact.timing.end_to_end_seconds[method] ==
            construction + evaluation
        @test !isempty(
            artifact.timing.evaluation_boundary_thread_ids[method])
        @test size(artifact.surface.consumption[method]) == (3, 3)
        @test size(artifact.surface.saving[method]) == (3, 3)
        @test artifact.surface.saving[method] ≈
            reshape(config.surface_cash_grid, :, 1) .-
            artifact.surface.consumption[method]
        @test size(artifact.traces.cash[method]) == (3, 2)
        @test size(artifact.traces.consumption[method]) == (3, 2)
        column = findfirst(==(method), artifact.welfare.methods)
        @test artifact.traces.replayed_utility[method] ==
            artifact.welfare.utility[[1, 3], column]
    end
    @test artifact.traces.path_ids == [1, 3]
    @test artifact.traces.initial_cash == [5.0, 7.0]
    @test artifact.traces.dates == [1, 2, 3]
    @test size(artifact.traces.income) == (3, 2)
    @test artifact.traces.innovation_rows ==
        artifact.welfare.innovations[[1, 3], :]
    @test artifact.surface.outside_reported_timing === true
    @test artifact.traces.outside_reported_timing === true
    @test artifact.traces.adds_welfare_observations === false

    bad_path_count = merge(artifact.welfare, (n_paths=5,))
    @test_throws ArgumentError AR1Exhibits._validate_welfare(
        bad_path_count, artifact.scientific_configuration)

    wrong_seed_block = copy(artifact.welfare.innovations)
    wrong_seed_block[1, 1] += 1.0
    bad_seed_welfare = merge(artifact.welfare, (
        innovations=wrong_seed_block,
        shock_fingerprint=Ptd.stable_fingerprint(
            wrong_seed_block),
    ))
    @test_throws ArgumentError AR1Exhibits._validate_welfare(
        bad_seed_welfare, artifact.scientific_configuration)

    bad_trace_ids = merge(artifact.traces, (path_ids=[1, 2],))
    @test_throws ArgumentError AR1Exhibits._validate_traces(
        bad_trace_ids, artifact.welfare,
        artifact.scientific_configuration)
    wrong_income = copy(artifact.traces.income)
    wrong_income[2, 1] += 1.0
    bad_trace_income = merge(artifact.traces, (income=wrong_income,))
    @test_throws ArgumentError AR1Exhibits._validate_traces(
        bad_trace_income, artifact.welfare,
        artifact.scientific_configuration)

    bad_surface_date = merge(artifact.surface, (date=2,))
    @test_throws ArgumentError AR1Exhibits._validate_surface(
        bad_surface_date, artifact.scientific_configuration)

    wrong_node_bits = collect(artifact.quadrature.node_bits)
    wrong_node_bits[1] = xor(wrong_node_bits[1], UInt64(1))
    bad_quadrature = merge(artifact.quadrature,
        (node_bits=Tuple(wrong_node_bits),))
    @test_throws ArgumentError AR1Exhibits._exhibit_validate_quadrature(
        bad_quadrature, artifact.scientific_configuration)

    bad_partitions = merge(artifact.timing,
        (partition_ranges=[2:4 for _ in 1:config.workers],))
    @test_throws ArgumentError AR1Exhibits._validate_timing(
        bad_partitions, artifact.scientific_configuration;
        require_publication=false)
    bad_thread_fields = merge(
        artifact.timing.evaluation_boundary_thread_ids,
        (ptd_sceq=[0],))
    bad_thread_ids = merge(artifact.timing,
        (evaluation_boundary_thread_ids=bad_thread_fields,))
    @test_throws ArgumentError AR1Exhibits._validate_timing(
        bad_thread_ids, artifact.scientific_configuration;
        require_publication=false)

    tampered_utility = copy(artifact.welfare.utility)
    tampered_utility[1, 1] += 1.0
    tampered = merge(artifact, (welfare=merge(
        artifact.welfare, (utility=tampered_utility,)),))
    @test_throws ArgumentError AR1Exhibits.validate_ar1_contest_exhibit_artifact(
        tampered; expected_config=config, require_publication=false)

    tampered_provenance = merge(artifact, (provenance=merge(
        artifact.provenance,
        (upstream=merge(artifact.provenance.upstream,
            (commit="0"^40,)),)),))
    @test_throws ArgumentError AR1Exhibits.validate_ar1_contest_exhibit_artifact(
        tampered_provenance;
        expected_config=config,
        require_publication=false)

    mktempdir() do directory
        path = joinpath(directory, "ar1-contest-exhibit.jld2")
        saved = AR1Exhibits.save_ar1_contest_exhibit_artifact(path, artifact;
            compress=false, require_publication=false)
        @test saved.path == abspath(path)
        @test isfile(path)
        loaded = AR1Exhibits.load_ar1_contest_exhibit_artifact(path;
            expected_config=config, require_publication=false)
        @test loaded.metadata.authentication_fingerprint ==
            artifact.metadata.authentication_fingerprint

        resumed = AR1Exhibits.run_ar1_contest_exhibits(config;
            output_path=path,
            warmup=false,
            require_publication=false,
            compress=false)
        @test resumed.status === :loaded
        @test resumed.quarantined === nothing
        @test resumed.artifact.metadata.authentication_fingerprint ==
            artifact.metadata.authentication_fingerprint

        lock_path = Ptd._runtime_lock_path(path, :ar1_exhibit)
        mkpath(dirname(lock_path))
        mkdir(lock_path)
        @test_throws ArgumentError AR1Exhibits.run_ar1_contest_exhibits(
            config;
            output_path=path,
            warmup=false,
            require_publication=false,
            compress=false,
            lock_run=true)
        rm(lock_path)
    end
end

@testset "AR(1) exhibit timing supports an interactive thread pool" begin
    # Two static partitions must actually use both default workers. With an
    # interactive pool their global IDs are 2 and 3, although nthreads() is 2.
    code = raw"""
    using Dates, Test, Ptd
    const Exhibits = Ptd.AR1
    @testset "two default workers and one interactive worker" begin
        @test Threads.nthreads(:default) == 2
        @test Threads.nthreads(:interactive) == 1
        grids = Ptd.AR1G2ContestGridConfiguration(
            :ar1_thread_pool_regression_v1,
            [0.0, 1.0, 4.0, 20.0], [-3.0, 0.0, 3.0],
            [3.0, 7.0, 15.0, 30.0, 45.0], [-3.0, 0.0, 3.0])
        config = Exhibits.AR1ContestExhibitConfig(
            horizon=3, initial_cash_support=[5.0, 7.0],
            n_paths=4, trace_paths=2, workers=2,
            surface_cash_grid=[3.0, 5.0, 7.0],
            surface_log_income_grid=[-0.1, 0.0, 0.1],
            g2egm_grids=grids,
            ptd_osa_options=Ptd.PTDOSAOptions(direct_iterations=8))
        artifact = Exhibits.build_ar1_contest_exhibit_artifact(config;
            warmup=false, require_publication=false,
            completed_at=DateTime(2026, 8, 5))
        @test artifact.timing.thread_pools.default_thread_ids == [2, 3]
        @test artifact.timing.thread_pools.interactive_thread_ids == [1]
        for method in Exhibits.AR1_CONTEST_EXHIBIT_METHODS
            @test artifact.timing.evaluation_boundary_thread_ids[method] == [2, 3]
        end
        @test Exhibits.validate_ar1_contest_exhibit_artifact(artifact;
            require_publication=false) === artifact

        # An interactive thread is not a valid evaluation worker, and metadata
        # cannot broaden the default pool to admit it or an out-of-range ID.
        for bad_pools in (
            merge(artifact.timing.thread_pools, (default_thread_ids=[1, 2, 3],)),
            merge(artifact.timing.thread_pools, (interactive_threads=0,)),
            merge(artifact.timing.thread_pools, (interactive_thread_ids=[2],)),
            merge(artifact.timing.thread_pools, (default_threads=3,)))
            bad_timing = merge(artifact.timing, (thread_pools=bad_pools,))
            @test_throws ArgumentError Exhibits._validate_timing(
                bad_timing, artifact.scientific_configuration;
                require_publication=false)
        end
        for bad_ids in ([1], [4])
            boundaries = merge(artifact.timing.evaluation_boundary_thread_ids,
                (ptd_sceq=bad_ids,))
            bad_timing = merge(artifact.timing,
                (evaluation_boundary_thread_ids=boundaries,))
            @test_throws ArgumentError Exhibits._validate_timing(
                bad_timing, artifact.scientific_configuration;
                require_publication=false)
        end

        # A different internally consistent pool layout still changes the
        # exact authenticated timing payload and must fail the full validator.
        altered_pools = (default_threads=2, interactive_threads=2,
            default_thread_ids=[3, 4], interactive_thread_ids=[1, 2])
        altered_boundaries = (ptd_sceq=[3, 4], ptd_osa=[3, 4], g2egm=[3, 4])
        altered_timing = merge(artifact.timing, (thread_pools=altered_pools,
            evaluation_boundary_thread_ids=altered_boundaries))
        @test isnothing(Exhibits._validate_timing(altered_timing,
            artifact.scientific_configuration; require_publication=false))
        @test_throws ArgumentError Exhibits.validate_ar1_contest_exhibit_artifact(
            merge(artifact, (timing=altered_timing,)); require_publication=false)
        mktempdir() do directory
            path = joinpath(directory, "thread-pools.jld2")
            Exhibits.save_ar1_contest_exhibit_artifact(path, artifact;
                compress=false, require_publication=false)
            loaded = Exhibits.load_ar1_contest_exhibit_artifact(path;
                require_publication=false)
            @test loaded.timing.thread_pools == artifact.timing.thread_pools
            @test loaded.metadata.authentication_fingerprint ==
                artifact.metadata.authentication_fingerprint
        end
    end
    """
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(pkgdir(Ptd)) --threads=2,1 -e $code`
    @test success(command)
end

end # module AR1ContestExhibitsTests
