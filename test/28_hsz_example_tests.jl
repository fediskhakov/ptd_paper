# Tests for src/simulation/hsz/example.jl and source-order file 27.
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Test
using Random: Xoshiro

const HSZExample = Ptd.HSZ

@testset "HSZ production grid and warm-up contract" begin
    config = HSZExample.production_hsz_example_config()
    post_points = config.asset_points * config.post_earnings_points *
        config.post_medical_points
    common_points = config.common_cash_points * config.common_earnings_points *
        config.common_medical_points

    @test HSZExample.HSZ_EXAMPLE_SCHEMA == "ptd-hsz-example-v3"
    @test config.unconditional_log_medical == log(1.5)
    @test config.unconditional_log_earnings == log(2.0)
    @test basename(HSZExample.default_hsz_example_artifact_path(config)) ==
        "hsz-example-medical-1p5.jld2"
    @test (config.asset_points, config.post_earnings_points,
        config.post_medical_points) == (401, 17, 17)
    @test (config.common_cash_points, config.common_earnings_points,
        config.common_medical_points) == (401, 17, 17)
    @test post_points == common_points == 115_889
    @test post_points > 1_601 * 61
    @test common_points > 1_201 * 41

    scientific = HSZExample._scientific_configuration(config)
    @test scientific.quadrature.ptd_sceq_uses_quadrature === false
    @test scientific.quadrature.ptd_osa_and_g2egm_share_rule === true
    @test scientific.g2egm.postdecision_total_points == post_points
    @test scientific.g2egm.common_total_points == common_points
    @test scientific.process.unconditional_log_medical == log(1.5)

    tiny = HSZExample.HSZExampleConfig(horizon=3,
        initial_cash_support=[5.5], n_paths=1, paths_per_initial_cash=1,
        asset_points=9, common_cash_points=9,
        post_earnings_points=5, post_medical_points=5,
        common_earnings_points=5, common_medical_points=5)
    @test isnothing(HSZExample.warm_hsz_example!(tiny))
end

@testset "HSZ logical workers support an interactive thread pool" begin
    # Default-pool IDs are 2 and 3 here, but exactly two policy instances exist.
    # Use enough paths to exercise both owners and compare with one owner.
    code = raw"""
    using Test, Ptd
    @test Threads.nthreads(:default) == 2
    @test Threads.nthreads(:interactive) == 1
    example = Ptd.HSZ.HSZExampleConfig(horizon=3,
        initial_cash_support=[5.5], n_paths=8, paths_per_initial_cash=8)
    config, process = Ptd.HSZ._model(example)
    rule = Ptd.gauss_hermite_rule(process, 5)
    policies = Ptd.HSZ._build_worker_policies(:ptd_sceq, config, process, rule, nothing)
    @test length(policies) == 2
    shocks = zeros(Float64, 8, 2, 2)
    initial_cash = collect(range(5.5, 12.5; length=8))
    parallel = Ptd.HSZ._evaluate_method(:ptd_sceq, policies, config, process,
        shocks, initial_cash)
    @test all(Ptd.plan_cache_statistics(policy.cache).builds > 0 for policy in policies)
    single_policy = Ptd.build_hsz_contest_policy(:ptd_sceq, config, process; base_rule=rule)
    sequential = Ptd.HSZ._evaluate_method(:ptd_sceq, [single_policy], config, process,
        shocks, initial_cash)
    @test parallel.cash == sequential.cash
    @test parallel.consumption == sequential.consumption
    @test parallel.utility == sequential.utility
    @test parallel.floor_hits == sequential.floor_hits
    @test all(isfinite, parallel.utility)
    """
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(pkgdir(Ptd)) --threads=2,1 -e $code`
    @test success(command)
end

@testset "HSZ medical calibration changes the scientific fingerprint" begin
    baseline_config = HSZExample.HSZExampleConfig(
        unconditional_log_medical=log(0.5))
    baseline_scientific = HSZExample._scientific_configuration(baseline_config)
    @test baseline_scientific.process.unconditional_log_medical == log(0.5)
    @test HSZExample._scientific_fingerprint(baseline_config) !=
        HSZExample._scientific_fingerprint(HSZExample.production_hsz_example_config())
end

@testset "tracked HSZ artifact and paper renderer" begin
    config = HSZExample.production_hsz_example_config()
    artifact_path = HSZExample.default_hsz_example_artifact_path(config)
    @test isfile(artifact_path)

    artifact = HSZExample.load_hsz_example(artifact_path)
    @test artifact.schema == HSZExample.HSZ_EXAMPLE_SCHEMA
    @test artifact.scientific_fingerprint ==
        HSZExample._scientific_fingerprint(config)
    @test artifact.scientific_configuration.process.unconditional_log_medical == log(1.5)
    @test artifact.innovations == randn(Xoshiro(config.seed),
        config.n_paths, config.horizon - 1, 2)
    @test artifact.initial_cash == HSZExample._initial_cash(config)
    # Check a medical-location shift using the saved shocks, without keeping
    # a second historical policy artifact or rerunning any solver.
    baseline_config = HSZExample.HSZExampleConfig(
        unconditional_log_medical=log(0.5))
    _, baseline_process = HSZExample._model(baseline_config)
    baseline_exogenous = HSZExample._common_exogenous_paths(
        baseline_config, baseline_process, artifact.innovations)
    @test artifact.exogenous.log_earnings == baseline_exogenous.log_earnings
    @test isapprox(artifact.exogenous.log_medical - baseline_exogenous.log_medical,
        fill(log(3.0), size(artifact.exogenous.log_medical)); atol=2e-14, rtol=0.0)
    @test artifact.julia.threads == 24
    @test artifact.julia.blas_threads == 1
    @test size(artifact.innovations) ==
        (config.n_paths, config.horizon - 1, 2)
    @test unique(artifact.initial_cash) == config.initial_cash_support
    @test all(count(==(cash), artifact.initial_cash) ==
        config.paths_per_initial_cash for cash in config.initial_cash_support)

    for method in HSZExample.HSZ_EXAMPLE_METHODS
        paths = getproperty(artifact.paths, method)
        @test size(paths.cash) == (config.horizon, config.n_paths)
        @test size(paths.consumption) == (config.horizon, config.n_paths)
        @test length(paths.utility) == config.n_paths
        @test all(isfinite, paths.cash)
        @test all(isfinite, paths.consumption)
        @test all(isfinite, paths.utility)
        @test all(0.0 .< paths.consumption .<= paths.cash)
        @test all(0 .<= paths.floor_hits .<= config.horizon - 1)
        @test all(isfinite, paths.total_transfers)
        @test all(paths.total_transfers .>= 0.0)
    end

    mktempdir() do root
        graph_path = joinpath(root, HSZExample.HSZ_PATHS_FILENAME)
        timing_path = joinpath(root, HSZExample.HSZ_TIMING_FILENAME)
        rendered = HSZExample.render_hsz_example(artifact_path;
            graph_path, timing_path)
        @test rendered.graph_path == graph_path
        @test rendered.timing_path == timing_path
        @test read(graph_path)[1:8] ==
            UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        table = read(timing_path, String)
        @test occursin("PTD-SCEQ", table)
        @test occursin("PTD-OSA", table)
        @test occursin("G2EGM", table)
        @test occursin("Mean time", table)
        @test occursin("Median time", table)
        @test occursin("not reported", table)
        @test !occursin("Construction (s)", table)
    end
end
