# Tests for src/simulation/common/contest_design.jl and source-order file 6.
using Test
using Ptd

const CD = Ptd

function stratum_indices(values, bounds)
    lower, upper = bounds
    n = length(values)
    return sort(floor.(Int, n .* ((values .- lower) ./ (upper - lower))))
end

@testset "Contest completion redraws failures one-for-one" begin
    config = CD.ContestDesignConfig(
        n_contests=4,
        n_paths=3,
        horizon=3,
        floor=1.0,
        bounds=CD.ParameterBounds(
            (0.92, 0.98), (1.0, 2.0), (1.0, 4.0)),
    )
    calls = Dict{Int,Int}()
    attempt = function (spec)
        calls[spec.id] = get(calls, spec.id, 0) + 1
        spec.id == 1 && calls[spec.id] == 1 && error("first draw fails")
        spec.id == 3 && calls[spec.id] <= 2 && return false
        return (contest_id=spec.id, beta=spec.beta)
    end
    serial = function (f, items, workers)
        workers == 1 || error("test scheduler expects one worker")
        foreach(f, items)
    end
    completed = CD.complete_contest_design(config;
        replacement_seed=UInt64(0x123456789abcdef0),
        attempt,
        scheduler=serial,
        workers=1,
        audit_schema="test-complete-contest-v1",
        failure_policy=:replace)

    @test length(completed.design) == 4
    @test getfield.(completed.design, :id) == collect(1:4)
    @test getfield.(completed.results, :contest_id) == collect(1:4)
    @test completed.audit.completed_contests == 4
    @test completed.audit.replacement_count == 2
    @test completed.audit.rule ===
        :replace_each_failed_configuration_one_for_one_until_complete
    @test completed.audit.rounds[1].failed_slots == (1, 3)
    @test completed.audit.rounds[2].failed_slots == (3,)
    @test isempty(completed.audit.rounds[3].failed_slots)
    @test calls == Dict(1 => 2, 2 => 1, 3 => 3, 4 => 1)
    @test completed.audit.final_design_fingerprint ==
        CD.stable_fingerprint(completed.plain_design)
end

@testset "Fixed contest design retains failures and bounds explicit replacements" begin
    config = CD.ContestDesignConfig(n_contests=3, n_paths=2, horizon=3)
    initial = CD.build_contest_design(config)
    serial = (f, items, workers) -> foreach(f, items)
    calls = zeros(Int, 3)
    attempt = spec -> begin
        calls[spec.id] += 1
        spec.id == 2 && error("monotone-saving certification failed")
        return spec.id
    end
    retained = CD.complete_contest_design(config;
        replacement_seed=UInt64(71), attempt, scheduler=serial, workers=1,
        audit_schema="test-fixed-contest-v1")
    @test retained isa CD.ContestDesignResult
    @test retained.design == initial
    @test retained.results[2] isa CD.ContestFailure
    @test retained.results[2].contest_id == 2
    @test occursin("certification", retained.results[2].message)
    @test calls == [1, 1, 1]
    @test retained.audit.failed_slots == (2,)
    @test retained.audit.completed_contests == 2
    @test retained.audit.failed_contests == 1
    @test retained.audit.original_failed_contests == 1
    @test retained.audit.replacement_count == 0

    failure = try
        CD.complete_contest_design(config;
            replacement_seed=UInt64(71), attempt=Returns(false),
            scheduler=serial, workers=1, audit_schema="test-fixed-contest-v1",
            failure_policy=:replace, max_replacement_rounds=2)
    catch error
        error
    end
    @test failure isa CD.ContestCompletionError
    @test failure.result.audit.failed_contests == 3
    @test failure.result.audit.total_failed_attempts == 9
    @test length(failure.result.audit.rounds) == 3
    @test occursin("budget exhausted", sprint(showerror, failure))

    @test_throws InterruptException CD.complete_contest_design(config;
        replacement_seed=UInt64(71), attempt=spec -> throw(InterruptException()),
        scheduler=serial, workers=1, audit_schema="test-fixed-contest-v1")
end

@testset "Nested IID contest design" begin
    defaults = CD.ContestDesignConfig()
    @test defaults.n_contests == 100
    @test defaults.n_paths == 100
    @test defaults.horizon == 50
    @test defaults.floor == 3.0
    @test defaults.bounds.beta == (0.90, 0.995)
    @test defaults.bounds.gamma == (0.5, 5.0)
    @test defaults.bounds.initial_cash == (3.0, 50.0)
    @test defaults.parameter_seed != defaults.evaluation_seed

    # Check the full production outer size and exact one-observation-per-stratum
    # construction without allocating the complete hidden-path collection.
    full_design = CD.build_contest_design(defaults)
    @test length(full_design) == 100
    @test stratum_indices(getfield.(full_design, :beta),
        defaults.bounds.beta) == collect(0:99)
    @test stratum_indices(getfield.(full_design, :gamma),
        defaults.bounds.gamma) == collect(0:99)
    @test stratum_indices(getfield.(full_design, :initial_cash),
        defaults.bounds.initial_cash) == collect(0:99)
    all_panel_seeds = vcat(
        [CD.panel_seed(spec, defaults, :baseline) for spec in full_design],
        [CD.panel_seed(spec, defaults, :high_variance) for spec in full_design])
    @test length(unique(all_panel_seeds)) == 200

    config = CD.ContestDesignConfig(n_contests=20, n_paths=11, horizon=7)
    design = CD.build_contest_design(config)
    repeated = CD.build_contest_design(config)

    @test design == repeated
    @test length(design) == config.n_contests
    @test all(config.bounds.beta[1] < spec.beta < config.bounds.beta[2]
              for spec in design)
    @test all(config.bounds.gamma[1] < spec.gamma < config.bounds.gamma[2]
              for spec in design)
    @test all(config.bounds.initial_cash[1] < spec.initial_cash <
              config.bounds.initial_cash[2] for spec in design)
    @test all(!hasproperty(spec, :baseline_shock_seed) &&
              !hasproperty(spec, :high_variance_shock_seed) for spec in design)

    baseline = CD.predraw_innovations(design[1], config, :baseline)
    baseline_again = CD.predraw_innovations(design[1], config, :baseline)
    next_contest = CD.predraw_innovations(design[2], config, :baseline)
    high_variance = CD.predraw_innovations(design[1], config, :high_variance)
    @test size(baseline) == (config.n_paths, config.horizon - 1)
    @test baseline == baseline_again
    @test baseline != next_contest
    @test baseline != high_variance
    @test all(isfinite, baseline)

    # Parameter and evaluation randomization are separate: changing only the
    # evaluation root leaves the outer design fixed but changes every path block.
    new_evaluation_seed = CD.ContestDesignConfig(
        n_contests=20, n_paths=11, horizon=7,
        evaluation_seed=config.evaluation_seed + UInt64(1))
    @test CD.build_contest_design(new_evaluation_seed) == design
    @test CD.predraw_innovations(
        design[1], new_evaluation_seed, :baseline) != baseline

    @test_throws ArgumentError CD.panel_seed(design[1], config, :unknown)
    @test_throws ArgumentError CD.predraw_innovations(
        CD.ContestSpec(config.n_contests + 1, 0.95, 2.0, 5.0),
        config, :baseline)
    @test_throws ArgumentError CD.ContestDesignConfig(
        parameter_seed=UInt64(7), evaluation_seed=UInt64(7)) |>
        CD.build_contest_design
    @test_throws ArgumentError CD.ContestDesignConfig(
        floor=3.0,
        bounds=CD.ParameterBounds(initial_cash=(2.0, 10.0))) |>
        CD.build_contest_design
end
