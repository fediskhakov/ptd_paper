# Tests for src/simulation/ar1/runner.jl.
using Test
import Ptd

const AR1Runner = Ptd.AR1

struct AR1FakePolicy{C,P,R}
    method::Symbol
    config::C
    process::P
    rule::R
end

function (policy::AR1FakePolicy)(observed)
    terminal = observed.date == policy.config.terminal.final_date
    consumption = terminal ? observed.cash : 0.4observed.cash
    return Ptd.ActionResult(consumption, observed.cash - consumption,
        nothing, policy.method, "ar1-trace-test-policy")
end

_ar1_bits(values) = Tuple(reinterpret(UInt64,
    Float64(value isa Tuple ? only(value) : value)) for value in values)

function _ar1_fake_predraw(spec, design, panel)
    panel === :ar1 || error("unexpected AR(1) panel")
    return [0.01sin(13spec.id + 5path + date)
        for path in 1:design.n_paths, date in 1:design.horizon-1]
end

_ar1_fake_build(method, config, process, rule, grids, g2options, osaoptions) =
    AR1FakePolicy(method, config, process, rule)

function _ar1_fake_evaluate(policy::AR1FakePolicy, method, config, process,
                            initial, innovations, action_audit)
    offset = method === :ptd_sceq ? 0.0 : method === :ptd_osa ? 0.1 : 0.2
    utility = [config.beta + offset + 0.001sum(row)
        for row in eachrow(innovations)]
    return Ptd.PolicyPathEvaluation(method, utility,
        zeros(Int, length(utility)), zeros(length(utility)))
end

function _ar1_trace_evaluate(policy::AR1FakePolicy, method, config, process,
                             initial, innovations, action_audit)
    return Ptd.evaluate_policy_paths(policy, method, config, process,
        initial, innovations; action_audit)
end

function _ar1_fake_audit(policy::AR1FakePolicy)
    policy.method === :ptd_sceq &&
        return (method=:ptd_sceq, uses_quadrature=false)
    return (method=policy.method, uses_quadrature=true,
        nodes_per_dimension=policy.rule.nodes_per_dimension,
        base_nodes=policy.rule.nodes,
        base_weights=policy.rule.weights,
        base_node_bits=_ar1_bits(policy.rule.nodes),
        base_weight_bits=_ar1_bits(policy.rule.weights))
end

function ar1_fake_hooks(; evaluate=_ar1_fake_evaluate,
                        build_policy=_ar1_fake_build)
    return AR1Runner.AR1RunnerHooks(
        predraw=_ar1_fake_predraw,
        build_policy=build_policy,
        evaluate=evaluate,
        diagnostics=policy -> (method=policy.method, calls=1),
        reset_diagnostics! = Returns(nothing),
        quadrature_audit=_ar1_fake_audit,
        validate_policy=(policy, method, config, process) ->
            policy.method == method ? nothing : error("method mismatch"),
    )
end

function ar1_tiny_grid()
    return Ptd.AR1G2ContestGridConfiguration(
        :ar1_runner_test_grid_v2,
        [0.0, 4.0, 12.0],
        [-1.0, 0.0, 1.0],
        [1.0, 6.0, 15.0],
        [-1.0, 0.0, 1.0])
end

function ar1_tiny_config(; n_contests=3, n_paths=4)
    return AR1Runner.AR1ContestRunConfig(
        n_contests=n_contests,
        n_paths=n_paths,
        horizon=3,
        return_rate=0.02,
        cash_floor=1.0,
        bounds=Ptd.ParameterBounds(
            (0.92, 0.98), (1.0, 2.0), (1.0, 4.0)),
        g2egm_grids=ar1_tiny_grid(),
        ptd_osa_options=Ptd.PTDOSAOptions(direct_iterations=8),
    )
end

@testset "AR(1) contest runner" begin
    production = AR1Runner.production_run_config()
    @test production.n_contests == production.n_paths == 100
    @test production.methods == (:ptd_sceq, :ptd_osa, :g2egm)
    @test production.parameter_seed != production.replacement_parameter_seed
    @test length(production.g2egm_grids.asset_grid) == 1_601
    @test length(production.g2egm_grids.post_log_income_grid) == 61
    @test length(production.g2egm_grids.common_cash_grid) == 1_201
    @test length(production.g2egm_grids.common_log_income_grid) == 41
    scientific = AR1Runner.scientific_configuration(production)
    @test scientific.outer_design.completion_rule ===
        :retain_original_configurations_and_failures
    @test !hasproperty(scientific.outer_design, :maximum_replacement_rounds)
    @test !hasproperty(scientific.hidden_evaluation, :chunk_size)

    # Production accepts the runtime Julia thread count while retaining the
    # prohibition on injected solvers, even when that count is not 24.
    @test_throws "injected hooks are restricted" AR1Runner.run_welfare_contest(
        production; workers=Threads.nthreads(), hooks=ar1_fake_hooks())
    @test_throws "workers must match the active Julia thread count" AR1Runner.run_welfare_contest(
        production; workers=Threads.nthreads() + 1)

    config = ar1_tiny_config()
    welfare = mktempdir() do root
        result = AR1Runner.run_welfare_contest(config;
            root,
            workers=1,
            scheduler=AR1Runner.serial_scheduler,
            hooks=ar1_fake_hooks(),
            gitpath=pkgdir(Ptd),
            lock_run=true)
        @test isempty(readdir(root))
        result
    end
    @test length(welfare.results) == config.n_contests
    @test welfare.sampler_audit.completed_contests == config.n_contests
    @test welfare.sampler_audit.replacement_count == 0
    threaded_welfare = AR1Runner.run_welfare_contest(config;
        workers=Threads.nthreads(), scheduler=AR1Runner.threaded_scheduler,
        hooks=ar1_fake_hooks(), lock_run=false)
    @test threaded_welfare.design == welfare.design
    @test [result.evaluation.utility
        for contest in threaded_welfare.results
        for result in contest.panels[:ar1].methods] ==
        [result.evaluation.utility
         for contest in welfare.results
         for result in contest.panels[:ar1].methods]
    for contest in welfare.results
        panel = contest.panels[:ar1]
        @test panel.policy_domain.gate_passed
        @test length(panel.methods) == 3
        @test all(length(result.evaluation.utility) == config.n_paths
            for result in panel.methods)
    end

    attempts = Dict{Int,Int}()
    certify = function (spec)
        attempts[spec.id] = get(attempts, spec.id, 0) + 1
        spec.id == 1 && attempts[spec.id] == 1 && return false
        spec.id == 3 && attempts[spec.id] <= 2 && return false
        true
    end
    selection = AR1Runner.sample_ar1_contest_design(config;
        certify_g2egm=certify,
        workers=1,
        scheduler=AR1Runner.serial_scheduler)
    @test length(selection.design) == config.n_contests
    @test selection.audit.completed_contests == 1
    @test selection.audit.failed_contests == 2
    @test selection.audit.replacement_count == 0
    @test selection.design == welfare.design
    @test attempts == Dict(1 => 1, 2 => 1, 3 => 1)
    @test selection.audit.rounds[1].failed_slots == (1, 3)
    @test length(selection.audit.rounds) == 1

    failed_once = Ref(false)
    evaluate_with_failure = function (policy, method, model, process,
                                      initial, innovations, audit)
        if !failed_once[] && method === :g2egm
            failed_once[] = true
            error("deliberate full-contest failure")
        end
        _ar1_fake_evaluate(policy, method, model, process,
            initial, innovations, audit)
    end
    replaced = AR1Runner.run_welfare_contest(config;
        root=mktempdir(), workers=1,
        scheduler=AR1Runner.serial_scheduler,
        hooks=ar1_fake_hooks(evaluate=evaluate_with_failure),
        gitpath=pkgdir(Ptd), lock_run=false)
    @test replaced.sampler_audit.completed_contests == config.n_contests - 1
    @test replaced.sampler_audit.failed_contests == 1
    @test replaced.sampler_audit.replacement_count == 0
    @test replaced.design == welfare.design
    @test replaced.results[1] isa Ptd.ContestFailure
    context = AR1Runner._capture_run_context(config, pkgdir(Ptd))
    partial_timing = AR1Runner._run_timing(config, ar1_fake_hooks(),
        (design=replaced.design, plain_design=replaced.outer_design,
         audit=replaced.sampler_audit);
        workers=1, scheduler=AR1Runner.serial_scheduler, context)
    @test partial_timing.timed_contests == config.n_contests - 1
    @test partial_timing.failed_contests == 1
    @test partial_timing.outer_design_fingerprint ==
        Ptd.stable_fingerprint(replaced.outer_design)

    all_failed = AR1Runner.run_welfare_contest(config;
        root=mktempdir(), workers=1, scheduler=AR1Runner.serial_scheduler,
        hooks=ar1_fake_hooks(evaluate=(args...) -> error("solver failed")),
        gitpath=pkgdir(Ptd), lock_run=false)
    @test all(result -> result isa Ptd.ContestFailure, all_failed.results)
    empty_timing = AR1Runner._run_timing(config, ar1_fake_hooks(),
        (design=all_failed.design, plain_design=all_failed.outer_design,
         audit=all_failed.sampler_audit);
        workers=1, scheduler=AR1Runner.serial_scheduler, context)
    @test empty_timing.timed_contests == 0
    @test empty_timing.failed_contests == config.n_contests
    @test all(row.status === :no_successful_contests && row.total_seconds == 0.0
        for row in empty_timing.totals)

    timing_selection = (design=welfare.design,
        plain_design=welfare.outer_design,
        audit=welfare.sampler_audit)
    context = AR1Runner._capture_run_context(config, pkgdir(Ptd))
    timing = AR1Runner._run_timing(config, ar1_fake_hooks(), timing_selection;
        workers=1, scheduler=AR1Runner.serial_scheduler, context)
    @test length(timing.totals) == 3
    @test all(row -> isfinite(row.total_seconds) && row.total_seconds >= 0,
        timing.totals)
end

@testset "AR(1) run context without git relies on the input tree" begin
    config = AR1Runner.production_run_config()
    mktempdir() do root
        fixture = joinpath(root, "src", "fixture.jl")
        mkpath(dirname(fixture))
        write(fixture, "# production provenance fixture\n")
        # As in an unpacked ZIP copy: no repository here or above.
        withenv("GIT_CEILING_DIRECTORIES" => dirname(root)) do
            context = AR1Runner._capture_run_context(config, root)
            @test context.source.commit == Ptd.UNVERSIONED_COMMIT
            @test !context.source.dirty
            validated = AR1Runner.validate_production_environment(config;
                project_root=root, context)
            @test validated.source.commit == Ptd.UNVERSIONED_COMMIT
            @test AR1Runner._assert_context!(context, "unchanged copy") ===
                context
            # The commit cannot change, but the SHA-256 input tree does.
            write(fixture, "# changed source\n")
            @test_throws "computational inputs changed" (
                AR1Runner._assert_context!(context, "changed copy"))
        end
    end
end
