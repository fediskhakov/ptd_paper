# Tests for src/simulation/iid/runner.jl.
using Test
import Ptd

const IIDRunner = Ptd.IID

struct IIDFakePolicy{C,P,R}
    method::Symbol
    config::C
    process::P
    rule::R
end

_iid_bits(values) = Tuple(reinterpret(UInt64,
    Float64(value isa Tuple ? only(value) : value)) for value in values)

function _iid_fake_predraw(spec, design, panel)
    offset = panel === :baseline ? 1000.0 : 2000.0
    return [sin(offset + 17spec.id + 3path + date)
        for path in 1:design.n_paths, date in 1:design.horizon-1]
end

_iid_fake_build(method, config, process, rule, assets, options) =
    IIDFakePolicy(method, config, process, rule)

function _iid_fake_evaluate(policy::IIDFakePolicy, method, config, process,
                            initial, innovations, action_audit)
    offset = method === :ptd_sceq ? 0.0 : method === :ptd_osa ? 0.1 : 0.2
    utility = [config.beta + offset + 0.001sum(row)
        for row in eachrow(innovations)]
    return Ptd.PolicyPathEvaluation(method, utility,
        zeros(Int, length(utility)), zeros(length(utility)))
end

function _iid_fake_audit(policy::IIDFakePolicy)
    policy.method === :ptd_sceq &&
        return (method=:ptd_sceq, uses_quadrature=false)
    return (method=policy.method, uses_quadrature=true,
        nodes_per_dimension=policy.rule.nodes_per_dimension,
        base_nodes=policy.rule.nodes,
        base_weights=policy.rule.weights,
        base_node_bits=_iid_bits(policy.rule.nodes),
        base_weight_bits=_iid_bits(policy.rule.weights))
end

function iid_fake_hooks(; evaluate=_iid_fake_evaluate,
                         predraw=_iid_fake_predraw)
    return IIDRunner.RunnerHooks(
        predraw=predraw,
        build_policy=_iid_fake_build,
        evaluate=evaluate,
        diagnostics=policy -> (method=policy.method, calls=1),
        reset_diagnostics! = Returns(nothing),
        quadrature_audit=_iid_fake_audit,
        validate_policy=(policy, method, config, process) ->
            policy.method == method ? nothing : error("method mismatch"),
    )
end

@testset "IID production accepts the selected Julia thread count" begin
    config = IIDRunner.production_run_config()
    IIDRunner.LinearAlgebra.BLAS.set_num_threads(1)
    mktempdir() do root
        mkpath(joinpath(root, "src"))
        write(joinpath(root, "src", "fixture.jl"), "# production provenance fixture\n")
        run(`git init -q $root`)
        run(Cmd(`git add src/fixture.jl`; dir=root))
        # A global commit-signing setting must not break this scratch commit.
        git_options = ["-c", "user.name=Replication-Test",
            "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false"]
        run(Cmd(`git $git_options commit -q -m initial`; dir=root))
        context = IIDRunner._capture_run_context(config, root)
        validated = IIDRunner.validate_production_environment(config;
            project_root=root, context)
        @test !validated.source.dirty
        @test validated.context.simulation_metadata.runtime.threads == Threads.nthreads()
        @test IIDRunner.LinearAlgebra.BLAS.get_num_threads() == 1
        @test IIDRunner._validate_public_production_call(
            config, Threads.nthreads(), IIDRunner.threaded_scheduler,
            IIDRunner.RunnerHooks(); gitpath=root, context).config === config
        @test_throws ArgumentError IIDRunner._validate_public_production_call(
            config, Threads.nthreads() + 1, IIDRunner.threaded_scheduler,
            IIDRunner.RunnerHooks(); gitpath=root, context)
        @test_throws ArgumentError IIDRunner._validate_public_production_call(
            config, Threads.nthreads(), IIDRunner.serial_scheduler,
            IIDRunner.RunnerHooks(); gitpath=root, context)
        @test_throws ArgumentError IIDRunner._validate_public_production_call(
            config, Threads.nthreads(), IIDRunner.threaded_scheduler,
            iid_fake_hooks(); gitpath=root, context)
        write(joinpath(root, "src", "fixture.jl"), "# changed source\n")
        @test_throws ArgumentError IIDRunner.validate_production_environment(
            config; project_root=root)
    end
end

@testset "IID run context without git relies on the input tree" begin
    config = IIDRunner.production_run_config()
    mktempdir() do root
        fixture = joinpath(root, "src", "fixture.jl")
        mkpath(dirname(fixture))
        write(fixture, "# production provenance fixture\n")
        # As in an unpacked ZIP copy: no repository here or above.
        withenv("GIT_CEILING_DIRECTORIES" => dirname(root)) do
            context = IIDRunner._capture_run_context(config, root)
            @test context.source.commit == Ptd.UNVERSIONED_COMMIT
            @test !context.source.dirty
            validated = IIDRunner.validate_production_environment(config;
                project_root=root, context)
            @test validated.source.commit == Ptd.UNVERSIONED_COMMIT
            @test IIDRunner._assert_run_context_unchanged!(
                context, "unchanged copy") === context
            # The commit cannot change, but the SHA-256 input tree does.
            write(fixture, "# changed source\n")
            @test_throws "computational inputs changed" (
                IIDRunner._assert_run_context_unchanged!(
                    context, "changed copy"))
        end
    end
end

function iid_tiny_config(; n_contests=3, n_paths=4)
    return IIDRunner.IIDContestRunConfig(
        n_contests=n_contests,
        n_paths=n_paths,
        horizon=3,
        return_rate=0.02,
        cash_floor=1.0,
        bounds=Ptd.ParameterBounds(
            (0.92, 0.98), (1.0, 2.0), (1.0, 4.0)),
        dcegm_asset_maximum=12.0,
        dcegm_asset_points=5,
        ptd_osa_options=Ptd.PTDOSAOptions(direct_iterations=8),
    )
end

@testset "IID contest runner" begin
    production = IIDRunner.production_run_config()
    @test production.n_contests == production.n_paths == 100
    @test production.methods == (:ptd_sceq, :ptd_osa, :dcegm)
    @test production.parameter_seed != production.replacement_parameter_seed
    scientific = IIDRunner.scientific_configuration(production)
    @test scientific.outer_design.completion_rule ===
        :retain_original_configurations_and_failures
    @test scientific.hidden_evaluation.n_paths == 100
    @test !hasproperty(scientific.hidden_evaluation, :chunk_size)

    config = iid_tiny_config()
    welfare = mktempdir() do root
        result = IIDRunner.run_welfare_contest(config;
            root,
            workers=1,
            scheduler=IIDRunner.serial_scheduler,
            hooks=iid_fake_hooks(),
            gitpath=pkgdir(Ptd),
            lock_run=true)
        @test isempty(readdir(root))
        result
    end
    @test length(welfare.results) == config.n_contests
    @test length(welfare.outer_design) == config.n_contests
    @test welfare.sampler_audit.completed_contests == config.n_contests
    @test welfare.sampler_audit.replacement_count == 0
    for contest in welfare.results
        @test contest.contest_id in 1:config.n_contests
        @test Set(keys(contest.panels)) == Set((:baseline, :high_variance))
        for panel in values(contest.panels)
            @test length(panel.methods) == 3
            @test all(length(result.evaluation.utility) == config.n_paths
                for result in panel.methods)
        end
    end

    failed_once = Ref(false)
    fail_one_configuration = function (policy, method, model, process,
                                       initial, innovations, audit)
        if !failed_once[] && method === :ptd_osa
            failed_once[] = true
            error("deliberate complete-contest failure")
        end
        return _iid_fake_evaluate(policy, method, model, process,
            initial, innovations, audit)
    end
    replaced = IIDRunner.run_welfare_contest(config;
        root=mktempdir(), workers=1, scheduler=IIDRunner.serial_scheduler,
        hooks=iid_fake_hooks(evaluate=fail_one_configuration),
        gitpath=pkgdir(Ptd), lock_run=false)
    @test replaced.sampler_audit.completed_contests == config.n_contests - 1
    @test replaced.sampler_audit.failed_contests == 1
    @test replaced.sampler_audit.replacement_count == 0
    @test replaced.design == welfare.design
    @test replaced.results[1] isa Ptd.ContestFailure
    @test replaced.sampler_audit.rounds[1].failed_slots == (1,)
    context = IIDRunner._capture_run_context(config, pkgdir(Ptd))
    partial_timing = IIDRunner._run_timing_unlocked(config, iid_fake_hooks();
        design=replaced.design, plain_design=replaced.outer_design,
        sampler_audit=replaced.sampler_audit, workers=1,
        scheduler=IIDRunner.serial_scheduler, context)
    @test partial_timing.timed_contests == config.n_contests - 1
    @test partial_timing.failed_contests == 1
    @test partial_timing.outer_design_fingerprint ==
        Ptd.stable_fingerprint(replaced.outer_design)

    all_failed = IIDRunner.run_welfare_contest(config;
        root=mktempdir(), workers=1, scheduler=IIDRunner.serial_scheduler,
        hooks=iid_fake_hooks(evaluate=(args...) -> error("solver failed")),
        gitpath=pkgdir(Ptd), lock_run=false)
    @test all(result -> result isa Ptd.ContestFailure, all_failed.results)
    empty_timing = IIDRunner._run_timing_unlocked(config, iid_fake_hooks();
        design=all_failed.design, plain_design=all_failed.outer_design,
        sampler_audit=all_failed.sampler_audit, workers=1,
        scheduler=IIDRunner.serial_scheduler, context)
    @test empty_timing.timed_contests == 0
    @test empty_timing.failed_contests == config.n_contests
    @test all(row.status === :no_successful_contests && row.total_seconds == 0.0
        for row in empty_timing.totals)

    timing = IIDRunner.run_timing_pass(config;
        design=welfare.design,
        workers=1,
        scheduler=IIDRunner.serial_scheduler,
        hooks=iid_fake_hooks(),
        gitpath=pkgdir(Ptd))
    @test length(timing.totals) == 6
    @test all(row -> isfinite(row.total_seconds) && row.total_seconds >= 0,
        timing.totals)
end

@testset "IID seeded accuracy is invariant to outer worker count" begin
    config = iid_tiny_config()
    # Use the real per-contest/per-panel random streams; mock only policy work.
    hooks = iid_fake_hooks(predraw=Ptd.predraw_innovations)
    context = IIDRunner._capture_run_context(config, pkgdir(Ptd))
    serial = IIDRunner._run_welfare_unlocked(config, hooks;
        workers=1, scheduler=IIDRunner.serial_scheduler, context)
    for workers in unique((1, Threads.nthreads(), 3))
        threaded = IIDRunner._run_welfare_unlocked(config, hooks;
            workers, scheduler=IIDRunner.threaded_scheduler, context)
        @test threaded.design == serial.design
        @test threaded.outer_design == serial.outer_design
        @test threaded.sampler_audit == serial.sampler_audit
        @test threaded.scientific_fingerprint == serial.scientific_fingerprint
        for (actual, expected) in zip(threaded.results, serial.results)
            @test actual.contest_id == expected.contest_id
            for panel in (:baseline, :high_variance)
                @test actual.panels[panel].block_fingerprint ==
                    expected.panels[panel].block_fingerprint
                @test getproperty.(actual.panels[panel].methods,
                    :evaluation_fingerprint) == getproperty.(
                    expected.panels[panel].methods, :evaluation_fingerprint)
            end
        end
    end
end
