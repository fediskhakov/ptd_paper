# Tests for src/simulation/ar1/artifact.jl.
using Test
using JLD2
import Ptd

const AR1Processing = Ptd.AR1

@testset "AR(1) single contest artifact" begin
    config = ar1_tiny_config(n_contests=2, n_paths=4)
    hooks = ar1_fake_hooks()
    context = AR1Processing._capture_run_context(config, pkgdir(Ptd))
    welfare = AR1Processing._run_welfare_unlocked(config, hooks;
        workers=1, scheduler=AR1Processing.serial_scheduler, context)
    selection = (design=welfare.design,
        plain_design=welfare.outer_design,
        audit=welfare.sampler_audit)
    timing = AR1Processing._run_timing(config, hooks, selection;
        workers=1, scheduler=AR1Processing.serial_scheduler, context)

    mktempdir() do root
        path = joinpath(root, "ar1-contest-welfare.jld2")
        publication = AR1Processing.publish_ar1_contest(
            config, welfare, timing;
            output_path=path,
            context,
            compress=false)
        @test publication.path == path
        @test readdir(root) == ["ar1-contest-welfare.jld2"]
        artifact = AR1Processing.load_ar1_contest_welfare(path)
        metadata = artifact["metadata"]
        @test metadata.artifact_schema ==
            AR1Processing.AR1_CONTEST_ARTIFACT_SCHEMA_VERSION
        @test metadata.n_contests == 2
        @test metadata.n_paths == 4
        @test metadata.utility_payload_shape == (4, 3, 2, 1)
        @test artifact["run_audit"].completion.completed_contests == 2
        @test size(artifact["pathwise_utilities"]) == (4, 3, 2, 1)
        @test length(artifact["contest_summaries"]) == 2
        @test length(artifact["panel_aggregates"]) == 1
        @test length(artifact["timings"]) == 3

        damaged = JLD2.load(path)
        damaged["pathwise_utilities"][1] += 1.0
        JLD2.jldopen(path, "w"; compress=false) do file
            for (name, value) in damaged
                file[name] = value
            end
        end
        @test_throws ArgumentError AR1Processing.load_ar1_contest_welfare(path)
    end
end

@testset "archived AR(1) summaries authenticate before numerical reconstruction" begin
    path = joinpath(pkgdir(Ptd), "data", "ar1-contest-welfare.jld2")
    artifact = AR1Processing.load_ar1_contest_welfare(path)
    @test artifact["metadata"].n_contests == 100
    summaries = artifact["contest_summaries"]
    rounded = deepcopy(summaries)
    rounded[1].expected_utility[1] = nextfloat(rounded[1].expected_utility[1])
    @test Ptd._contest_summary_reproduces(summaries, rounded)
    changed_label = deepcopy(summaries)
    changed_label[1] = merge(changed_label[1], (; classification=:changed))
    @test !Ptd._contest_summary_reproduces(summaries, changed_label)
    changed_calibration = deepcopy(summaries)
    changed_calibration[1] = merge(changed_calibration[1],
        (; beta=nextfloat(changed_calibration[1].beta)))
    @test !Ptd._contest_summary_reproduces(summaries, changed_calibration)
    changed_mean = deepcopy(summaries)
    changed_mean[1].expected_utility[1] += 1e-8
    @test !Ptd._contest_summary_reproduces(summaries, changed_mean)
    damaged = copy(artifact)
    damaged["contest_summaries"] = rounded
    @test_throws "contest summary fingerprint mismatch" Ptd._validate_contest_artifact(damaged)
end

@testset "AR(1) artifact retains failed original configurations" begin
    config = ar1_tiny_config(n_contests=2, n_paths=4)
    calls = Ref(0)
    evaluate = function (args...)
        calls[] += 1
        calls[] == 2 && error("injected original AR(1) failure")
        return _ar1_fake_evaluate(args...)
    end
    hooks = ar1_fake_hooks(; evaluate)
    context = AR1Processing._capture_run_context(config, pkgdir(Ptd))
    welfare = AR1Processing._run_welfare_unlocked(config, hooks;
        workers=1, scheduler=AR1Processing.serial_scheduler, context)
    selection = (design=welfare.design, plain_design=welfare.outer_design,
        audit=welfare.sampler_audit)
    timing = AR1Processing._run_timing(config, hooks, selection;
        workers=1, scheduler=AR1Processing.serial_scheduler, context)
    mktempdir() do root
        path = joinpath(root, "ar1-failures.jld2")
        AR1Processing.publish_ar1_contest(config, welfare, timing;
            output_path=path, context, compress=false)
        artifact = AR1Processing.load_ar1_contest_welfare(path)
        @test length(artifact["outer_design"]) == 2
        @test size(artifact["pathwise_utilities"]) == (4, 3, 1, 1)
        @test only(artifact["run_audit"].failures).contest_id == 1
        @test only(artifact["contest_summaries"]).contest_id == 2
        rendered = AR1Processing.render_ar1_contest(path;
            output_root=joinpath(root, "tables"), compress=false)
        @test all(row -> row.timing_paths == 4,
            rendered.artifact["timing_rows"])
        @test occursin("Original solver failures: 1/2",
            read(rendered.table_paths[:outcome_shares], String))
    end
end
