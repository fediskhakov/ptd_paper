# Tests for src/simulation/iid/artifact.jl.
using Test
using JLD2
import Ptd

const IIDProcessing = Ptd.IID

@testset "IID single contest artifact" begin
    config = iid_tiny_config(n_contests=2, n_paths=4)
    hooks = iid_fake_hooks()
    context = IIDProcessing._capture_run_context(config, pkgdir(Ptd))
    welfare = IIDProcessing._run_welfare_unlocked(config, hooks;
        workers=1, scheduler=IIDProcessing.serial_scheduler, context)
    timing = IIDProcessing._run_timing_unlocked(config, hooks;
        design=welfare.design,
        plain_design=welfare.outer_design,
        sampler_audit=welfare.sampler_audit,
        workers=1,
        scheduler=IIDProcessing.serial_scheduler,
        context)

    mktempdir() do root
        path = joinpath(root, "iid-contest.jld2")
        publication = IIDProcessing.publish_iid_contest(
            config, welfare, timing;
            output_path=path,
            context,
            compress=false)
        @test publication.path == path
        @test readdir(root) == ["iid-contest.jld2"]
        artifact = IIDProcessing.load_iid_contest(path)
        metadata = artifact["metadata"]
        @test metadata.artifact_schema ==
            IIDProcessing.IID_CONTEST_ARTIFACT_SCHEMA_VERSION
        @test metadata.n_contests == 2
        @test metadata.n_paths == 4
        @test metadata.utility_payload_shape == (4, 3, 2, 2)
        @test artifact["run_audit"].completion.completed_contests == 2
        @test size(artifact["pathwise_utilities"]) == (4, 3, 2, 2)
        @test length(artifact["contest_summaries"]) == 4
        @test length(artifact["panel_aggregates"]) == 2
        @test length(artifact["timings"]) == 6
        @test all(startswith(row.relative_path, "embedded/timings/")
            for row in artifact["timings"])

        damaged = JLD2.load(path)
        damaged["pathwise_utilities"][1] += 1.0
        JLD2.jldopen(path, "w"; compress=false) do file
            for (name, value) in damaged
                file[name] = value
            end
        end
        @test_throws ArgumentError IIDProcessing.load_iid_contest(path)
    end
end

@testset "IID artifact retains failed original configurations" begin
    for all_fail in (false, true)
        config = iid_tiny_config(n_contests=2, n_paths=4)
        calls = Ref(0)
        evaluate = function (args...)
            calls[] += 1
            (all_fail || calls[] == 2) && error("injected original-configuration failure")
            return _iid_fake_evaluate(args...)
        end
        hooks = iid_fake_hooks(; evaluate)
        context = IIDProcessing._capture_run_context(config, pkgdir(Ptd))
        welfare = IIDProcessing._run_welfare_unlocked(config, hooks;
            workers=1, scheduler=IIDProcessing.serial_scheduler, context)
        timing = IIDProcessing._run_timing_unlocked(config, hooks;
            design=welfare.design, plain_design=welfare.outer_design,
            sampler_audit=welfare.sampler_audit, workers=1,
            scheduler=IIDProcessing.serial_scheduler, context)
        successful = all_fail ? 0 : 1
        mktempdir() do root
            path = joinpath(root, "iid-failures.jld2")
            IIDProcessing.publish_iid_contest(config, welfare, timing;
                output_path=path, context, compress=false)
            artifact = IIDProcessing.load_iid_contest(path)
            @test length(artifact["outer_design"]) == 2
            @test artifact["metadata"].n_contests == 2
            @test size(artifact["pathwise_utilities"]) == (4, 3, successful, 2)
            @test length(artifact["run_audit"].failures) == 2 - successful
            @test artifact["run_audit"].failures[1].contest_id == 1
            @test occursin("injected", artifact["run_audit"].failures[1].message)
            @test length(artifact["contest_summaries"]) == 2successful
            if !all_fail
                @test all(row -> row.contest_id == 2, artifact["contest_summaries"])
                rendered = IIDProcessing.render_iid_contest(path;
                    output_root=joinpath(root, "tables"), compress=false)
                @test all(row -> row.n_contests == 1 &&
                    row.sample.original_failed_contests == 1,
                    rendered.artifact["outcome_rows"])
                @test all(row -> row.timing_paths == 4,
                    rendered.artifact["timing_rows"])
                @test occursin("Original solver failures: 1/2",
                    read(rendered.table_paths[:outcome_shares], String))
                @test occursin("conditional on successful solver completion",
                    read(rendered.table_paths[:outcome_shares], String))
            end
            table = IIDProcessing.render_iid_paper_table(path;
                output_path=joinpath(root, "paper-table.tex"))
            @test occursin("Original solver failures: $(2-successful)/2",
                read(table, String))
            all_fail && @test occursin("No welfare comparisons", read(table, String))
        end
    end
end

@testset "saved IID paper results remain readable and authenticated" begin
    path = joinpath(pkgdir(Ptd), "data", "iid-contest.jld2")
    artifact = IIDProcessing.load_iid_contest(path)
    @test artifact["metadata"].n_contests == 100
    @test artifact["methods"] == [:ptd_sceq, :ptd_osa, :dcegm]
    @test length(artifact["contest_summaries"]) == 200
    @test IIDProcessing._completion_status(artifact).original_failed_contests == 0
    mktempdir() do root
        table = IIDProcessing.render_iid_paper_table(path;
            output_path=joinpath(root, "iid-saved.tex"))
        @test occursin("DC-EGM", read(table, String))
        @test occursin("Original solver failures: 0/100", read(table, String))
        damaged = JLD2.load(path)
        damaged["contest_summaries"][1].expected_utility[1] += 1.0
        altered_path = joinpath(root, "tampered-saved.jld2")
        JLD2.jldopen(altered_path, "w"; compress=false) do file
            for (name, value) in damaged
                file[name] = value
            end
        end
        @test_throws ArgumentError IIDProcessing.load_iid_contest(altered_path)
    end
end
