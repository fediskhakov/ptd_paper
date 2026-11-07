# Recompute heterogeneous observations independently of the historical replay API.
using Test
using JLD2
import Ptd

const UpdatedAR1 = Ptd.AR1

struct UpdatedPathFakePolicy{C,P,R}
    method::Symbol
    config::C
    process::P
    rule::R
    fraction::Float64
end

function (policy::UpdatedPathFakePolicy)(state)
    consumption = state.date == policy.config.terminal.final_date ? state.cash :
        policy.fraction * state.cash
    return Ptd.ActionResult(consumption, state.cash - consumption, nothing,
        policy.method, "updated-path-test-policy")
end

_updated_bits(values) = Tuple(reinterpret(UInt64,
    Float64(value isa Tuple ? only(value) : value)) for value in values)

function _updated_test_hooks(fraction; calls=Ref(0), changed_shocks=false)
    build = (method, config, process, rule, grids, g2, osa) -> begin
        calls[] += 1
        UpdatedPathFakePolicy(method, config, process, rule, fraction)
    end
    quadrature = policy -> policy.method === :ptd_sceq ?
        (method=:ptd_sceq, uses_quadrature=false) :
        (method=policy.method, uses_quadrature=true,
         nodes_per_dimension=policy.rule.nodes_per_dimension,
         base_nodes=policy.rule.nodes, base_weights=policy.rule.weights,
         base_node_bits=_updated_bits(policy.rule.nodes),
         base_weight_bits=_updated_bits(policy.rule.weights))
    predraw = (spec, config, panel) -> begin
        shocks = UpdatedAR1.predraw_ar1_innovations(spec, config, panel)
        changed_shocks && (shocks[1, 1] += 0.01)
        shocks
    end
    return UpdatedAR1.AR1RunnerHooks(;
        predraw, build_policy=build,
        evaluate=(policy, method, config, process, initial, shocks, audit) ->
            Ptd.evaluate_policy_paths(policy, method, config, process, initial,
                shocks; action_audit=audit),
        diagnostics=policy -> (method=policy.method, checked=true),
        reset_diagnostics! = Returns(nothing), quadrature_audit=quadrature,
        validate_policy=(policy, method, config, process) ->
            policy.method == method ? nothing : error("wrong method"))
end

function _updated_test_config(; return_rate=0.02)
    grids = Ptd.AR1G2ContestGridConfiguration(:updated_path_test_grid,
        [0.0, 4.0, 12.0], [-1.0, 0.0, 1.0],
        [1.0, 6.0, 15.0], [-1.0, 0.0, 1.0])
    return UpdatedAR1.AR1ContestRunConfig(n_contests=3, n_paths=4, horizon=3,
        return_rate=return_rate, cash_floor=1.0,
        bounds=Ptd.ParameterBounds((0.92, 0.98), (1.2, 2.2), (1.0, 4.0)),
        g2egm_grids=grids,
        ptd_osa_options=Ptd.PTDOSAOptions(direct_iterations=8))
end

@testset "Updated heterogeneous AR(1) paths are separate authenticated evidence" begin
    config = _updated_test_config()
    old_hooks = _updated_test_hooks(0.35)
    context = UpdatedAR1._capture_run_context(config, pkgdir(Ptd))
    welfare = UpdatedAR1._run_welfare_unlocked(config, old_hooks;
        workers=1, scheduler=UpdatedAR1.serial_scheduler, context)
    timing = UpdatedAR1._run_timing(config, old_hooks,
        (design=welfare.design, plain_design=welfare.outer_design,
         audit=welfare.sampler_audit);
        workers=1, scheduler=UpdatedAR1.serial_scheduler, context)
    options = UpdatedAR1.AR1UpdatedPathOptions(targets=(1.5, 3.5),
        paths_per_configuration=2)
    reason = "Test a declared solver revision on unchanged observations."
    mktempdir() do directory
        source_path = joinpath(directory, "historical.jld2")
        output_path = joinpath(directory, "updated.jld2")
        UpdatedAR1.publish_ar1_contest(config, welfare, timing;
            output_path=source_path, context, compress=false)
        source_bytes = read(source_path)
        calls = Ref(0)
        hooks = _updated_test_hooks(0.45; calls)
        result = UpdatedAR1.recompute_ar1_contest_paths(source_path;
            output_path, config, options, workers=1,
            solver_update_reason=reason, hooks, compress=false)
        @test result isa UpdatedAR1.AR1UpdatedPathResult
        @test result.path == output_path
        @test calls[] == 6
        @test read(source_path) == source_bytes
        @test result.traces.is_exact_historical_replay === false
        @test result.traces.updates_archived_welfare === false
        @test result.artifact.solver_update_reason == reason
        selected = UpdatedAR1._select_ar1_contest_path_configurations(
            welfare.design, options.targets)
        @test [row.contest_id for row in result.traces.configurations] ==
            [selection.spec.id for selection in selected]
        @test length(unique(row.beta for row in result.traces.configurations)) == 2
        @test length(unique(row.gamma for row in result.traces.configurations)) == 2
        for (group, selection) in enumerate(selected)
            spec = selection.spec
            row = result.traces.configurations[group]
            @test (row.beta, row.gamma, row.initial_cash) ==
                (spec.beta, spec.gamma, spec.initial_cash)
            shocks = UpdatedAR1.predraw_ar1_innovations(spec,
                UpdatedAR1.contest_design_config(config), :ar1)
            columns = (2group - 1):(2group)
            @test result.traces.innovation_rows[columns, :] == shocks[1:2, :]
            @test result.traces.block_fingerprints[group] ==
                UpdatedAR1.shock_fingerprint(shocks)
        end
        @test all(row.current_utility - row.historical_utility == row.utility_change
            for row in result.artifact.comparisons)
        @test any(any(!iszero, row.utility_change) for row in result.artifact.comparisons)
        @test size(result.traces.consumption.g2egm) == (3, 4)
        @test all(row.feasible_actions_checked for row in result.artifact.diagnostics)

        loaded = UpdatedAR1.load_ar1_updated_paths(output_path)
        @test loaded.metadata.payload_fingerprint == result.artifact.metadata.payload_fingerprint
        rendered = UpdatedAR1.render_ar1_updated_paths(output_path;
            graph_directory=joinpath(directory, "figures"), size=(500, 500), dpi=72)
        @test isfile(rendered.paths_path)
        @test calls[] == 6 # Loading and rendering never construct policies.

        damaged = deepcopy(loaded)
        damaged.traces.consumption.g2egm[1, 1] += 0.01
        JLD2.jldopen(output_path, "w") do file
            file["artifact"] = damaged
        end
        @test_throws "payload fingerprint mismatch" UpdatedAR1.load_ar1_updated_paths(output_path)
        UpdatedAR1.save_ar1_updated_paths(output_path, loaded; overwrite=true, compress=false)

        # The welfare source is saved without its local folder: by its name
        # alone here, because it lies outside the package.
        @test loaded.traces.source_artifact == "historical.jld2"
        @test UpdatedAR1._recorded_welfare_source("data/welfare.jld2") ==
            joinpath(pkgdir(Ptd), "data", "welfare.jld2")
        @test UpdatedAR1._recorded_welfare_source(source_path) == source_path
        # Older artifacts recorded an absolute source path. Saving over
        # that source is still refused.
        legacy = merge(loaded, (traces=merge(loaded.traces,
            (source_artifact=source_path,)),))
        legacy_metadata = merge(legacy.metadata, (payload_fingerprint=
            Ptd.stable_fingerprint(UpdatedAR1._updated_payload(legacy)),))
        legacy_metadata = merge(legacy_metadata, (authentication_fingerprint=
            UpdatedAR1._updated_authentication(legacy_metadata),))
        legacy = merge(legacy, (metadata=legacy_metadata,))
        @test_throws "cannot overwrite their archived welfare source" (
            UpdatedAR1.save_ar1_updated_paths(source_path, legacy;
                overwrite=true, compress=false))
        @test read(source_path) == source_bytes
        @test_throws "already exists" UpdatedAR1.recompute_ar1_contest_paths(source_path;
            output_path, config, options, solver_update_reason=reason, hooks)
        @test_throws "cannot overwrite" UpdatedAR1.recompute_ar1_contest_paths(source_path;
            output_path=source_path, config, options, solver_update_reason=reason,
            hooks, overwrite=true)
        @test_throws "nonempty solver_update_reason" UpdatedAR1.recompute_ar1_contest_paths(source_path;
            output_path=joinpath(directory, "empty-reason.jld2"), config, options,
            solver_update_reason="", hooks)
        @test_throws "economic inputs" UpdatedAR1.recompute_ar1_contest_paths(source_path;
            output_path=joinpath(directory, "changed-model.jld2"),
            config=_updated_test_config(return_rate=0.03), options,
            solver_update_reason=reason, hooks)
        @test_throws "regenerated shocks differ" UpdatedAR1.recompute_ar1_contest_paths(source_path;
            output_path=joinpath(directory, "changed-shocks.jld2"), config, options,
            solver_update_reason=reason, workers=1,
            hooks=_updated_test_hooks(0.45; changed_shocks=true))
        @test calls[] == 6

        # A self-consistent historical artifact can be inspected explicitly,
        # but default reuse rejects a different current computational tree.
        stale = merge(loaded, (provenance=merge(loaded.provenance,
            (input_tree_fingerprint=repeat("0", 64),)),))
        metadata = merge(stale.metadata,
            (payload_fingerprint=Ptd.stable_fingerprint(UpdatedAR1._updated_payload(stale)),))
        metadata = merge(metadata,
            (authentication_fingerprint=UpdatedAR1._updated_authentication(metadata),))
        stale = merge(stale, (; metadata))
        JLD2.jldopen(output_path, "w") do file
            file["artifact"] = stale
        end
        @test_throws "different current sources" UpdatedAR1.load_ar1_updated_paths(output_path)
        @test UpdatedAR1.load_ar1_updated_paths(output_path;
            require_current_source=false).metadata.payload_fingerprint == metadata.payload_fingerprint
    end
end
