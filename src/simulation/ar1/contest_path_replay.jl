# Deterministic replay and rendering of the heterogeneous AR(1) contest paths.
using Plots
using Plots.PlotMeasures
using Printf
using ..Ptd: portable_path

export replay_ar1_contest_paths, render_ar1_stored_paper_figures

const AR1_CONTEST_PATH_TARGETS = (5.5, 10.75, 16.0, 21.25, 26.5)
const AR1_CONTEST_PATHS_PER_CONFIGURATION = 20

"""Convert the authenticated outer design to ordered `ContestSpec`s."""
function _ar1_contest_specs(artifact)
    entries = collect(_exhibit_render_field(
        artifact, :outer_design, "contest artifact"))
    specs = ContestSpec[]
    for (index, entry) in pairs(entries)
        spec = ContestSpec(
            Int(_exhibit_render_field(entry, :id, "outer-design entry")),
            Float64(_exhibit_render_field(
                entry, :beta, "outer-design entry")),
            Float64(_exhibit_render_field(
                entry, :gamma, "outer-design entry")),
            Float64(_exhibit_render_field(
                entry, :initial_cash, "outer-design entry")),
        )
        spec.id == index || throw(ArgumentError(
            "outer-design contest identifiers must equal their positions"))
        all(isfinite, (spec.beta, spec.gamma, spec.initial_cash)) ||
            throw(ArgumentError(
                "outer-design configurations must be finite"))
        push!(specs, spec)
    end
    isempty(specs) && throw(ArgumentError("the outer design is empty"))
    return specs
end

"""
    _select_ar1_contest_path_configurations(specs, targets)

Select one distinct contest configuration nearest each target initial-cash
value. Target order is preserved, and exact distance ties are resolved by the
smaller contest identifier.
"""
function _select_ar1_contest_path_configurations(specs, targets)
    target_values = Float64.(collect(targets))
    !isempty(target_values) && all(isfinite, target_values) ||
        throw(ArgumentError(
            "AR(1) path targets must be a nonempty finite collection"))
    length(target_values) <= length(specs) || throw(ArgumentError(
        "there are fewer contest configurations than path targets"))

    taken = Set{Int}()
    selected = NamedTuple[]
    for (group, target) in pairs(target_values)
        best_index = 0
        best_distance = Inf
        best_id = typemax(Int)
        for (index, spec) in pairs(specs)
            spec.id in taken && continue
            distance = abs(spec.initial_cash - target)
            if distance < best_distance ||
                    (distance == best_distance && spec.id < best_id)
                best_index = index
                best_distance = distance
                best_id = spec.id
            end
        end
        best_index > 0 || throw(ErrorException(
            "failed to select a distinct AR(1) contest configuration"))
        spec = specs[best_index]
        push!(taken, spec.id)
        push!(selected, (;
            group=Int(group),
            target_initial_cash=target,
            contest_index=best_index,
            spec,
        ))
    end
    return selected
end

"""Return the authenticated AR(1) shock fingerprint for one contest."""
function _stored_ar1_block_fingerprint(artifact, contest_index::Integer;
                                       expected_contest_id=nothing)
    contests = collect(_exhibit_render_field(
        _exhibit_render_field(artifact, :run_audit, "contest artifact"),
        :contests, "run audit"))
    index = Int(contest_index)
    index in eachindex(contests) || throw(BoundsError(contests, index))
    contest = contests[index]
    if !isnothing(expected_contest_id)
        Int(_exhibit_render_field(contest, :contest_id, "contest audit")) ==
            Int(expected_contest_id) || throw(ArgumentError(
                "the compact contest audit does not match the original identifier"))
    end
    panels = collect(_exhibit_render_field(
        contest, :panels, "contest audit"))
    matches = [panel for panel in panels if Symbol(_exhibit_render_field(
        panel, :panel, "panel audit")) === AR1_PANEL]
    length(matches) == 1 || throw(ArgumentError(
        "the contest audit must contain exactly one AR(1) panel"))
    return String(_exhibit_render_field(
        only(matches), :block_fingerprint, "AR(1) panel audit"))
end

"""Convert recorded trajectory rows to complete path-by-date matrices."""
function _ar1_trajectory_matrices(rows, horizon::Integer,
                                  path_count::Integer)
    dates = Int(horizon)
    paths = Int(path_count)
    length(rows) == dates * paths || throw(DimensionMismatch(
        "AR(1) replay returned the wrong number of trajectory rows"))
    income = fill(NaN, dates, paths)
    cash = fill(NaN, dates, paths)
    consumption = fill(NaN, dates, paths)
    for row in rows
        date = Int(row.observed.date)
        path = Int(row.path)
        1 <= date <= dates && 1 <= path <= paths || throw(BoundsError(
            (dates=1:dates, paths=1:paths), (date, path)))
        isnan(cash[date, path]) || throw(ArgumentError(
            "AR(1) replay contains a duplicate trajectory row"))
        income[date, path] = exp(Float64(row.observed.exogenous.log_income))
        cash[date, path] = Float64(row.observed.cash)
        consumption[date, path] = Float64(row.action.consumption)
    end
    all(isfinite, income) && all(isfinite, cash) &&
        all(isfinite, consumption) || throw(ErrorException(
            "AR(1) replay left an unfilled or nonfinite trajectory cell"))
    return (; income, cash, consumption)
end

"""Replay and authenticate the displayed paths for one configuration."""
function _replay_ar1_contest_configuration(
    artifact, config::AR1ContestRunConfig, design_config,
    selection, path_count::Int, hooks::AR1RunnerHooks)

    spec = selection.spec
    innovations = _validate_predraw(
        hooks.predraw(spec, design_config, AR1_PANEL), config)
    _evaluator_domain_audit(config, spec, innovations)
    block_fingerprint = shock_fingerprint(innovations)
    block_fingerprint == _stored_ar1_block_fingerprint(
        artifact, selection.contest_index;
        expected_contest_id=spec.id) || throw(ArgumentError(
            "reconstructed shocks do not match the authenticated block " *
            "for contest $(spec.id)"))

    inputs = _model_inputs(config, spec)
    effective_g2egm_options = _effective_g2egm_options(config)
    utilities = _exhibit_render_field(
        artifact, :pathwise_utilities, "contest artifact")
    series = Dict{Symbol,Any}()
    replayed_utility = Dict{Symbol,Vector{Float64}}()
    for (method_index, method) in pairs(config.methods)
        policy = hooks.build_policy(method, inputs.model_config,
            inputs.process, inputs.base_rule, config.g2egm_grids,
            effective_g2egm_options, config.ptd_osa_options)
        hooks.validate_policy(
            policy, method, inputs.model_config, inputs.process)
        _compact_quadrature_audit(
            method, hooks.quadrature_audit(policy), inputs.base_rule)
        hooks.reset_diagnostics!(policy)
        rows = Any[]
        evaluation = evaluate_policy_paths(
            policy, method, inputs.model_config, inputs.process,
            inputs.initial, view(innovations, 1:path_count, :);
            action_audit=PolicyActionAudit(), trajectory_rows=rows)
        _validate_evaluation(evaluation, method, path_count)
        expected = Float64.(collect(view(utilities,
            1:path_count, method_index, selection.contest_index, 1)))
        evaluation.utility == expected || throw(ArgumentError(
            "replayed utility does not match the authenticated artifact " *
            "for contest $(spec.id), method $method"))
        series[method] = _ar1_trajectory_matrices(
            rows, config.horizon, path_count)
        replayed_utility[method] = evaluation.utility
    end
    reference_income = series[first(config.methods)].income
    all(method -> series[method].income == reference_income,
        config.methods) || throw(ErrorException(
            "methods did not replay identical income paths"))
    return (;
        selection,
        block_fingerprint,
        innovation_rows=Matrix{Float64}(
            view(innovations, 1:path_count, :)),
        income=reference_income,
        series,
        replayed_utility,
    )
end

"""Combine authenticated configuration replays into one figure payload."""
function _combine_ar1_contest_path_replays(
    artifact_path::AbstractString, artifact, config::AR1ContestRunConfig,
    selections, replays, path_count::Int)

    configuration_count = length(selections)
    total_paths = configuration_count * path_count
    groups = reduce(vcat,
        (fill(index, path_count) for index in 1:configuration_count);
        init=Int[])
    source_contest_ids = reduce(vcat,
        (fill(selection.spec.id, path_count) for selection in selections);
        init=Int[])
    source_path_ids = repeat(collect(1:path_count), configuration_count)
    initial_cash = reduce(vcat,
        (fill(selection.spec.initial_cash, path_count)
         for selection in selections); init=Float64[])
    configurations = [(;
        group=selection.group,
        target_initial_cash=selection.target_initial_cash,
        contest_index=selection.contest_index,
        contest_id=selection.spec.id,
        beta=selection.spec.beta,
        gamma=selection.spec.gamma,
        initial_cash=selection.spec.initial_cash,
        display_range=((selection.group - 1) * path_count + 1):
            (selection.group * path_count),
    ) for selection in selections]
    income = hcat((replay.income for replay in replays)...)
    cash = _method_named_tuple(Dict(method =>
        hcat((replay.series[method].cash for replay in replays)...)
        for method in config.methods))
    consumption = _method_named_tuple(Dict(method =>
        hcat((replay.series[method].consumption for replay in replays)...)
        for method in config.methods))
    replayed_utility = _method_named_tuple(Dict(method =>
        reduce(vcat, (replay.replayed_utility[method]
            for replay in replays); init=Float64[])
        for method in config.methods))
    innovation_rows = reduce(vcat,
        (replay.innovation_rows for replay in replays))
    block_fingerprints = [replay.block_fingerprint for replay in replays]
    metadata = _exhibit_render_field(artifact, :metadata, "contest artifact")
    source_authentication_fingerprint = String(_exhibit_render_field(
        metadata, :authentication_fingerprint, "contest metadata"))
    replay_fingerprint = stable_fingerprint((;
        schema="ptd-ar1-contest-path-replay-v1",
        source_authentication_fingerprint,
        configurations,
        block_fingerprints,
        paths_per_configuration=path_count,
        dates=collect(1:config.horizon),
        configuration_groups=groups,
        source_contest_ids,
        source_path_ids,
        innovation_rows,
        income,
        cash,
        consumption,
        replayed_utility,
    ))
    payload = (;
        dates=collect(1:config.horizon),
        path_ids=collect(1:total_paths),
        configuration_groups=groups,
        source_contest_ids,
        source_path_ids,
        initial_cash,
        configurations,
        income,
        cash,
        consumption,
        replayed_utility,
        innovation_rows,
        block_fingerprints,
        source_artifact=portable_path(artifact_path),
        source_authentication_fingerprint,
        replay_fingerprint,
        outside_reported_timing=true,
        adds_welfare_observations=false,
    )
    _validate_normalized_traces(payload)
    return payload
end

"""
    replay_ar1_contest_paths(artifact_path; config=production_run_config(), ...)

Deterministically reconstruct the five heterogeneous configurations displayed
in the paper's AR(1) path figure. The complete hidden innovation block for each
selected successful contest is regenerated from the seed recorded by the scientific
configuration and checked against the artifact audit before the first 20 paths
are evaluated. Every replayed utility must exactly match its stored artifact
value, so this operation adds neither configurations nor welfare observations.
Failed configurations remain in the artifact's original design and failure audit;
they are excluded from the display, whose labels retain original contest IDs.
"""
function replay_ar1_contest_paths(
    artifact_path::AbstractString;
    config::AR1ContestRunConfig=production_run_config(),
    targets=AR1_CONTEST_PATH_TARGETS,
    paths_per_configuration::Integer=
        AR1_CONTEST_PATHS_PER_CONFIGURATION,
    workers::Integer=min(Threads.nthreads(), length(targets)),
    hooks::AR1RunnerHooks=AR1RunnerHooks())

    validate_run_config(config)
    LinearAlgebra.BLAS.set_num_threads(1)
    LinearAlgebra.BLAS.get_num_threads() == 1 || throw(ErrorException(
        "AR(1) path replay requires one BLAS thread"))
    artifact = load_ar1_contest_welfare(artifact_path)
    metadata = _exhibit_render_field(artifact, :metadata, "contest artifact")
    String(_exhibit_render_field(metadata,
        :scientific_fingerprint, "contest metadata")) ==
        scientific_fingerprint(config) || throw(ArgumentError(
            "the AR(1) replay configuration does not match the artifact"))
    Symbol.(collect(_exhibit_render_field(
        artifact, :methods, "contest artifact"))) == collect(config.methods) ||
        throw(ArgumentError("the AR(1) artifact method order is invalid"))
    Symbol.(collect(_exhibit_render_field(
        artifact, :panels, "contest artifact"))) == [AR1_PANEL] ||
        throw(ArgumentError("the AR(1) artifact panel declaration is invalid"))
    path_count = Int(paths_per_configuration)
    1 <= path_count <= config.n_paths || throw(ArgumentError(
        "paths_per_configuration must lie in 1:n_paths"))
    worker_count = Int(workers)
    worker_count > 0 || throw(ArgumentError("workers must be positive"))
    _is_production_config(config) && !_is_default_hooks(hooks) &&
        throw(ArgumentError(
            "production AR(1) path replay requires the default runner hooks"))

    specs = _ar1_contest_specs(artifact)
    length(specs) == config.n_contests || throw(DimensionMismatch(
        "the AR(1) artifact design does not match the run configuration"))
    # The utility cube and audit rows contain successful configurations only;
    # their positions differ from original design IDs after a retained failure.
    contests = _exhibit_render_field(
        _exhibit_render_field(artifact, :run_audit, "contest artifact"),
        :contests, "run audit")
    successful_specs = [specs[Int(_exhibit_render_field(
        contest, :contest_id, "contest audit"))] for contest in contests]
    isempty(successful_specs) && throw(ArgumentError(
        "no successful configurations are available for AR(1) path replay"))
    selections = _select_ar1_contest_path_configurations(successful_specs, targets)
    design_config = contest_design_config(config)
    replays = Vector{Any}(undef, length(selections))
    replay_one = index -> begin
        replays[index] = _replay_ar1_contest_configuration(
            artifact, config, design_config, selections[index],
            path_count, hooks)
        nothing
    end
    if worker_count == 1
        serial_scheduler(replay_one, eachindex(selections), 1)
    else
        threaded_scheduler(replay_one, eachindex(selections),
            min(worker_count, length(selections)))
    end
    return _combine_ar1_contest_path_replays(
        artifact_path, artifact, config, selections, replays, path_count)
end

"""Validate the grouping metadata attached to an AR(1) path replay."""
function _validate_ar1_contest_trace_groups(trace_payload, normalized)
    groups = Int.(collect(_exhibit_render_field(
        trace_payload, :configuration_groups, "contest traces")))
    length(groups) == length(normalized.path_ids) || throw(DimensionMismatch(
        "trace configuration groups have the wrong length"))
    configurations = collect(_exhibit_render_field(
        trace_payload, :configurations, "contest traces"))
    !isempty(configurations) || throw(ArgumentError(
        "contest traces must contain at least one configuration"))
    groups == sort(groups) && unique(groups) == collect(eachindex(configurations)) ||
        throw(ArgumentError(
            "trace configuration groups must be ordered and contiguous"))
    counts = [count(==(group), groups) for group in eachindex(configurations)]
    length(unique(counts)) == 1 || throw(ArgumentError(
        "trace configuration groups must contain equal path counts"))
    source_contest_ids = Int.(collect(_exhibit_render_field(
        trace_payload, :source_contest_ids, "contest traces")))
    source_path_ids = Int.(collect(_exhibit_render_field(
        trace_payload, :source_path_ids, "contest traces")))
    length(source_contest_ids) == length(groups) == length(source_path_ids) ||
        throw(DimensionMismatch("trace source identifiers have the wrong length"))
    for (group, configuration) in pairs(configurations)
        columns = findall(==(group), groups)
        contest_id = Int(_exhibit_render_field(
            configuration, :contest_id, "trace configuration"))
        cash = Float64(_exhibit_render_field(
            configuration, :initial_cash, "trace configuration"))
        all(source_contest_ids[columns] .== contest_id) &&
            source_path_ids[columns] == collect(1:length(columns)) &&
            all(normalized.initial_cash[columns] .== cash) ||
            throw(ArgumentError(
                "trace source identifiers disagree with their configuration"))
    end
    return (; groups, configurations, paths_per_configuration=only(unique(counts)))
end

"""Draw the grouped heterogeneous-preference AR(1) replay figure."""
function _ar1_contest_trace_figure(trace_payload;
    size::Tuple{Int,Int}=(1_650, 1_500), dpi::Int=300)

    normalized = _validate_normalized_traces(trace_payload)
    grouping = _validate_ar1_contest_trace_groups(trace_payload, normalized)
    dates = normalized.dates
    income = normalized.income
    cash = normalized.cash
    consumption = normalized.consumption
    colors = palette(:tab10, max(10, length(grouping.configurations)))

    # The legend sits above the income panel, in two columns, so that it
    # hides no path.
    income_panel = Plots.plot(; xlabel="t", ylabel="Income, Yₜ",
        title="Common income paths", legend=:outertop,
        legend_column=min(2, length(grouping.configurations)), grid=false,
        framestyle=:box, titlefontsize=_PAPER_FIGURE_FONTS.title,
        guidefontsize=_PAPER_FIGURE_FONTS.guide,
        tickfontsize=_PAPER_FIGURE_FONTS.tick,
        legendfontsize=_PAPER_FIGURE_FONTS.legend)
    for (group, configuration) in pairs(grouping.configurations)
        columns = findall(==(group), grouping.groups)
        label = @sprintf("x₁ = %.1f, β = %.3f, γ = %.3f",
            Float64(_exhibit_render_field(configuration, :initial_cash,
                "trace configuration")),
            Float64(_exhibit_render_field(configuration, :beta,
                "trace configuration")),
            Float64(_exhibit_render_field(configuration, :gamma,
                "trace configuration")))
        for (path_in_group, column) in pairs(columns)
            Plots.plot!(income_panel, dates, income[:, column];
                color=colors[group], alpha=0.55, linewidth=0.9,
                label=path_in_group == 1 ? label : false)
        end
    end

    cash_panels = Plots.Plot[]
    consumption_panels = Plots.Plot[]
    for method in AR1_CONTEST_EXHIBIT_METHODS
        method_label = _exhibit_render_method_label(method)
        cash_panel = Plots.plot(; xlabel="t", ylabel="Cash on hand, xₜ",
            title="Cash, $method_label", legend=false, grid=false,
            framestyle=:box, titlefontsize=_PAPER_FIGURE_FONTS.title,
            guidefontsize=_PAPER_FIGURE_FONTS.guide,
            tickfontsize=_PAPER_FIGURE_FONTS.tick)
        consumption_panel = Plots.plot(;
            xlabel="t", ylabel="Consumption, cₜ",
            title="Consumption, $method_label", legend=false, grid=false,
            framestyle=:box, titlefontsize=_PAPER_FIGURE_FONTS.title,
            guidefontsize=_PAPER_FIGURE_FONTS.guide,
            tickfontsize=_PAPER_FIGURE_FONTS.tick)
        for group in eachindex(grouping.configurations)
            for column in findall(==(group), grouping.groups)
                Plots.plot!(cash_panel, dates, cash[method][:, column];
                    color=colors[group], alpha=0.55, linewidth=0.9,
                    label=false)
                Plots.plot!(consumption_panel, dates,
                    consumption[method][:, column]; color=colors[group],
                    alpha=0.55, linewidth=0.9, label=false)
            end
        end
        push!(cash_panels, cash_panel)
        push!(consumption_panels, consumption_panel)
    end
    layout = @layout [a{0.33h}; grid(2, 3)]
    return Plots.plot(income_panel, cash_panels..., consumption_panels...;
        layout, size, dpi, margin=3mm, left_margin=7mm)
end

"""Paths and configuration provenance of the two paper-facing AR(1) figures."""
struct AR1PaperFigureResult{C}
    surface_path::String
    paths_path::String
    selected_configurations::C
    replay_fingerprint::String
    scope::Symbol
end

"""
    render_ar1_paper_figures(contest_artifact_path, exhibit_artifact_path; ...)

Render the paper's two AR(1) figures. Figure 8 uses the fixed-calibration
surface stored by the exhibit artifact. With `updated_paths_path`, Figure 9
uses separately saved current-solver trajectories for the original heterogeneous
configurations. Their source artifact must match `contest_artifact_path`.
Cached rendering authenticates the recorded source and data. Set
`require_current_path_source=true` when also requiring the current source tree
to match the one that computed those paths.
Without that argument, the historical replay API still requires exact matches
between reconstructed and archived utilities.
"""
function render_ar1_paper_figures(
    contest_artifact_path::AbstractString,
    exhibit_artifact_path::AbstractString;
    graph_directory::AbstractString=graphs_dir(),
    require_publication::Bool=true,
    surface_size::Tuple{Int,Int}=(1_650, 1_080),
    trace_size::Tuple{Int,Int}=(1_650, 1_500),
    dpi::Int=300,
    camera::Tuple{<:Real,<:Real}=(48, 28),
    updated_paths_path::Union{Nothing,AbstractString}=nothing,
    require_current_path_source::Bool=false,
    replay_config::AR1ContestRunConfig=production_run_config(),
    replay_targets=AR1_CONTEST_PATH_TARGETS,
    paths_per_configuration::Integer=
        AR1_CONTEST_PATHS_PER_CONFIGURATION,
    replay_workers::Integer=min(Threads.nthreads(), length(replay_targets)))

    exhibit = load_ar1_contest_exhibit_artifact(exhibit_artifact_path;
        require_publication, require_live_inputs=false)
    validate_ar1_contest_exhibit_artifact(exhibit; require_publication)
    surface = _exhibit_render_field(exhibit, :surface, "exhibit artifact")
    traces = if isnothing(updated_paths_path)
        replay_ar1_contest_paths(contest_artifact_path;
            config=replay_config, targets=replay_targets,
            paths_per_configuration, workers=replay_workers)
    else
        updated = load_ar1_updated_paths(updated_paths_path;
            require_current_source=require_current_path_source)
        source = load_ar1_contest_welfare(contest_artifact_path)
        source_metadata = _exhibit_render_field(source, :metadata, "contest artifact")
        String(_exhibit_render_field(source_metadata,
            :authentication_fingerprint, "contest metadata")) ==
            updated.metadata.source_authentication_fingerprint ||
            throw(ArgumentError("updated path trajectories belong to another contest artifact"))
        updated.traces
    end
    surface_path = joinpath(abspath(graph_directory),
        AR1_CONTEST_EXHIBIT_SURFACE_FILENAME)
    paths_path = joinpath(abspath(graph_directory),
        AR1_CONTEST_EXHIBIT_PATHS_FILENAME)
    _atomic_savefig(_surface_figure(surface; size=surface_size, dpi, camera),
        surface_path)
    _atomic_savefig(
        _ar1_contest_trace_figure(traces; size=trace_size, dpi), paths_path)
    scope = isnothing(updated_paths_path) ? :authenticated_heterogeneous_replay :
        :updated_heterogeneous_paths
    return AR1PaperFigureResult(surface_path, paths_path,
        traces.configurations, traces.replay_fingerprint, scope)
end

"""
    render_ar1_stored_paper_figures(exhibit_artifact_path; ...)

Render the paper's surface and fixed-calibration paths entirely from an
authenticated exhibit artifact. No solver, shock generation, or policy replay
is invoked. The separate `replay_ar1_contest_paths` API continues to require an
exact current configuration and exact matches of replayed and stored utilities.
"""
function render_ar1_stored_paper_figures(
    exhibit_artifact_path::AbstractString;
    graph_directory::AbstractString=graphs_dir(),
    require_publication::Bool=true,
    surface_size::Tuple{Int,Int}=(1_650, 1_080),
    trace_size::Tuple{Int,Int}=(1_650, 1_500),
    dpi::Int=300,
    camera::Tuple{<:Real,<:Real}=(48, 28))

    exhibit = load_ar1_contest_exhibit_artifact(exhibit_artifact_path;
        require_publication, require_live_inputs=false)
    surface = _exhibit_render_field(exhibit, :surface, "exhibit artifact")
    traces = _exhibit_render_field(exhibit, :traces, "exhibit artifact")
    scientific = _exhibit_render_field(exhibit, :scientific_configuration,
        "exhibit artifact")
    model = _exhibit_render_field(scientific, :model, "scientific_configuration")
    metadata = _exhibit_render_field(exhibit, :metadata, "exhibit artifact")
    surface_path = joinpath(abspath(graph_directory),
        AR1_CONTEST_EXHIBIT_SURFACE_FILENAME)
    paths_path = joinpath(abspath(graph_directory),
        AR1_CONTEST_EXHIBIT_PATHS_FILENAME)
    _atomic_savefig(_surface_figure(surface; size=surface_size, dpi, camera),
        surface_path)
    _atomic_savefig(_trace_figure(traces; size=trace_size, dpi), paths_path)
    return (; surface_path, paths_path,
        scope=:stored_fixed_calibration_traces,
        calibration=(beta=Float64(_exhibit_render_field(model, :beta, "model")),
            gamma=Float64(_exhibit_render_field(model, :gamma, "model"))),
        n_paths=length(_exhibit_render_field(traces, :path_ids, "traces")),
        source_artifact=abspath(exhibit_artifact_path),
        source_authentication_fingerprint=String(_exhibit_render_field(
            metadata, :authentication_fingerprint, "metadata")),
        trace_fingerprint=stable_fingerprint(traces))
end
