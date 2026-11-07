using Plots
using Plots.PlotMeasures
using Printf
using JLD2
using ..Ptd: load_single_path_timing_summary, tables_dir, graphs_dir

export HSZ_PATHS_FILENAME, HSZ_TIMING_FILENAME, render_hsz_example

const HSZ_PATHS_FILENAME = "hsz_policy_paths.png"
const HSZ_TIMING_FILENAME = "hsz_example_timings_total.tex"
const LABELS = (ptd_sceq="PTD-SCEQ", ptd_osa="PTD-OSA", g2egm="G2EGM")
# Font sizes of the paper's HSZ path figure. Its 1,500-pixel canvas is 15
# inches wide at Plots' 100 pixels per inch, and the paper prints it about
# 5.6 inches wide, so text is set about 2.7 times its printed size: these
# values print at roughly 7-9 pt.
const _PAPER_FIGURE_FONTS = (title=23, guide=21, tick=19, plot_title=24)

_method_label(method) = LABELS[method]

function _field(container, name::Symbol)
    hasproperty(container, name) && return getproperty(container, name)
    container isa AbstractDict && haskey(container, name) && return container[name]
    container isa AbstractDict && haskey(container, String(name)) &&
        return container[String(name)]
    throw(ArgumentError("artifact is missing field $name"))
end

function _trace_panel(dates, values, colors; title, ylabel="", xlabel="")
    panel = plot(; title, ylabel, xlabel, legend=false, grid=false,
        framestyle=:box, xlims=extrema(dates),
        titlefontsize=_PAPER_FIGURE_FONTS.title,
        guidefontsize=_PAPER_FIGURE_FONTS.guide,
        tickfontsize=_PAPER_FIGURE_FONTS.tick)
    for path in axes(values, 2)
        plot!(panel, dates, values[:, path]; color=colors[path],
            linewidth=0.72, alpha=0.72, label=false)
    end
    return panel
end

function _path_figure(artifact)
    configuration = artifact.scientific_configuration
    horizon = configuration.horizon
    dates = collect(1:horizon)
    n_paths = length(artifact.initial_cash)
    colors = palette(:turbo, n_paths)
    exogenous = artifact.exogenous
    earnings = exp.(exogenous.log_earnings)
    medical = exp.(exogenous.log_medical)
    net_income = exogenous.net_income
    # Short titles and symbol-only axis labels fit the narrow top-row panels
    # at the paper's font sizes; the plot title says the paths are common.
    top = (
        _trace_panel(dates, earnings, colors;
            title="Earnings paths", ylabel="Y(t)"),
        _trace_panel(dates, medical, colors;
            title="Medical-expense paths", ylabel="M(t)"),
        _trace_panel(dates, net_income, colors;
            title="Net-income paths", ylabel="Y(t) - M(t)"))
    cash_panels = Plots.Plot[]
    consumption_panels = Plots.Plot[]
    for method in HSZ_EXAMPLE_METHODS
        paths = _field(artifact.paths, method)
        push!(cash_panels, _trace_panel(dates, paths.cash, colors;
            title=_method_label(method), ylabel="Cash on hand, x(t)"))
        push!(consumption_panels,
            _trace_panel(dates, paths.consumption, colors;
                title=_method_label(method), ylabel="Consumption, c(t)",
                xlabel="Age index, t"))
    end
    support = unique(artifact.initial_cash)
    counts = [count(==(cash), artifact.initial_cash) for cash in support]
    return plot(top..., cash_panels..., consumption_panels...;
        layout=(3, 3), size=(1_500, 1_200), dpi=250,
        margin=3mm, left_margin=5mm, right_margin=6mm,
        plot_title="HSZ common-shock replay: $(n_paths) paths, $(only(unique(counts))) at each of $(length(support)) initial cash levels",
        plot_titlefontsize=_PAPER_FIGURE_FONTS.plot_title)
end

function _hsz_paper_timing_summary(timing_path, artifact)
    isnothing(timing_path) && return nothing
    schema = _field(JLD2.load(timing_path, "artifact"), :schema)
    if schema == HSZ_PATH_TIMING_SCHEMA
        summary = load_hsz_path_timing_summary(timing_path)
        settings = summary.configuration.scientific_configuration
        workload = :shared_policy_paths
    else
        summary = load_single_path_timing_summary(timing_path; required_panels=(:hsz,))
        settings = summary.configurations[:hsz].settings
        workload = :fresh_single_path
    end
    scientific = artifact.scientific_configuration
    # The timing path uses initial cash 16; the accuracy exercise retains its
    # full initial-cash distribution and shock sample.
    all(name -> _field(settings, name) == _field(scientific, name),
        (:utility, :beta, :return_rate, :cash_floor, :horizon,
            :process, :quadrature, :g2egm)) || throw(ArgumentError(
        "HSZ timing calibration or grid differs from the accuracy exercise"))
    return merge(summary, (; workload))
end

function _write_timing(path, artifact; timing_artifact_path=nothing)
    timing = _hsz_paper_timing_summary(timing_artifact_path, artifact)
    if !isnothing(timing) && timing.workload === :shared_policy_paths
        return _write_hsz_path_timing_table(path, artifact, timing)
    end
    single_measurement = !isnothing(timing) &&
        all(method -> timing.rows[(:hsz, method)].repeats == 1, HSZ_EXAMPLE_METHODS)
    lines = String[
        "% Generated by scripts/5_hsz.jl; do not edit.",
        "% Accuracy scientific fingerprint: $(artifact.scientific_fingerprint)",
        single_measurement ? "\\begin{tabular}{lr}" : "\\begin{tabular}{lrr}",
        "\\toprule",
        single_measurement ? "Method & \\makecell{Time\\\\(s)} \\\\" :
            "Method & \\makecell{Mean time\\\\(s)} & \\makecell{Median time\\\\(s)} \\\\",
        "\\midrule",
    ]
    for method in HSZ_EXAMPLE_METHODS
        mean_seconds = isnothing(timing) ? "not reported" :
            @sprintf("%.3f", timing.rows[(:hsz, method)].mean_total_seconds)
        median_seconds = isnothing(timing) ? "not reported" :
            @sprintf("%.3f", timing.rows[(:hsz, method)].median_total_seconds)
        columns = single_measurement ? mean_seconds : "$mean_seconds & $median_seconds"
        push!(lines, "$(_method_label(method)) & $columns \\\\")
    end
    append!(lines, ["\\bottomrule", "\\end{tabular}", ""])
    open(path, "w") do io
        if !isnothing(timing)
            println(io, "% Single-thread timing artifact SHA-256: ", timing.artifact_sha256)
            println(io, "% Single-thread timing artifact fingerprint: ", timing.artifact_fingerprint)
            println(io, "% Single-thread timing source fingerprint: ", timing.source_fingerprint)
            println(io, "% Single-thread timing provenance fingerprint: ", timing.provenance_fingerprint)
        end
        write(io, join(lines, '\n'))
    end
    return path
end

function _write_hsz_path_timing_table(path, artifact, timing)
    open(path, "w") do io
        println(io, "% Generated by scripts/5_hsz.jl; do not edit.")
        println(io, "% Timing artifact SHA-256: ", timing.artifact_sha256)
        println(io, "% Timing artifact fingerprint: ", timing.artifact_fingerprint)
        println(io, "% Timing source fingerprint: ", timing.source_fingerprint)
        println(io, "% Timing provenance fingerprint: ", timing.provenance_fingerprint)
        println(io, "% Accuracy scientific fingerprint: ", artifact.scientific_fingerprint)
        println(io, "% One policy per method is reused across all measured paths.")
        println(io, "% Mean and median times add the fixed construction time to the path simulation time.")
        println(io, raw"\begin{tabular}{lrrr}")
        println(io, raw"\toprule")
        println(io, raw"Method & \makecell{Construction\\(s)} & \makecell{Mean time\\(s)} & \makecell{Median time\\(s)}", " \\\\")
        println(io, raw"\midrule")
        for method in HSZ_EXAMPLE_METHODS
            row = timing.rows[(:hsz, method)]
            @printf(io, "%s & %.3f & %.3f & %.3f \\\\\n", _method_label(method),
                row.construction_seconds,
                row.construction_seconds + row.mean_evaluation_seconds,
                row.construction_seconds + row.median_evaluation_seconds)
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
    return path
end

"""Render the HSZ path figure and timing table used in the paper."""
function render_hsz_example(artifact_path::AbstractString;
    graph_path::AbstractString=graphs_dir(HSZ_PATHS_FILENAME),
    timing_path::AbstractString=tables_dir(HSZ_TIMING_FILENAME),
    timing_artifact_path::Union{Nothing,AbstractString}=nothing)
    _validate_data_output(graph_path)
    _validate_data_output(timing_path)
    artifact = load_hsz_example(artifact_path)
    isnothing(timing_artifact_path) || _validate_utility_render_paths(
        timing_artifact_path, (graph_path, timing_path))
    _hsz_paper_timing_summary(timing_artifact_path, artifact)
    mkpath(dirname(graph_path))
    mkpath(dirname(timing_path))
    figure = _path_figure(artifact)
    savefig(figure, graph_path)
    _write_timing(timing_path, artifact; timing_artifact_path)
    return (; graph_path=abspath(graph_path), timing_path=abspath(timing_path))
end
