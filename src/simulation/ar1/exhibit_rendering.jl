using Dates
using JLD2
using Plots
using Plots.PlotMeasures
using Printf
using SHA
using Statistics: mean
using DrWatson: projectdir

using ..IID: consumption_equivalent_bps
using ..Ptd: paired_comparison, stable_fingerprint, _validate_data_output,
    _temporary_jld_path, tables_dir, graphs_dir

export AR1_CONTEST_EXHIBIT_RENDER_SCHEMA,
       AR1_CONTEST_EXHIBIT_SURFACE_FILENAME,
       AR1_CONTEST_EXHIBIT_PATHS_FILENAME,
       AR1_CONTEST_EXHIBIT_TIMING_FILENAME,
       AR1_CONTEST_EXHIBIT_WELFARE_FILENAME,
       AR1_CONTEST_EXHIBIT_MANIFEST_FILENAME,
       render_ar1_contest_exhibits,
       render_ar1_paper_figures,
       load_ar1_contest_exhibit_render_manifest

const AR1_CONTEST_EXHIBIT_RENDER_SCHEMA =
    "ptd-ar1-contest-exhibit-render-v4"
const AR1_CONTEST_EXHIBIT_RENDER_AUTHENTICATION_SCHEMA =
    "ptd-ar1-contest-exhibit-render-authentication-v4"
const AR1_CONTEST_EXHIBIT_SURFACE_FILENAME = "ar1_policy_surface.png"
const AR1_CONTEST_EXHIBIT_PATHS_FILENAME = "ar1_policy_paths.png"
const AR1_CONTEST_EXHIBIT_DIAGNOSTIC_PATHS_FILENAME =
    "ar1_fixed_calibration_paths.png"
const AR1_CONTEST_EXHIBIT_TIMING_FILENAME =
    "ar1_contest_timings.tex"
const AR1_CONTEST_EXHIBIT_WELFARE_FILENAME =
    "ar1_contest_welfare.tex"
const AR1_CONTEST_EXHIBIT_MANIFEST_FILENAME =
    "ar1_contest_render_manifest.jld2"
# Okabe-Ito colorblind-safe hues; linestyles are a secondary encoding so
# slice identity never rests on color alone.
const _SLICE_QUANTILES = (0.25, 0.75)
const _SLICE_COLORS = ("#0072B2", "#D55E00")
const _SLICE_LINESTYLES = (:solid, :dash)
const _EXHIBIT_RENDER_UTC_FORMAT = dateformat"yyyy-mm-ddTHH:MM:SS.sss"
const _PNG_SIGNATURE =
    UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
# Font sizes of the paper's AR(1) figures (policy surfaces and paths). Their
# 1,650-pixel canvases are 16.5 inches wide at Plots' 100 pixels per inch,
# and the paper prints them 5.6 to 6.3 inches wide, so text is set 2.6 to
# 2.9 times its printed size: these values print at roughly 6-9 pt.
const _PAPER_FIGURE_FONTS = (title=24, guide=22, tick=19, legend=18)

_exhibit_render_utc_string(value::DateTime) = Dates.format(value, _EXHIBIT_RENDER_UTC_FORMAT) * "Z"
_exhibit_render_sha256_file(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

function _exhibit_render_field(container, name::Symbol, location::AbstractString)
    if container isa NamedTuple
        hasproperty(container, name) || throw(ArgumentError(
            "$location is missing field $name"))
        return getproperty(container, name)
    elseif container isa AbstractDict
        haskey(container, name) && return container[name]
        haskey(container, String(name)) && return container[String(name)]
        throw(ArgumentError("$location is missing field $name"))
    end
    throw(ArgumentError("$location must be a named tuple or dictionary"))
end

function _method_dict(container, location::AbstractString)
    return Dict(method => _exhibit_render_field(container, method, location)
                for method in AR1_CONTEST_EXHIBIT_METHODS)
end

function _exhibit_render_method_label(method::Symbol)
    method in AR1_CONTEST_EXHIBIT_METHODS || throw(ArgumentError(
        "unknown contest-exhibit method $method"))
    return AR1_CONTEST_EXHIBIT_LABELS[method]
end

function _validate_normalized_surface(surface_payload)
    cash_grid = Float64.(collect(_exhibit_render_field(
        surface_payload, :cash_grid, "surface")))
    log_income_grid = Float64.(collect(_exhibit_render_field(
        surface_payload, :log_income_grid, "surface")))
    length(cash_grid) >= 2 && all(isfinite, cash_grid) &&
        all(diff(cash_grid) .> 0.0) || throw(ArgumentError(
        "surface cash grid must be finite and strictly increasing"))
    length(log_income_grid) >= 2 && all(isfinite, log_income_grid) &&
        all(diff(log_income_grid) .> 0.0) || throw(ArgumentError(
        "surface log-income grid must be finite and strictly increasing"))
    consumption = _method_dict(_exhibit_render_field(
        surface_payload, :consumption, "surface"), "surface.consumption")
    for method in AR1_CONTEST_EXHIBIT_METHODS
        values = consumption[method]
        values isa AbstractMatrix{<:Real} &&
            size(values) == (length(cash_grid), length(log_income_grid)) &&
            all(isfinite, values) || throw(ArgumentError(
            "surface consumption for $method has an invalid shape or value"))
        tolerance = 128eps(max(last(cash_grid), 1.0))
        all(values .>= -tolerance) &&
            all(values .<= reshape(cash_grid, :, 1) .+ tolerance) ||
            throw(ArgumentError(
                "surface consumption for $method is infeasible"))
    end
    return (; cash_grid, log_income_grid, consumption)
end

function _validate_normalized_traces(trace_payload)
    dates = Int.(collect(_exhibit_render_field(trace_payload, :dates, "traces")))
    path_ids = Int.(collect(_exhibit_render_field(trace_payload, :path_ids, "traces")))
    length(dates) >= 2 && dates == collect(first(dates):last(dates)) ||
        throw(ArgumentError("trace dates must be consecutive"))
    !isempty(path_ids) && length(unique(path_ids)) == length(path_ids) ||
        throw(ArgumentError("trace path ids must be nonempty and distinct"))
    initial_cash = Float64.(collect(_exhibit_render_field(
        trace_payload, :initial_cash, "traces")))
    length(initial_cash) == length(path_ids) && all(isfinite, initial_cash) ||
        throw(ArgumentError("trace initial cash is invalid"))
    income = _exhibit_render_field(trace_payload, :income, "traces")
    income isa AbstractMatrix{<:Real} &&
        size(income) == (length(dates), length(path_ids)) &&
        all(isfinite, income) && all(income .> 0.0) || throw(ArgumentError(
        "common income traces have an invalid shape or value"))
    cash = _method_dict(_exhibit_render_field(trace_payload, :cash, "traces"),
        "traces.cash")
    consumption = _method_dict(_exhibit_render_field(
        trace_payload, :consumption, "traces"), "traces.consumption")
    for method in AR1_CONTEST_EXHIBIT_METHODS
        method_cash = cash[method]
        method_consumption = consumption[method]
        for (values, name) in ((method_cash, "cash"),
                               (method_consumption, "consumption"))
            values isa AbstractMatrix{<:Real} &&
                size(values) == size(income) && all(isfinite, values) ||
                throw(ArgumentError(
                    "trace $name for $method has an invalid shape or value"))
        end
        all(method_cash .>= 0.0) && all(method_consumption .>= 0.0) &&
            all(method_consumption .<= method_cash) || throw(ArgumentError(
            "trace action for $method is infeasible"))
        vec(method_cash[1, :]) == initial_cash || throw(ArgumentError(
            "trace cash does not begin at the declared initial values"))
    end
    return (; dates, path_ids, initial_cash, income, cash, consumption)
end

"""
    _slice_column_indices(log_income_grid)

Select up to two distinct log-income grid columns at the quartile
positions of the grid index range for the one-dimensional policy slices.

# Arguments
- `log_income_grid`: strictly increasing vector of log-income grid
  values.

# Returns
`Vector{Int}` of distinct increasing column indices: two on any grid
with enough columns, one on degenerate grids.
"""
function _slice_column_indices(log_income_grid)

    count = length(log_income_grid)
    indices = [clamp(round(Int, 1 + (count - 1) * quantile), 1, count)
               for quantile in _SLICE_QUANTILES]

    return unique(indices)
end

"""
    _surface_slice_levels(surface_payload)

Return the log-income levels at which the surface figure draws its
one-dimensional consumption-policy slices.

# Arguments
- `surface_payload`: the artifact surface section accepted by
  `_validate_normalized_surface`.

# Returns
`Vector{Float64}` of the selected log-income grid values.
"""
function _surface_slice_levels(surface_payload)

    normalized = _validate_normalized_surface(surface_payload)
    grid = normalized.log_income_grid
    levels = [Float64(grid[column])
              for column in _slice_column_indices(grid)]

    return levels
end

"""
    _sparse_ticks(grid, max_ticks)

Pick at most `max_ticks` round tick values inside the range of `grid`.

The 3D panels of the policy-surface figure are small at the paper's font
sizes, so their horizontal axes carry fewer ticks than Plots' default.

# Arguments
- `grid::AbstractVector{<:Real}`: grid whose smallest and largest values
  bound the axis.
- `max_ticks::Int`: largest number of ticks to return; at least 2.

# Returns
- `Vector{Float64}`: increasing tick values within `extrema(grid)`.
"""
function _sparse_ticks(grid::AbstractVector{<:Real}, max_ticks::Int)

    lower, upper = Float64.(extrema(grid))
    ticks = first(Plots.PlotUtils.optimize_ticks(lower, upper;
        k_min=2, k_max=max_ticks))

    return filter(tick -> lower <= tick <= upper, ticks)
end

function _surface_figure(surface_payload;
    size::Tuple{Int,Int}=(1_650, 1_080), dpi::Int=300,
    camera::Tuple{<:Real,<:Real}=(48, 28))

    normalized = _validate_normalized_surface(surface_payload)
    cash_grid = normalized.cash_grid
    log_income_grid = normalized.log_income_grid
    consumption = normalized.consumption
    zlower = min(0.0, minimum(minimum(consumption[method])
        for method in AR1_CONTEST_EXHIBIT_METHODS))
    zupper = maximum(maximum(consumption[method])
        for method in AR1_CONTEST_EXHIBIT_METHODS)
    zupper > zlower || (zupper = zlower + 1.0)

    surface_panels = Plots.Plot[]
    for (index, method) in pairs(AR1_CONTEST_EXHIBIT_METHODS)
        values = Matrix{Float64}(consumption[method])
        panel = Plots.surface(cash_grid, log_income_grid,
            permutedims(values);
            title="($(Char(Int('a') + index - 1))) $(_exhibit_render_method_label(method))",
            xlabel="x₁",
            ylabel="ℓ₁",
            zlabel="c₁",
            xlims=extrema(cash_grid), ylims=extrema(log_income_grid),
            zlims=(zlower, zupper), clims=(zlower, zupper),
            xticks=_sparse_ticks(cash_grid, 2),
            yticks=_sparse_ticks(log_income_grid, 3),
            color=:viridis,
            colorbar=false,
            camera=(Float64(camera[1]), Float64(camera[2])),
            grid=false, framestyle=:box,
            titlefontsize=_PAPER_FIGURE_FONTS.title,
            guidefontsize=_PAPER_FIGURE_FONTS.guide,
            tickfontsize=_PAPER_FIGURE_FONTS.tick)
        push!(surface_panels, panel)
    end
    slice_indices = _slice_column_indices(log_income_grid)
    method_count = length(AR1_CONTEST_EXHIBIT_METHODS)
    slice_panels = Plots.Plot[]
    for (index, method) in pairs(AR1_CONTEST_EXHIBIT_METHODS)
        values = Matrix{Float64}(consumption[method])
        letter = Char(Int('a') + method_count + index - 1)
        method_label = _exhibit_render_method_label(method)
        panel = Plots.plot(;
            title="($letter) $method_label, ℓ₁ slices",
            xlabel="x₁",
            ylabel="c₁",
            xlims=extrema(cash_grid), ylims=(zlower, zupper),
            legend=index == 1 ? :topright : false,
            grid=false, framestyle=:box,
            titlefontsize=_PAPER_FIGURE_FONTS.title,
            guidefontsize=_PAPER_FIGURE_FONTS.guide,
            tickfontsize=_PAPER_FIGURE_FONTS.tick,
            legendfontsize=_PAPER_FIGURE_FONTS.legend)
        for (slot, column) in pairs(slice_indices)
            Plots.plot!(panel, cash_grid, values[:, column];
                color=_SLICE_COLORS[slot],
                linestyle=_SLICE_LINESTYLES[slot], linewidth=2.0,
                label=@sprintf("ℓ₁ = %.2f", log_income_grid[column]))
        end
        push!(slice_panels, panel)
    end
    return Plots.plot(surface_panels..., slice_panels...;
        layout=(2, 3), size, dpi, margin=2mm, left_margin=5mm,
        bottom_margin=5mm)
end

function _positive_limits(values)
    upper = maximum(maximum(values[method])
                    for method in AR1_CONTEST_EXHIBIT_METHODS)
    return (0.0, max(1.04 * upper, 1.0))
end

function _trace_figure(trace_payload;
    size::Tuple{Int,Int}=(1_500, 1_000), dpi::Int=300)

    normalized = _validate_normalized_traces(trace_payload)
    dates, path_ids = normalized.dates, normalized.path_ids
    initial_cash = normalized.initial_cash
    income = normalized.income
    cash, consumption = normalized.cash, normalized.consumption
    # A deterministic palette binds row/path id to one color and is reused in
    # every subplot. A 100-entry legend would obscure the evidence, so path ids
    # remain recorded in the artifact/manifest rather than printed as a legend.
    colors = palette(:turbo, length(path_ids))

    income_panel = Plots.plot(; title="Common income paths",
        ylabel="Income, Yₜ", legend=false, grid=false, framestyle=:box,
        xlims=extrema(dates), titlefontsize=11, guidefontsize=9,
        tickfontsize=7)
    for column in eachindex(path_ids)
        Plots.plot!(income_panel, dates, income[:, column];
            color=colors[column], linewidth=0.85, alpha=0.78, label=false)
    end

    cash_limits = _positive_limits(cash)
    consumption_limits = _positive_limits(consumption)
    cash_panels = Plots.Plot[]
    consumption_panels = Plots.Plot[]
    for method in AR1_CONTEST_EXHIBIT_METHODS
        cash_panel = Plots.plot(; title=_exhibit_render_method_label(method),
            ylabel="Cash on hand (wealth), xₜ", legend=false, grid=false,
            framestyle=:box, xlims=extrema(dates), ylims=cash_limits,
            titlefontsize=10, guidefontsize=9, tickfontsize=7)
        consumption_panel = Plots.plot(; title=_exhibit_render_method_label(method),
            xlabel="Age index, t", ylabel="Consumption, cₜ",
            legend=false, grid=false, framestyle=:box,
            xlims=extrema(dates), ylims=consumption_limits,
            titlefontsize=10, guidefontsize=9, tickfontsize=7)
        for column in eachindex(path_ids)
            Plots.plot!(cash_panel, dates, cash[method][:, column];
                color=colors[column], linewidth=0.8, alpha=0.75, label=false)
            Plots.plot!(consumption_panel, dates,
                consumption[method][:, column]; color=colors[column],
                linewidth=0.8, alpha=0.75, label=false)
        end
        push!(cash_panels, cash_panel)
        push!(consumption_panels, consumption_panel)
    end
    layout = @layout [a{0.30h}; grid(2, 3)]
    support = unique(initial_cash)
    counts = [count(==(cash), initial_cash) for cash in support]
    length(unique(counts)) == 1 || throw(ArgumentError(
        "trace initial wealth groups are not balanced"))
    return Plots.plot(income_panel, cash_panels..., consumption_panels...;
        layout, size, dpi, margin=2mm, left_margin=7mm)
end

function _timing_rows(timing_payload, evaluation_paths::Int)
    construction = _method_dict(_exhibit_render_field(timing_payload,
        :construction_seconds, "timing"), "timing.construction_seconds")
    evaluation = _method_dict(_exhibit_render_field(timing_payload,
        :evaluation_seconds, "timing"), "timing.evaluation_seconds")
    end_to_end = _method_dict(_exhibit_render_field(timing_payload,
        :end_to_end_seconds, "timing"), "timing.end_to_end_seconds")
    threads = Int(_exhibit_render_field(timing_payload, :julia_threads, "timing"))
    threads >= 1 || throw(ArgumentError("timing thread count is invalid"))
    evaluation_paths >= 2 || throw(ArgumentError(
        "timing evaluation path count is invalid"))
    rows = NamedTuple[]
    for method in AR1_CONTEST_EXHIBIT_METHODS
        values = Float64.((construction[method], evaluation[method],
                          end_to_end[method]))
        all(isfinite, values) && all(values .>= 0.0) || throw(ArgumentError(
            "timing for $method is nonfinite or negative"))
        values[3] == values[1] + values[2] || throw(ArgumentError(
            "end-to-end timing for $method is not construction plus evaluation"))
        push!(rows, (; method, construction_seconds=values[1],
            evaluation_seconds=values[2], end_to_end_seconds=values[3],
            amortized_seconds_per_path=values[3] / evaluation_paths,
            julia_threads=threads, evaluation_paths))
    end
    return rows
end

function _welfare_rows(welfare_payload, model_payload)
    methods = Symbol.(collect(_exhibit_render_field(welfare_payload, :methods, "welfare")))
    methods == collect(AR1_CONTEST_EXHIBIT_METHODS) || throw(ArgumentError(
        "welfare method order is invalid"))
    utilities = Matrix{Float64}(_exhibit_render_field(welfare_payload, :utility, "welfare"))
    size(utilities, 1) >= 2 &&
        size(utilities, 2) == length(methods) && all(isfinite, utilities) ||
        throw(ArgumentError("welfare utility matrix is invalid"))
    confidence_level = Float64(_exhibit_render_field(
        welfare_payload, :confidence_level, "welfare"))
    confidence_level == 0.95 || throw(ArgumentError(
        "contest-exhibit welfare comparisons require 95% intervals"))
    beta = Float64(_exhibit_render_field(model_payload, :beta, "model"))
    gamma = Float64(_exhibit_render_field(model_payload, :gamma, "model"))
    horizon = Int(_exhibit_render_field(model_payload, :horizon, "model"))
    gamma == 1.0 || throw(ArgumentError(
        "contest-exhibit welfare rendering requires log utility"))
    expected = vec(mean(utilities; dims=1))
    rows = NamedTuple[]
    for a in 1:length(methods)-1, b in a+1:length(methods)
        method_a, method_b = methods[a], methods[b]
        comparison = paired_comparison(method_a, method_b,
            view(utilities, :, a), view(utilities, :, b);
            level=confidence_level)
        point_bps = consumption_equivalent_bps(
            expected[a], expected[b], beta, gamma, horizon)
        lower_bps = consumption_equivalent_bps(
            expected[b] + comparison.ci_lower, expected[b],
            beta, gamma, horizon)
        upper_bps = consumption_equivalent_bps(
            expected[b] + comparison.ci_upper, expected[b],
            beta, gamma, horizon)
        lower_bps <= point_bps <= upper_bps || throw(ArgumentError(
            "plug-in CEG endpoints do not contain their point estimate"))
        classification = lower_bps > 0.0 ? :method_a_resolved :
            upper_bps < 0.0 ? :method_b_resolved : :mc_unresolved
        push!(rows, (; method_a, method_b, n_paths=size(utilities, 1),
            confidence_level, point_bps, plugin_lower_bps=lower_bps,
            plugin_upper_bps=upper_bps, classification,
            interval_endpoint_kind=
                :plugin_ce_transform_of_paired_mc_utility_gap_interval,
            interval_has_nominal_ce_coverage=false))
    end
    length(rows) == 3 || throw(ErrorException(
        "contest-exhibit contest must produce three method pairs"))
    return rows
end

function _exhibit_render_trace_header(io, title, source, row_fingerprint)
    println(io, "% Generated by Ptd.AR1; do not edit by hand.")
    println(io, "% Table: ", title)
    println(io, "% Source artifact schema: ", source.artifact_schema)
    println(io, "% Source authentication fingerprint: ",
        source.authentication_fingerprint)
    println(io, "% Source result fingerprint: ", source.result_fingerprint)
    println(io, "% Source file SHA-256: ", source.artifact_file_sha256)
    println(io, "% Row-data fingerprint: ", row_fingerprint)
end

_exhibit_render_seconds(value) = @sprintf("%.6f", Float64(value))
_exhibit_render_bps(value) = @sprintf("%.3f", Float64(value))

function _timing_tex(rows, source, row_fingerprint)
    io = IOBuffer()
    _exhibit_render_trace_header(io, "AR(1) contest-exhibit method timing", source,
        row_fingerprint)
    paths = only(unique(row.evaluation_paths for row in rows))
    println(io, "\\begin{tabular}{lrrrrr}")
    println(io, "\\toprule")
    println(io, "Method & Construction (s) & $paths-path evaluation (s) & End-to-end (s) & Amortized seconds/path & Julia threads \\\\")
    println(io, "\\midrule")
    for row in rows
        println(io, _exhibit_render_method_label(row.method), " & ",
            _exhibit_render_seconds(row.construction_seconds), " & ",
            _exhibit_render_seconds(row.evaluation_seconds), " & ",
            _exhibit_render_seconds(row.end_to_end_seconds), " & ",
            _exhibit_render_seconds(row.amortized_seconds_per_path), " & ",
            row.julia_threads,
            " \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\par\\smallskip{\\footnotesize \\emph{Note:} End-to-end time is policy construction plus evaluation of the fixed common-shock paths. Amortized seconds per path divide end-to-end time by $paths and therefore include each path's equal share of policy construction. Surface queries and the 100-path illustration replay are excluded.}")
    return String(take!(io))
end

function _classification_label(row)
    row.classification === :method_a_resolved && return "A resolved over B"
    row.classification === :method_b_resolved && return "B resolved over A"
    row.classification === :mc_unresolved && return "Unresolved"
    throw(ArgumentError("unknown welfare classification"))
end

function _welfare_tex(rows, source, row_fingerprint)
    io = IOBuffer()
    _exhibit_render_trace_header(io, "AR(1) contest-exhibit pairwise welfare", source,
        row_fingerprint)
    println(io, "% Positive CEGs in bps scale method B consumption to match method A.")
    println(io, "\\begin{tabular}{llrrrl}")
    println(io, "\\toprule")
    println(io, "Method A & Method B & Point CEG (bps) & Plug-in lower (bps) & Plug-in upper (bps) & Direction \\\\")
    println(io, "\\midrule")
    for row in rows
        println(io, _exhibit_render_method_label(row.method_a), " & ",
            _exhibit_render_method_label(row.method_b), " & ", _exhibit_render_bps(row.point_bps), " & ",
            _exhibit_render_bps(row.plugin_lower_bps), " & ",
            _exhibit_render_bps(row.plugin_upper_bps), " & ",
            _classification_label(row), " \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\par\\smallskip{\\footnotesize \\emph{Note:} CEG denotes the consumption-equivalent gap. Positive point CEGs favor method A. Endpoint columns are plug-in consumption-equivalent transforms of paired 95\\% Monte Carlo utility-gap interval endpoints. They preserve the sign used for resolution, but are not asserted to be population confidence intervals or nominal consumption-equivalent confidence intervals.}")
    return String(take!(io))
end

function _exhibit_render_atomic_write(path::AbstractString, bytes::Vector{UInt8})
    destination = _validate_data_output(path)
    directory = dirname(destination)
    mkpath(directory)
    temporary_directory = mktempdir(directory;
        prefix=".ar1-exhibit-render-write-", cleanup=false)
    temporary = joinpath(temporary_directory, basename(destination))
    try
        open(temporary, "w") do io
            write(io, bytes)
        end
        read(temporary) == bytes || throw(ErrorException(
            "atomic output failed byte-for-byte read-back"))
        Base.Filesystem.rename(temporary, destination)
    finally
        ispath(temporary_directory) && rm(temporary_directory; recursive=true)
    end
    return destination
end

_exhibit_render_atomic_write(path::AbstractString, text::AbstractString) =
    _exhibit_render_atomic_write(path, Vector{UInt8}(codeunits(String(text))))

function _atomic_savefig(figure, path::AbstractString)
    destination = _validate_data_output(path)
    directory = dirname(destination)
    mkpath(directory)
    temporary_directory = mktempdir(directory;
        prefix=".ar1-exhibit-render-figure-", cleanup=false)
    temporary = joinpath(temporary_directory, basename(destination))
    try
        Plots.savefig(figure, temporary)
        bytes = read(temporary)
        length(bytes) > length(_PNG_SIGNATURE) &&
            bytes[1:length(_PNG_SIGNATURE)] == _PNG_SIGNATURE ||
            throw(ErrorException("rendered output is not a valid PNG"))
        Base.Filesystem.rename(temporary, destination)
    finally
        ispath(temporary_directory) && rm(temporary_directory; recursive=true)
    end
    return destination
end

function _exhibit_render_source_provenance(artifact_path::AbstractString, artifact;
                            require_publication::Bool)
    path = abspath(artifact_path)
    metadata = _exhibit_render_field(artifact, :metadata, "artifact")
    return (;
        artifact_schema=String(_exhibit_render_field(
            metadata, :artifact_schema, "metadata")),
        artifact_relative_path=relpath(path, projectdir()),
        artifact_bytes=filesize(path),
        artifact_file_sha256=_exhibit_render_sha256_file(path),
        scientific_fingerprint=String(_exhibit_render_field(
            metadata, :scientific_fingerprint, "metadata")),
        result_fingerprint=String(_exhibit_render_field(
            metadata, :result_fingerprint, "metadata")),
        timing_fingerprint=String(_exhibit_render_field(
            metadata, :timing_fingerprint, "metadata")),
        authentication_fingerprint=String(_exhibit_render_field(
            metadata, :authentication_fingerprint, "metadata")),
        require_publication,
    )
end

function _output_descriptor(id::Symbol, path::AbstractString,
                            content_fingerprint::String)
    absolute = abspath(path)
    return (;
        id,
        filename=basename(absolute),
        relative_path=relpath(absolute, projectdir()),
        bytes=filesize(absolute),
        sha256=_exhibit_render_sha256_file(absolute),
        content_fingerprint,
    )
end

function _exhibit_render_authentication(metadata)
    return stable_fingerprint((;
        schema=AR1_CONTEST_EXHIBIT_RENDER_AUTHENTICATION_SCHEMA,
        artifact_schema=AR1_CONTEST_EXHIBIT_RENDER_SCHEMA,
        status=_exhibit_render_field(metadata, :status, "metadata"),
        source_identity_fingerprint=String(_exhibit_render_field(
            metadata, :source_identity_fingerprint, "metadata")),
        render_configuration_fingerprint=String(_exhibit_render_field(
            metadata, :render_configuration_fingerprint, "metadata")),
        timing_rows_fingerprint=String(_exhibit_render_field(
            metadata, :timing_rows_fingerprint, "metadata")),
        welfare_rows_fingerprint=String(_exhibit_render_field(
            metadata, :welfare_rows_fingerprint, "metadata")),
        output_catalog_fingerprint=String(_exhibit_render_field(
            metadata, :output_catalog_fingerprint, "metadata")),
        result_fingerprint=String(_exhibit_render_field(
            metadata, :result_fingerprint, "metadata")),
    ))
end

function _manifest_result_fingerprint(source, render_configuration,
    timing_rows, welfare_rows, outputs)
    return stable_fingerprint((;
        source_identity_fingerprint=stable_fingerprint(source),
        render_configuration_fingerprint=
            stable_fingerprint(render_configuration),
        timing_rows_fingerprint=stable_fingerprint(timing_rows),
        welfare_rows_fingerprint=stable_fingerprint(welfare_rows),
        output_catalog_fingerprint=stable_fingerprint(outputs),
    ))
end

function _exhibit_render_write_manifest(path::AbstractString, manifest; compress::Bool)
    destination = _validate_data_output(path)
    JLD2.jldopen(destination, "w"; compress) do file
        file["manifest"] = manifest
    end
    return path
end

function _validate_output(descriptor)
    path = normpath(joinpath(projectdir(), String(_exhibit_render_field(
        descriptor, :relative_path, "output descriptor"))))
    isfile(path) || throw(ArgumentError(
        "rendered output is missing: $path"))
    filesize(path) == Int(_exhibit_render_field(descriptor, :bytes, "output descriptor")) ||
        throw(ArgumentError("rendered output byte count mismatch"))
    _exhibit_render_sha256_file(path) == String(_exhibit_render_field(
        descriptor, :sha256, "output descriptor")) ||
        throw(ArgumentError("rendered output SHA-256 mismatch"))
    id = Symbol(_exhibit_render_field(descriptor, :id, "output descriptor"))
    if id in (:policy_surface, :policy_paths)
        bytes = read(path)
        length(bytes) > length(_PNG_SIGNATURE) &&
            bytes[1:length(_PNG_SIGNATURE)] == _PNG_SIGNATURE ||
            throw(ArgumentError("rendered figure is not a valid PNG"))
    else
        source = read(path, String)
        occursin("\\begin{tabular}", source) &&
            occursin("\\end{tabular}", source) &&
            occursin("% Row-data fingerprint: " * String(_exhibit_render_field(
                descriptor, :content_fingerprint, "output descriptor")),
                source) || throw(ArgumentError(
            "rendered TeX output is not a complete tabular"))
    end
    return nothing
end

function _exhibit_render_validate_manifest(manifest;
                                   validate_source::Bool=true)
    metadata = _exhibit_render_field(manifest, :metadata, "manifest")
    String(_exhibit_render_field(metadata, :artifact_schema, "metadata")) ==
        AR1_CONTEST_EXHIBIT_RENDER_SCHEMA || throw(ArgumentError(
        "contest-exhibit render schema mismatch"))
    _exhibit_render_field(metadata, :status, "metadata") === :complete ||
        throw(ArgumentError("contest-exhibit render is incomplete"))
    completed = String(_exhibit_render_field(metadata, :completed_utc, "metadata"))
    !isempty(completed) && endswith(completed, 'Z') || throw(ArgumentError(
        "contest-exhibit render completion timestamp is invalid"))

    source = _exhibit_render_field(manifest, :source, "manifest")
    String(_exhibit_render_field(source, :artifact_schema, "source")) ==
        AR1_CONTEST_EXHIBIT_SCHEMA || throw(ArgumentError(
        "renderer source schema mismatch"))
    stable_fingerprint(source) == String(_exhibit_render_field(
        metadata, :source_identity_fingerprint, "metadata")) ||
        throw(ArgumentError("renderer source identity mismatch"))
    if validate_source
        source_path = normpath(joinpath(projectdir(), String(_exhibit_render_field(
            source, :artifact_relative_path, "source"))))
        isfile(source_path) || throw(ArgumentError(
            "renderer source artifact is missing"))
        filesize(source_path) == Int(_exhibit_render_field(
            source, :artifact_bytes, "source")) &&
            _exhibit_render_sha256_file(source_path) == String(_exhibit_render_field(
                source, :artifact_file_sha256, "source")) ||
            throw(ArgumentError("renderer source file identity mismatch"))
        loaded = load_ar1_contest_exhibit_artifact(source_path;
            require_publication=Bool(_exhibit_render_field(
                source, :require_publication, "source")),
            require_live_inputs=false)
        loaded_metadata = _exhibit_render_field(loaded, :metadata, "source artifact")
        for name in (:scientific_fingerprint, :result_fingerprint,
                     :timing_fingerprint, :authentication_fingerprint)
            String(_exhibit_render_field(loaded_metadata, name, "source metadata")) ==
                String(_exhibit_render_field(source, name, "source")) ||
                throw(ArgumentError(
                    "renderer source $name mismatch"))
        end
    end

    configuration = _exhibit_render_field(manifest, :render_configuration, "manifest")
    stable_fingerprint(configuration) == String(_exhibit_render_field(metadata,
        :render_configuration_fingerprint, "metadata")) ||
        throw(ArgumentError("render configuration fingerprint mismatch"))
    timing_rows = collect(_exhibit_render_field(manifest, :timing_rows, "manifest"))
    welfare_rows = collect(_exhibit_render_field(manifest, :welfare_rows, "manifest"))
    stable_fingerprint(timing_rows) == String(_exhibit_render_field(
        metadata, :timing_rows_fingerprint, "metadata")) ||
        throw(ArgumentError("timing-row fingerprint mismatch"))
    stable_fingerprint(welfare_rows) == String(_exhibit_render_field(
        metadata, :welfare_rows_fingerprint, "metadata")) ||
        throw(ArgumentError("welfare-row fingerprint mismatch"))
    length(timing_rows) == 3 && length(welfare_rows) == 3 &&
        Set(row.method for row in timing_rows) ==
            Set(AR1_CONTEST_EXHIBIT_METHODS) &&
        Set((row.method_a, row.method_b) for row in welfare_rows) ==
            Set((AR1_CONTEST_EXHIBIT_METHODS[a],
                 AR1_CONTEST_EXHIBIT_METHODS[b])
                for a in 1:2 for b in a+1:3) ||
        throw(ArgumentError("render manifest row counts are incomplete"))
    outputs = collect(_exhibit_render_field(manifest, :outputs, "manifest"))
    expected_ids = Set((:policy_surface, :policy_paths,
        :timings, :welfare))
    output_by_id = Dict(Symbol(_exhibit_render_field(output, :id, "output descriptor")) =>
        output for output in outputs)
    length(outputs) == 4 && length(output_by_id) == 4 &&
        Set(keys(output_by_id)) == expected_ids || throw(ArgumentError(
        "render output catalog is incomplete"))
    stable_fingerprint(outputs) == String(_exhibit_render_field(
        metadata, :output_catalog_fingerprint, "metadata")) ||
        throw(ArgumentError("render output catalog fingerprint mismatch"))
    expected_content = Dict(
        :policy_surface => stable_fingerprint((
            String(_exhibit_render_field(source, :result_fingerprint, "source")),
            _exhibit_render_field(configuration, :surface, "render configuration"))),
        :policy_paths => stable_fingerprint((
            String(_exhibit_render_field(source, :result_fingerprint, "source")),
            _exhibit_render_field(configuration, :traces, "render configuration"))),
        :timings => stable_fingerprint(timing_rows),
        :welfare => stable_fingerprint(welfare_rows),
    )
    for (id, output) in output_by_id
        String(_exhibit_render_field(output, :content_fingerprint,
            "output descriptor")) == expected_content[id] ||
            throw(ArgumentError("rendered $id content fingerprint mismatch"))
        basename(String(_exhibit_render_field(output, :relative_path,
            "output descriptor"))) == String(_exhibit_render_field(
            output, :filename, "output descriptor")) ||
            throw(ArgumentError("rendered $id filename/path mismatch"))
    end
    foreach(_validate_output, outputs)

    result_fingerprint = _manifest_result_fingerprint(
        source, configuration, timing_rows, welfare_rows, outputs)
    result_fingerprint == String(_exhibit_render_field(
        metadata, :result_fingerprint, "metadata")) ||
        throw(ArgumentError("render result fingerprint mismatch"))
    String(_exhibit_render_field(metadata, :authentication_fingerprint, "metadata")) ==
        _exhibit_render_authentication(metadata) || throw(ArgumentError(
        "render manifest authentication mismatch"))
    return manifest
end

"""Load a render manifest and verify its source plus all four outputs."""
function load_ar1_contest_exhibit_render_manifest(
    path::AbstractString; validate_source::Bool=true)
    isfile(path) || throw(ArgumentError(
        "contest-exhibit render manifest does not exist: $path"))
    manifest = JLD2.jldopen(path, "r") do file
        haskey(file, "manifest") || throw(ArgumentError(
            "JLD2 file is missing the render manifest"))
        file["manifest"]
    end
    return _exhibit_render_validate_manifest(manifest; validate_source)
end

function _atomic_save_manifest(path::AbstractString, manifest;
                               compress::Bool)
    destination = _validate_data_output(path)
    directory = dirname(destination)
    mkpath(directory)
    temporary = _temporary_jld_path(destination)
    try
        _exhibit_render_write_manifest(temporary, manifest; compress)
        JLD2.jldopen(temporary, "r") do file
            haskey(file, "manifest") || throw(ErrorException(
                "render manifest failed read-back"))
        end
        Base.Filesystem.rename(temporary, destination)
    finally
        rm(temporary; force=true)
    end
    return destination
end

"""
    render_ar1_contest_exhibits(artifact_path; ...)

Authenticate the paper's AR(1) contest-exhibit artifact, then render the policy
surface, a fixed-calibration diagnostic PNG, and two complete TeX tabular files.
The diagnostic uses a separate filename from the paper path figure. This function has no solver, RNG, or policy-evaluation entry point; display
queries and trace replay must already be present in the source artifact.
"""
function render_ar1_contest_exhibits(
    artifact_path::AbstractString;
    graph_directory::AbstractString=graphs_dir(),
    table_directory::AbstractString=tables_dir(),
    manifest_path::AbstractString=joinpath(table_directory,
        AR1_CONTEST_EXHIBIT_MANIFEST_FILENAME),
    require_publication::Bool=true,
    completed_at::DateTime=now(UTC),
    compress::Bool=true,
    surface_size::Tuple{Int,Int}=(1_650, 1_080),
    trace_size::Tuple{Int,Int}=(1_500, 1_000),
    dpi::Int=300,
    camera::Tuple{<:Real,<:Real}=(48, 28))

    # Rendering draws stored, authenticated numbers only, so it must not
    # demand a src tree identical to the contest run's; that would force a
    # full contest re-run for every figure-styling change.
    artifact = load_ar1_contest_exhibit_artifact(artifact_path;
        require_publication, require_live_inputs=false)
    # Keep this explicit even though the loader already validates: the
    # renderer's trust boundary is the runner-owned validator, not local shape
    # checks or a second artifact schema.
    validate_ar1_contest_exhibit_artifact(artifact; require_publication)
    metadata = _exhibit_render_field(artifact, :metadata, "artifact")
    scientific = _exhibit_render_field(
        artifact, :scientific_configuration, "artifact")
    model = _exhibit_render_field(scientific, :model, "scientific_configuration")
    evaluation = _exhibit_render_field(
        scientific, :evaluation, "scientific_configuration")
    surface_payload = _exhibit_render_field(artifact, :surface, "artifact")
    trace_payload = _exhibit_render_field(artifact, :traces, "artifact")
    timing_payload = _exhibit_render_field(artifact, :timing, "artifact")
    welfare_payload = _exhibit_render_field(artifact, :welfare, "artifact")

    timing_rows = _timing_rows(timing_payload,
        Int(_exhibit_render_field(evaluation, :n_paths, "evaluation")))
    welfare_rows = _welfare_rows(welfare_payload, model)
    source = _exhibit_render_source_provenance(artifact_path, artifact;
        require_publication)
    timing_fingerprint = stable_fingerprint(timing_rows)
    welfare_fingerprint = stable_fingerprint(welfare_rows)

    surface_path = joinpath(abspath(graph_directory),
        AR1_CONTEST_EXHIBIT_SURFACE_FILENAME)
    paths_path = joinpath(abspath(graph_directory),
        AR1_CONTEST_EXHIBIT_DIAGNOSTIC_PATHS_FILENAME)
    timing_path = joinpath(abspath(table_directory),
        AR1_CONTEST_EXHIBIT_TIMING_FILENAME)
    welfare_path = joinpath(abspath(table_directory),
        AR1_CONTEST_EXHIBIT_WELFARE_FILENAME)
    length(unique((surface_path, paths_path, timing_path, welfare_path,
        abspath(manifest_path)))) == 5 || throw(ArgumentError(
        "contest-exhibit render output paths must be distinct"))
    foreach(_validate_data_output,
        (surface_path, paths_path, timing_path, welfare_path, manifest_path))

    surface_figure = _surface_figure(surface_payload;
        size=surface_size, dpi, camera)
    paths_figure = _trace_figure(trace_payload; size=trace_size, dpi)
    _atomic_savefig(surface_figure, surface_path)
    _atomic_savefig(paths_figure, paths_path)
    _exhibit_render_atomic_write(timing_path,
        _timing_tex(timing_rows, source, timing_fingerprint))
    _exhibit_render_atomic_write(welfare_path,
        _welfare_tex(welfare_rows, source, welfare_fingerprint))

    render_configuration = (;
        schema="ptd-ar1-contest-exhibit-render-configuration-v3",
        source_result_fingerprint=String(_exhibit_render_field(
            metadata, :result_fingerprint, "metadata")),
        surface=(filename=basename(surface_path), size=surface_size,
            dpi, camera=(Float64(camera[1]), Float64(camera[2])),
            common_axes=true, common_camera=true, common_color_limits=true,
            per_panel_colorbar=false, equal_panel_sizes=true,
            contest_markers=false,
            slice_log_income_levels=_surface_slice_levels(surface_payload)),
        traces=(filename=basename(paths_path),
            path_ids=Int.(collect(_exhibit_render_field(
                trace_payload, :path_ids, "traces"))),
            initial_cash=Float64.(collect(_exhibit_render_field(
                trace_payload, :initial_cash, "traces"))),
            common_income=true, common_path_colors=true,
            path_legend=false,
            replayed_utility_matches_selected_welfare_paths=true,
            outside_reported_timing=Bool(_exhibit_render_field(
                trace_payload, :outside_reported_timing, "traces")),
            adds_welfare_observations=Bool(_exhibit_render_field(
                trace_payload, :adds_welfare_observations, "traces"))),
        tables=(timing=basename(timing_path),
            welfare=basename(welfare_path)),
    )
    outputs = [
        _output_descriptor(:policy_surface, surface_path,
            stable_fingerprint((source.result_fingerprint,
                render_configuration.surface))),
        _output_descriptor(:policy_paths, paths_path,
            stable_fingerprint((source.result_fingerprint,
                render_configuration.traces))),
        _output_descriptor(:timings, timing_path, timing_fingerprint),
        _output_descriptor(:welfare, welfare_path, welfare_fingerprint),
    ]
    result_fingerprint = _manifest_result_fingerprint(source,
        render_configuration, timing_rows, welfare_rows, outputs)
    metadata_without_authentication = (;
        artifact_schema=AR1_CONTEST_EXHIBIT_RENDER_SCHEMA,
        status=:complete,
        completed_utc=_exhibit_render_utc_string(completed_at),
        source_identity_fingerprint=stable_fingerprint(source),
        render_configuration_fingerprint=
            stable_fingerprint(render_configuration),
        timing_rows_fingerprint=timing_fingerprint,
        welfare_rows_fingerprint=welfare_fingerprint,
        output_catalog_fingerprint=stable_fingerprint(outputs),
        result_fingerprint,
    )
    render_metadata = merge(metadata_without_authentication,
        (authentication_fingerprint=
            _exhibit_render_authentication(metadata_without_authentication),))
    manifest = (;
        metadata=render_metadata,
        source,
        render_configuration,
        timing_rows,
        welfare_rows,
        outputs,
    )
    _exhibit_render_validate_manifest(manifest; validate_source=true)
    saved_manifest = _atomic_save_manifest(
        manifest_path, manifest; compress)
    loaded_manifest = load_ar1_contest_exhibit_render_manifest(
        saved_manifest; validate_source=true)
    return (;
        artifact_path=abspath(artifact_path),
        surface_path,
        paths_path,
        timing_path,
        welfare_path,
        manifest_path=saved_manifest,
        manifest=loaded_manifest,
    )
end

"""Reject the obsolete one-artifact paper renderer.

Figure 9 requires the heterogeneous contest-welfare artifact in addition to
the fixed-calibration exhibit artifact. Use the two-argument method.
"""
function render_ar1_paper_figures(
    ::AbstractString; kwargs...)
    throw(ArgumentError(
        "render_ar1_paper_figures requires both the contest-welfare " *
        "artifact and the fixed-calibration exhibit artifact"))
end
